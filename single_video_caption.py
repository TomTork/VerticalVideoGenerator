#!/usr/bin/env python3
"""Add automatic animated subtitles and a tilted headline to one video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from four_video_layout import (
    CliError,
    DEFAULT_FFMPEG_FULL,
    TimedWord,
    ass_escape,
    ass_time,
    available_filters,
    escape_filter_path,
    estimated_text_width,
    parse_rate,
    parse_whisper_words,
    run_cmd,
    run_whisper,
    video_encoder_args,
    write_animated_ass,
)


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Burn automatic Whisper subtitles and a large tilted headline "
            "with a black background into one video."
        )
    )
    parser.add_argument("input_video", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--title", required=False, default="")
    parser.add_argument(
        "--subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate animated color-changing subtitles with Whisper.",
    )
    parser.add_argument("--subtitle-language", default="ru")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--whisper-threads", type=int, default=0)
    parser.add_argument("--whisper-bin", default="whisper")
    parser.add_argument("--subtitle-words", type=int, default=4)
    parser.add_argument("--subtitle-font", default="Arial")
    parser.add_argument("--subtitle-font-size", type=int)
    parser.add_argument("--subtitle-scale", type=int, default=132)
    parser.add_argument("--subtitle-y", type=int)
    parser.add_argument("--subtitle-side-margin", type=int)

    parser.add_argument("--title-font", default="Arial")
    parser.add_argument("--title-font-size", type=int)
    parser.add_argument("--title-angle", type=float, default=-4.0)
    parser.add_argument("--title-y", type=int)
    parser.add_argument(
        "--title-width-ratio",
        type=float,
        default=0.82,
        help="Maximum headline width as a share of the displayed video width.",
    )
    parser.add_argument(
        "--title-duration",
        type=float,
        help="Headline display duration. Defaults to the whole video.",
    )

    parser.add_argument(
        "--encoder",
        choices=("auto", "videotoolbox", "libx264"),
        default="auto",
    )
    parser.add_argument("--video-bitrate", default="20M")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--keep-subtitles", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--ffprobe-bin")
    parser.add_argument("--self-test", action="store_true")
    return parser


def resolve_tools(args: argparse.Namespace) -> tuple[str, str]:
    ffmpeg = args.ffmpeg_bin or os.environ.get("FFMPEG_BIN") or "ffmpeg"
    if "subtitles" not in available_filters(ffmpeg):
        if DEFAULT_FFMPEG_FULL.exists() and "subtitles" in available_filters(str(DEFAULT_FFMPEG_FULL)):
            ffmpeg = str(DEFAULT_FFMPEG_FULL)
    if "subtitles" not in available_filters(ffmpeg):
        raise CliError(
            "The selected ffmpeg has no subtitles/libass filter. "
            "Install ffmpeg-full or pass --ffmpeg-bin."
        )

    ffprobe = args.ffprobe_bin or os.environ.get("FFPROBE_BIN")
    if not ffprobe:
        sibling = Path(shutil.which(ffmpeg) or ffmpeg).with_name("ffprobe")
        ffprobe = str(sibling) if sibling.exists() else "ffprobe"
    return ffmpeg, ffprobe


def probe_video(path: Path, ffprobe: str) -> VideoInfo:
    result = run_cmd(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:"
                "stream=codec_type,width,height,avg_frame_rate,duration:"
                "stream_side_data=rotation"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise CliError(f"No video stream found: {path}")

    duration_value = data.get("format", {}).get("duration") or video.get("duration")
    if duration_value in (None, "N/A"):
        raise CliError(f"Could not determine video duration: {path}")

    width = int(video["width"])
    height = int(video["height"])
    rotation = 0
    for side_data in video.get("side_data_list") or []:
        if side_data.get("rotation") is not None:
            rotation = int(side_data["rotation"])
            break
    if abs(rotation) % 180 == 90:
        width, height = height, width

    return VideoInfo(
        duration=float(duration_value),
        fps=parse_rate(video.get("avg_frame_rate")),
        width=width,
        height=height,
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )


def wrap_title(text: str, *, font_size: int, maximum_width: float) -> str:
    paragraphs = [part.strip() for part in text.replace("\\n", "\n").splitlines()]
    wrapped: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            if wrapped and wrapped[-1]:
                wrapped.append("")
            continue
        current: list[str] = []
        for word in paragraph.split():
            candidate = " ".join([*current, word])
            if current and estimated_text_width(candidate, font_size) > maximum_width:
                wrapped.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            wrapped.append(" ".join(current))
    return "\n".join(wrapped)


def add_title_to_ass(
    path: Path,
    *,
    title: str,
    width: int,
    height: int,
    duration: float,
    font: str,
    font_size: int | None,
    angle: float,
    y: int | None,
    width_ratio: float,
) -> str:
    size = font_size or max(64, round(min(width, height) * 0.072))
    title_y = y if y is not None else round(height * 0.06) + 180
    maximum_width = width * width_ratio
    wrapped_title = wrap_title(title, font_size=size, maximum_width=maximum_width)
    box_padding = max(10, round(size * 0.15))
    side_margin = max(24, round(width * 0.05))

    lines = path.read_text(encoding="utf-8").splitlines()
    events_index = lines.index("[Events]")
    lines.insert(
        events_index - 1,
        (
            f"Style: Title,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,{angle:.2f},3,{box_padding},0,8,"
            f"{side_margin},{side_margin},0,1"
        ),
    )
    event_format_index = lines.index(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )
    lines.insert(
        event_format_index + 1,
        (
            f"Dialogue: 1,0:00:00.00,{ass_time(duration)},Title,,0,0,0,,"
            rf"{{\an8\pos({width // 2},{title_y})}}{ass_escape(wrapped_title)}"
        ),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return wrapped_title


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_video:
        raise CliError("Pass an input video.")
    if not args.title.strip():
        raise CliError("Pass headline text with --title.")
    if args.subtitle_words <= 0:
        raise CliError("--subtitle-words must be greater than zero.")
    if not 100 <= args.subtitle_scale <= 250:
        raise CliError("--subtitle-scale must be between 100 and 250.")
    if not 0.2 <= args.title_width_ratio <= 0.95:
        raise CliError("--title-width-ratio must be between 0.2 and 0.95.")
    if args.title_duration is not None and args.title_duration <= 0:
        raise CliError("--title-duration must be greater than zero.")
    if args.duration is not None and args.duration <= 0:
        raise CliError("--duration must be greater than zero.")


def render_video(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    input_video: Path,
    output: Path,
    subtitle_file: Path,
    duration: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"subtitles=filename='{escape_filter_path(subtitle_file)}',format=yuv420p",
        "-t",
        f"{duration:.6f}",
        *video_encoder_args(args, ffmpeg),
        "-c:a",
        "aac",
        "-b:a",
        args.audio_bitrate,
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run_cmd(command, dry_run=args.dry_run)


def run_pipeline(args: argparse.Namespace) -> Path:
    validate_args(args)
    ffmpeg, ffprobe = resolve_tools(args)
    input_video = args.input_video.expanduser().resolve()
    if not input_video.exists():
        raise CliError(f"Input video does not exist: {input_video}")
    info = probe_video(input_video, ffprobe)
    if args.subtitles and not info.has_audio:
        raise CliError("Automatic subtitles require an audio track.")

    duration = min(args.duration or info.duration, info.duration)
    title_duration = min(args.title_duration or duration, duration)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else input_video.with_name(f"{input_video.stem}_captioned.mp4")
    )
    subtitle_y = args.subtitle_y if args.subtitle_y is not None else round(info.height * 0.72)
    side_margin = (
        args.subtitle_side_margin
        if args.subtitle_side_margin is not None
        else max(48, round(info.width * 0.075))
    )
    if side_margin * 2 >= info.width:
        raise CliError("Subtitle side margins leave no usable width.")

    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_dir = output.with_suffix("").with_name(f"{output.stem}_work")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="single_video_caption_")
        temp_dir = Path(temp_owner.name)

    try:
        words: Sequence[TimedWord] = []
        whisper_json: Path | None = None
        if args.subtitles:
            whisper_json = run_whisper(
                main_video=input_video,
                temp_dir=temp_dir,
                args=args,
                source_duration=duration,
                dry_run=args.dry_run,
            )
            words = [] if args.dry_run else parse_whisper_words(whisper_json)
            if not args.dry_run and not words:
                raise CliError("Whisper produced no word timestamps.")
            words = [word for word in words if word.start < duration]

        subtitle_file = write_animated_ass(
            words,
            temp_dir / "captioned.ass",
            width=info.width,
            height=info.height,
            junction_y=subtitle_y,
            font=args.subtitle_font,
            font_size=args.subtitle_font_size,
            maximum_words=args.subtitle_words,
            scale=args.subtitle_scale,
            side_margin=side_margin,
        )
        wrapped_title = add_title_to_ass(
            subtitle_file,
            title=args.title.strip(),
            width=info.width,
            height=info.height,
            duration=title_duration,
            font=args.title_font,
            font_size=args.title_font_size,
            angle=args.title_angle,
            y=args.title_y,
            width_ratio=args.title_width_ratio,
        )
        print(
            f"Video: {info.width}x{info.height}, {info.fps:.3f} fps, "
            f"duration={duration:.3f}s"
        )
        print(f"Headline:\n{wrapped_title}")
        print(f"Subtitle words: {len(words)}")

        render_video(
            args,
            ffmpeg=ffmpeg,
            input_video=input_video,
            output=output,
            subtitle_file=subtitle_file,
            duration=duration,
        )

        if args.keep_subtitles and not args.dry_run:
            shutil.copy2(subtitle_file, output.with_suffix(".ass"))
            if whisper_json is not None:
                shutil.copy2(whisper_json, output.with_suffix(".whisper.json"))
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_dir}")
        print(f"Output: {output}")
        return output
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()


def create_captioned_video(
    input_video: str | Path,
    output_video: str | Path,
    title: str,
    *,
    whisper_model: str = "base",
    subtitle_language: str = "ru",
) -> Path:
    """Create one captioned video using the same subtitle colors as the four-video layout."""

    args = build_parser().parse_args(
        [
            str(input_video),
            "--output",
            str(output_video),
            "--title",
            title,
            "--whisper-model",
            whisper_model,
            "--subtitle-language",
            subtitle_language,
        ]
    )
    return run_pipeline(args)


def run_self_test() -> int:
    wrapped = wrap_title(
        "Зумеры готовят поэтический контент",
        font_size=156,
        maximum_width=1771,
    )
    assert wrapped == "Зумеры готовят\nпоэтический контент"
    with tempfile.TemporaryDirectory() as directory:
        path = write_animated_ass(
            [TimedWord("тест", 0.1, 0.5)],
            Path(directory) / "test.ass",
            width=2160,
            height=3840,
            junction_y=2765,
            font="Arial",
            font_size=None,
            maximum_words=4,
            scale=132,
            side_margin=162,
        )
        add_title_to_ass(
            path,
            title="Зумеры готовят поэтический контент",
            width=2160,
            height=3840,
            duration=10,
            font="Arial",
            font_size=None,
            angle=-4,
            y=None,
            width_ratio=0.82,
        )
        contents = path.read_text(encoding="utf-8")
        assert "Style: Title" in contents
        assert "BorderStyle" in contents
        assert "\\pos(1080,410)" in contents
        assert "Зумеры готовят\\Nпоэтический контент" in contents
        assert "Style: Color0" in contents
    print("Self-test passed.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    try:
        run_pipeline(args)
    except (CliError, json.JSONDecodeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
