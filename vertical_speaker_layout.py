#!/usr/bin/env python3
"""Vertical 1080x1920 layout: screen capture on top, speaker webcam at the bottom.

The webcam overlay is cropped out of the screen half and re-used as the bottom half,
animated colour subtitles are burned in the middle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
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
    parse_whisper_words,
    run_cmd,
    video_encoder_args,
    write_animated_ass,
)

DEFAULT_WEBCAM = (1920, 1074, 640, 366)
QR_TOP_RATIO = 0.36
QR_SIZE_RATIO = 0.46


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--words-json", type=Path, help="Whisper JSON with word timestamps.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, help="Clip length. Defaults to the whole video.")
    parser.add_argument(
        "--webcam",
        default=",".join(str(value) for value in DEFAULT_WEBCAM),
        help="Speaker box in the source as x,y,w,h.",
    )
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--subtitles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtitle-words", type=int, default=3)
    parser.add_argument("--subtitle-font", default="Arial")
    parser.add_argument("--subtitle-font-size", type=int)
    parser.add_argument("--subtitle-scale", type=int, default=132)
    parser.add_argument("--subtitle-y", type=int, help="Defaults to the middle of the frame.")
    parser.add_argument("--blur", type=int, default=24, help="Backdrop blur for the top half.")
    parser.add_argument(
        "--top-crop",
        help="Region of the screen to show on top as x,y,w,h. Defaults to the whole screen.",
    )
    parser.add_argument(
        "--top-fit",
        choices=("fit", "fill"),
        default="fit",
        help="fit keeps the whole screen with a blurred backdrop, fill zooms to the half frame.",
    )
    parser.add_argument("--encoder", choices=("auto", "videotoolbox", "libx264"), default="auto")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed of the source.")
    parser.add_argument("--music", type=Path, help="Background music, looped over the whole clip.")
    parser.add_argument("--music-volume", type=float, default=0.3)
    parser.add_argument("--music-fade", type=float, default=1.0)
    parser.add_argument("--banner", help="Coloured headline banner text.")
    parser.add_argument("--banner-start", type=float, default=10.0)
    parser.add_argument("--watermark", help="Semi-transparent bottom line, e.g. tsystem.pro.")
    parser.add_argument("--watermark-start", type=float, default=10.0)
    parser.add_argument("--qr", type=Path, help="QR image shown at the end.")
    parser.add_argument("--qr-link", default="", help="Link printed under the QR code.")
    parser.add_argument("--qr-tail", type=float, default=5.0, help="Seconds of QR at the end.")
    parser.add_argument("--outro-text", default="", help="Headline shown on the dark outro.")
    parser.add_argument("--outro-dim", type=float, default=0.16, help="Outro brightness, 0..1.")
    parser.add_argument("--logo", type=Path, help="Logo image drawn inside the top banner.")
    parser.add_argument("--money", type=Path, help="Falling-money overlay (RGBA .mov).")
    parser.add_argument(
        "--money-mode",
        choices=("outro", "banner", "always"),
        default="outro",
        help="When the money rain is visible.",
    )
    parser.add_argument("--video-bitrate", default="12M")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--keep-subtitles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--self-test", action="store_true")
    return parser


def resolve_ffmpeg(args: argparse.Namespace) -> str:
    ffmpeg = args.ffmpeg_bin or "ffmpeg"
    if "subtitles" not in available_filters(ffmpeg) and DEFAULT_FFMPEG_FULL.exists():
        ffmpeg = str(DEFAULT_FFMPEG_FULL)
    if "subtitles" not in available_filters(ffmpeg):
        raise CliError("ffmpeg without the subtitles/libass filter; pass --ffmpeg-bin.")
    return ffmpeg


def parse_webcam(value: str) -> tuple[int, int, int, int]:
    parts = [int(item) for item in value.split(",")]
    if len(parts) != 4 or any(item < 0 for item in parts) or parts[2] <= 0 or parts[3] <= 0:
        raise CliError("--webcam must be x,y,w,h with positive size.")
    return parts[0], parts[1], parts[2], parts[3]


def build_filter_graph(
    *,
    webcam: tuple[int, int, int, int],
    width: int,
    height: int,
    blur: int,
    subtitle_file: Path | None,
    top_crop: tuple[int, int, int, int] | None = None,
    top_fit: str = "fit",
    speed: float = 1.0,
    inputs: dict[str, int] | None = None,
    qr_from: float = 0.0,
    banner_start: float = 0.0,
    outro_dim: float = 0.16,
    money_mode: str = "outro",
    music_volume: float = 0.3,
    music_fade: float = 1.0,
    duration: float = 0.0,
) -> str:
    inputs = inputs or {}
    x, y, w, h = webcam
    half = height // 2
    screen = f"crop={top_crop[2]}:{top_crop[3]}:{top_crop[0]}:{top_crop[1]}" if top_crop else f"crop=iw:{y}:0:0"
    if top_fit == "fill":
        chains = [
            f"[0:v]{screen},scale={width}:{half}:force_original_aspect_ratio=increase,"
            f"crop={width}:{half}[top]"
        ]
    else:
        chains = [
            f"[0:v]{screen},split=2[screen][screenbg]",
            (
                f"[screenbg]scale={width}:{half}:force_original_aspect_ratio=increase,"
                f"crop={width}:{half},boxblur={blur}[bg]"
            ),
            f"[screen]scale={width}:-2[fit]",
            "[bg][fit]overlay=(W-w)/2:(H-h)/2[top]",
        ]
    chains.extend(
        [
            (
                f"[0:v]crop={w}:{h}:{x}:{y},scale={width}:{half}:"
                f"force_original_aspect_ratio=increase,crop={width}:{half}[cam]"
            ),
            "[top][cam]vstack=inputs=2[stacked]",
        ]
    )
    last = "stacked"
    if speed != 1.0:
        chains.append(f"[{last}]setpts=PTS/{speed}[sped]")
        last = "sped"

    outro = "qr" in inputs
    if outro:
        chains.append(
            f"[{last}]eq=brightness=-{1.0 - outro_dim:.3f}:saturation=0.25:"
            f"enable='gte(t,{qr_from:.3f})'[dark]"
        )
        last = "dark"
    if "money" in inputs:
        if money_mode == "outro":
            enable = f":enable='gte(t,{qr_from:.3f})'"
        elif money_mode == "banner":
            enable = f":enable='gte(t,{banner_start:.3f})'"
        else:
            enable = ""
        chains.append(f"[{inputs['money']}:v]scale={width}:{height}[money]")
        chains.append(f"[{last}][money]overlay=0:0:shortest=0{enable}[rain]")
        last = "rain"
    if subtitle_file is not None:
        chains.append(
            f"[{last}]subtitles=filename='{escape_filter_path(subtitle_file)}'[subbed]"
        )
        last = "subbed"
    if "logo" in inputs:
        logo_w = round(width * 0.30)
        chains.append(f"[{inputs['logo']}:v]scale={logo_w}:-1[logo]")
        chains.append(
            f"[{last}][logo]overlay=(W-w)/2:{round(height * 0.018)}:"
            f"enable='gte(t,{banner_start:.3f})'[logoed]"
        )
        last = "logoed"
    if outro:
        size = round(width * QR_SIZE_RATIO)
        chains.append(f"[{inputs['qr']}:v]scale={size}:{size}[qrimg]")
        chains.append(
            f"[{last}][qrimg]overlay=(W-w)/2:{round(height * QR_TOP_RATIO)}:"
            f"enable='gte(t,{qr_from:.3f})'[qred]"
        )
        last = "qred"
    chains.append(f"[{last}]format=yuv420p[v]")

    speech = f"[0:a]atempo={speed}[speech]" if speed != 1.0 else "[0:a]anull[speech]"
    if "music" in inputs:
        fade_start = max(0.0, duration - music_fade)
        chains.extend(
            [
                speech,
                (
                    f"[{inputs['music']}:a]volume={music_volume},"
                    f"afade=t=out:st={fade_start:.3f}:d={music_fade}[music]"
                ),
                "[speech][music]amix=inputs=2:duration=first:"
                "dropout_transition=0:normalize=0[a]",
            ]
        )
    elif speed != 1.0:
        chains.append(f"[0:a]atempo={speed}[a]")
    return ";".join(chains)


def clip_words(
    words: Sequence[TimedWord], start: float, duration: float, speed: float = 1.0
) -> list[TimedWord]:
    end = start + duration
    return [
        TimedWord(
            word.text,
            (word.start - start) / speed,
            (min(word.end, end) - start) / speed,
        )
        for word in words
        if word.end > start and word.start < end
    ]


def trim_subtitle_events(path: Path, cutoff: float) -> None:
    """Drop or shorten caption events so nothing overlaps the outro."""

    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for line in lines:
        if not line.startswith("Dialogue:"):
            kept.append(line)
            continue
        head, _, rest = line.partition(",")
        start_text, _, tail = rest.partition(",")
        end_text, _, remainder = tail.partition(",")
        if not remainder.startswith("Color"):
            kept.append(line)
            continue
        start = ass_seconds(start_text)
        if start >= cutoff:
            continue
        end = ass_seconds(end_text)
        if end > cutoff:
            line = f"{head},{start_text},{ass_time(cutoff)},{remainder}"
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def ass_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def wrap_outro(text: str, font_size: int, maximum_width: float) -> list[str]:
    """Greedy word wrap for the outro headline."""

    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word])
        if current and estimated_text_width(candidate, font_size) > maximum_width:
            lines.append(ass_escape(" ".join(current)))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(ass_escape(" ".join(current)))
    return lines


def add_branding_to_ass(
    path: Path,
    *,
    width: int,
    height: int,
    duration: float,
    banner: str | None,
    banner_start: float,
    watermark: str | None,
    watermark_start: float,
    qr_link: str,
    qr_tail: float,
    outro_text: str = "",
) -> None:
    """Append banner, watermark and QR caption events to an existing ASS file."""

    banner_size = round(width * 0.036)
    mark_size = round(width * 0.038)
    outro_size = round(width * 0.075)
    if banner:
        while estimated_text_width(banner, banner_size) > width * 0.86 and banner_size > 20:
            banner_size -= 2
    styles = [
        (
            f"Style: Banner,Arial,{banner_size},&H00FFFFFF,&H00FFFFFF,&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,{round(banner_size * 0.16)},0,8,40,40,60,1"
        ),
        (
            f"Style: Mark,Arial,{mark_size},&H00FFFFFF,&H00FFFFFF,&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,{round(mark_size * 0.14)},0,2,40,40,50,1"
        ),
        (
            f"Style: QrLink,Arial,{round(width * 0.048)},&H0060F0FF,&H0060F0FF,&H00201005,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,{round(width * 0.007)},0,2,40,40,60,1"
        ),
        (
            f"Style: Outro,Arial,{outro_size},&H00FFFFFF,&H00FFFFFF,&H0038B764,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,{round(outro_size * 0.12)},0,8,60,60,60,1"
        ),
    ]
    events: list[str] = []
    if banner:
        stop = max(banner_start, duration - qr_tail) if qr_link else duration
        events.append(
            f"Dialogue: 2,{ass_time(banner_start)},{ass_time(stop)},Banner,,0,0,0,,"
            rf"{{\fad(400,300)\an8\pos({width // 2},{round(height * 0.062)})}}"
            f"{ass_escape(banner)}"
        )
    if watermark:
        margin_x = round(width * 0.16)
        margin_y = round(height * 0.05)
        # Corners of the top and bottom halves; the subtitle band in the middle stays clear.
        spots = [
            (width - margin_x, round(height * 0.20)),
            (margin_x, round(height * 0.88)),
            (width - margin_x, round(height * 0.60)),
            (margin_x, round(height * 0.16)),
            (width - margin_x, height - margin_y),
            (margin_x, round(height * 0.62)),
        ]
        hold = 5.0
        index = 0
        moment = watermark_start
        while moment < duration - 0.2:
            spot = spots[index % len(spots)]
            events.append(
                f"Dialogue: 2,{ass_time(moment)},{ass_time(min(moment + hold, duration))},"
                "Mark,,0,0,0,,"
                rf"{{\fad(350,250)\alpha&H66&\an5\pos({spot[0]},{spot[1]})}}"
                f"{ass_escape(watermark)}"
            )
            moment += hold
            index += 1
    if outro_text:
        start = max(0.0, duration - qr_tail)
        wrapped = "\\N".join(wrap_outro(outro_text, outro_size, width * 0.84))
        events.append(
            f"Dialogue: 3,{ass_time(start)},{ass_time(duration)},Outro,,0,0,0,,"
            rf"{{\fad(500,0)\an8\pos({width // 2},{round(height * 0.12)})"
            rf"\t(0,600,1,\fscx100\fscy100)\fscx70\fscy70}}" + wrapped
        )
    if qr_link:
        qr_bottom = round(height * QR_TOP_RATIO) + round(width * QR_SIZE_RATIO)
        events.append(
            f"Dialogue: 3,{ass_time(max(0.0, duration - qr_tail))},{ass_time(duration)},"
            "QrLink,,0,0,0,,"
            rf"{{\fad(400,0)\an8\pos({width // 2},{qr_bottom + round(width * 0.012)})}}"
            f"{ass_escape(qr_link)}"
        )
    if not events:
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    events_index = lines.index("[Events]")
    lines[events_index:events_index] = [*styles, ""]
    lines.append("\n".join(events))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Path:
    if not args.input_video:
        raise CliError("Pass an input video.")
    ffmpeg = resolve_ffmpeg(args)
    source = args.input_video.expanduser().resolve()
    if not source.exists():
        raise CliError(f"Input video does not exist: {source}")
    webcam = parse_webcam(args.webcam)
    duration = args.duration
    if duration is None:
        probe = run_cmd(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)],
            capture=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"]) - args.start
    if duration <= 0:
        raise CliError("Clip duration must be greater than zero.")
    if args.speed <= 0.5 or args.speed > 2.0:
        raise CliError("--speed must be between 0.5 and 2.0 (atempo limit).")
    output_duration = duration / args.speed

    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_vertical.mp4")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vertical_layout_") as directory:
        temp_dir = Path(directory)
        subtitle_file = None
        if args.subtitles:
            if not args.words_json:
                raise CliError("Pass --words-json or --no-subtitles.")
            words = clip_words(
                parse_whisper_words(args.words_json), args.start, duration, args.speed
            )
            if args.qr:
                cutoff = max(0.0, output_duration - args.qr_tail)
                words = [word for word in words if word.start < cutoff]
            subtitle_file = write_animated_ass(
                words,
                temp_dir / "captions.ass",
                width=args.width,
                height=args.height,
                junction_y=args.subtitle_y if args.subtitle_y is not None else args.height // 2,
                font=args.subtitle_font,
                font_size=args.subtitle_font_size,
                maximum_words=args.subtitle_words,
                scale=args.subtitle_scale,
                side_margin=max(40, round(args.width * 0.06)),
            )
            print(f"Subtitle words: {len(words)}")
            if args.qr:
                trim_subtitle_events(subtitle_file, max(0.0, output_duration - args.qr_tail))
        if subtitle_file is not None and (args.banner or args.watermark or args.qr):
            add_branding_to_ass(
                subtitle_file,
                width=args.width,
                height=args.height,
                duration=output_duration,
                banner=args.banner,
                banner_start=args.banner_start,
                watermark=args.watermark,
                watermark_start=args.watermark_start,
                qr_link=args.qr_link if args.qr else "",
                qr_tail=args.qr_tail,
                outro_text=args.outro_text if args.qr else "",
            )

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-ss",
            f"{args.start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
        ]
        inputs: dict[str, int] = {}
        if args.qr:
            inputs["qr"] = len(inputs) + 1
            command.extend(["-loop", "1", "-i", str(args.qr.expanduser())])
        if args.logo:
            inputs["logo"] = len(inputs) + 1
            command.extend(["-loop", "1", "-i", str(args.logo.expanduser())])
        if args.money:
            inputs["money"] = len(inputs) + 1
            command.extend(
                ["-stream_loop", "-1", "-t", f"{output_duration:.3f}", "-i", str(args.money.expanduser())]
            )
        if args.music:
            inputs["music"] = len(inputs) + 1
            command.extend(
                ["-stream_loop", "-1", "-t", f"{output_duration:.3f}", "-i", str(args.music.expanduser())]
            )
        command.extend(
            [
                "-filter_complex",
                build_filter_graph(
                    webcam=webcam,
                    width=args.width,
                    height=args.height,
                    blur=args.blur,
                    subtitle_file=subtitle_file,
                    top_crop=parse_webcam(args.top_crop) if args.top_crop else None,
                    top_fit=args.top_fit,
                    speed=args.speed,
                    inputs=inputs,
                    qr_from=max(0.0, output_duration - args.qr_tail),
                    banner_start=args.banner_start,
                    outro_dim=args.outro_dim,
                    money_mode=args.money_mode,
                    music_volume=args.music_volume,
                    music_fade=args.music_fade,
                    duration=output_duration,
                ),
                "-map",
                "[v]",
                "-map",
                "[a]" if (args.music or args.speed != 1.0) else "0:a?",
                "-t",
                f"{output_duration:.3f}",
                *video_encoder_args(args, ffmpeg),
                "-c:a",
                "aac",
                "-b:a",
                args.audio_bitrate,
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        run_cmd(command, dry_run=args.dry_run)
        if args.keep_subtitles and subtitle_file is not None and not args.dry_run:
            shutil.copy2(subtitle_file, output.with_suffix(".ass"))
    print(f"Output: {output}")
    return output


def run_self_test() -> int:
    graph = build_filter_graph(
        webcam=DEFAULT_WEBCAM, width=1080, height=1920, blur=24, subtitle_file=None
    )
    assert "crop=640:366:1920:1074" in graph
    assert "vstack=inputs=2" in graph
    words = clip_words(
        [TimedWord("a", 1.0, 2.0), TimedWord("b", 40.0, 41.0)], start=0.5, duration=30.0
    )
    assert [word.text for word in words] == ["a"]
    assert abs(words[0].start - 0.5) < 1e-9
    fast = clip_words([TimedWord("a", 1.0, 2.0)], start=0.0, duration=30.0, speed=1.2)
    assert abs(fast[0].start - 1.0 / 1.2) < 1e-9
    branded = build_filter_graph(
        webcam=DEFAULT_WEBCAM,
        width=1080,
        height=1920,
        blur=24,
        subtitle_file=None,
        speed=1.2,
        inputs={"qr": 1, "logo": 2, "money": 3, "music": 4},
        qr_from=25.0,
        banner_start=10.0,
        duration=30.0,
    )
    assert "setpts=PTS/1.2" in branded
    assert "atempo=1.2[speech]" in branded
    assert "[4:a]volume=0.3" in branded
    assert "[3:v]scale=1080:1920[money]" in branded
    assert "[2:v]scale=324:-1[logo]" in branded
    assert "eq=brightness=-0.840" in branded
    assert "enable='gte(t,25.000)'" in branded
    with tempfile.TemporaryDirectory() as directory:
        path = write_animated_ass(
            [TimedWord("тест", 0.1, 0.5)],
            Path(directory) / "t.ass",
            width=1080,
            height=1920,
            junction_y=960,
            font="Arial",
            font_size=None,
            maximum_words=3,
            scale=132,
            side_margin=64,
        )
        add_branding_to_ass(
            path,
            width=1080,
            height=1920,
            duration=30.0,
            banner="TSystem.pro",
            banner_start=10.0,
            watermark="tsystem.pro",
            watermark_start=10.0,
            qr_link="tsystem.pro",
            qr_tail=5.0,
            outro_text="Смотри продолжение на сайте",
        )
        text = path.read_text(encoding="utf-8")
        assert "Style: Banner" in text and "Style: Mark" in text and "Style: Outro" in text
        assert "0:00:10.00,0:00:25.00,Banner" in text
        assert "0:00:25.00,0:00:30.00,QrLink" in text
        assert "0:00:25.00,0:00:30.00,Outro" in text
        assert text.count(",Mark,,0,0,0,,") == 4
        trimmed = write_animated_ass(
            [TimedWord("раз", 20.0, 21.0), TimedWord("два", 26.0, 27.0)],
            Path(directory) / "trim.ass",
            width=1080,
            height=1920,
            junction_y=960,
            font="Arial",
            font_size=None,
            maximum_words=1,
            scale=132,
            side_margin=64,
        )
        trim_subtitle_events(trimmed, 21.0)
        body = trimmed.read_text(encoding="utf-8")
        assert "два" not in body
        assert "0:00:20.00,0:00:21.00,Color0" in body
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
