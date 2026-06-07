#!/usr/bin/env python3
"""Render a configurable four-video vertical composition with animated subtitles."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_FFMPEG_FULL = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
DEFAULT_COLORS = (
    "&H00303BFF",  # red
    "&H000AD6FF",  # yellow
    "&H000095FF",  # orange
    "&H00FFD264",  # cyan
    "&H00FF840A",  # blue
    "&H00FFFFFF",  # white
    "&H0058D130",  # green
    "&H00F25ABF",  # violet
)


class CliError(RuntimeError):
    """A command-line error that can be shown without a traceback."""


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    fps: float
    has_audio: bool


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose main, bottom, left, and jumping top-corner videos into one "
            "vertical frame. The main video's audio is primary; bottom audio is mixed quietly."
        )
    )
    parser.add_argument("main_video", type=Path, nargs="?")
    parser.add_argument("bottom_video", type=Path, nargs="?")
    parser.add_argument("left_video", type=Path, nargs="?")
    parser.add_argument("top_video", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/four_video_result.m4v"))

    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument(
        "--main-ratio",
        type=float,
        default=0.60,
        help="Height share occupied by the main video; the bottom video gets the rest.",
    )
    parser.add_argument("--main-speed", type=float, default=1.20)
    parser.add_argument(
        "--main-exposure",
        type=float,
        default=0.12,
        help="Main-video exposure adjustment in stops.",
    )
    parser.add_argument("--fps", type=float, help="Output FPS; defaults to the main video FPS.")

    parser.add_argument("--left-size", type=int, default=90)
    parser.add_argument(
        "--left-overflow",
        type=int,
        default=10,
        help="Pixels of the left square placed beyond the frame's left edge.",
    )
    parser.add_argument(
        "--left-y-offset",
        type=int,
        default=0,
        help="Vertical offset from a position centered on the main/bottom junction.",
    )

    parser.add_argument("--top-size", type=int, default=64)
    parser.add_argument("--top-margin", type=int, default=18)
    parser.add_argument("--top-horizontal-margin", type=int, default=18)
    parser.add_argument("--top-jump-interval", type=float, default=5.0)
    parser.add_argument("--top-random-offset", type=int, default=10)
    parser.add_argument("--top-jump-seed", type=int, default=20260607)

    parser.add_argument(
        "--bottom-audio-volume",
        type=float,
        default=0.05,
        help="Linear bottom-audio volume relative to the main audio.",
    )
    parser.add_argument("--audio-bitrate", default="192k")

    parser.add_argument(
        "--subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate animated ASS subtitles from the main audio.",
    )
    parser.add_argument(
        "--subtitle-text-file",
        type=Path,
        help="Optional transcript whose words are aligned to Whisper timestamps.",
    )
    parser.add_argument("--subtitle-language", default="ru")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--whisper-threads", type=int, default=0)
    parser.add_argument("--subtitle-words", type=int, default=4)
    parser.add_argument("--subtitle-font", default="Arial")
    parser.add_argument("--subtitle-font-size", type=int)
    parser.add_argument("--subtitle-scale", type=int, default=132)
    parser.add_argument(
        "--subtitle-y-offset",
        type=int,
        default=0,
        help="Vertical subtitle offset from the main/bottom junction.",
    )
    parser.add_argument(
        "--keep-subtitles",
        action="store_true",
        help="Keep the generated ASS and Whisper JSON beside the output.",
    )

    parser.add_argument(
        "--encoder",
        choices=("auto", "videotoolbox", "libx264"),
        default="auto",
    )
    parser.add_argument("--video-bitrate", default="12M")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--ffprobe-bin")
    parser.add_argument("--whisper-bin", default="whisper")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_cmd(
    cmd: Sequence[str],
    *,
    capture: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    try:
        return subprocess.run(
            list(cmd),
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required executable not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            raise CliError(f"Command failed: {shlex.join(cmd)}\n{detail}") from exc
        raise CliError(f"Command failed: {shlex.join(cmd)}") from exc


def available_filters(ffmpeg_bin: str) -> set[str]:
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-filters"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    filters: set[str] = set()
    for line in (result.stdout + result.stderr).splitlines():
        parts = line.split()
        if len(parts) >= 2 and "->" in line:
            filters.add(parts[1])
    return filters


def resolve_tools(args: argparse.Namespace) -> tuple[str, str]:
    ffmpeg = args.ffmpeg_bin or os.environ.get("FFMPEG_BIN") or "ffmpeg"
    if args.subtitles and "subtitles" not in available_filters(ffmpeg):
        if DEFAULT_FFMPEG_FULL.exists() and "subtitles" in available_filters(str(DEFAULT_FFMPEG_FULL)):
            ffmpeg = str(DEFAULT_FFMPEG_FULL)
    ffprobe = args.ffprobe_bin or os.environ.get("FFPROBE_BIN")
    if not ffprobe:
        sibling = Path(shutil.which(ffmpeg) or ffmpeg).with_name("ffprobe")
        ffprobe = str(sibling) if sibling.exists() else "ffprobe"
    return ffmpeg, ffprobe


def parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        divisor = float(denominator)
        return float(numerator) / divisor if divisor else 0.0
    return float(value)


def probe_media(path: Path, ffprobe: str) -> MediaInfo:
    result = run_cmd(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,duration",
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
        raise CliError(f"Could not determine duration: {path}")
    return MediaInfo(
        duration=float(duration_value),
        fps=parse_rate(video.get("avg_frame_rate")),
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )


def even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value - 1


def validate_args(args: argparse.Namespace) -> None:
    if args.width < 2 or args.height < 2:
        raise CliError("--width and --height must be at least 2.")
    if not 0.1 <= args.main_ratio <= 0.9:
        raise CliError("--main-ratio must be between 0.1 and 0.9.")
    if args.main_speed <= 0:
        raise CliError("--main-speed must be greater than zero.")
    if not -3.0 <= args.main_exposure <= 3.0:
        raise CliError("--main-exposure must be between -3 and 3.")
    if args.left_size < 2 or args.top_size < 2:
        raise CliError("Overlay sizes must be at least 2 pixels.")
    if args.left_overflow < 0:
        raise CliError("--left-overflow cannot be negative.")
    if args.top_jump_interval <= 0:
        raise CliError("--top-jump-interval must be greater than zero.")
    if args.top_random_offset < 0:
        raise CliError("--top-random-offset cannot be negative.")
    if not 0.0 <= args.bottom_audio_volume <= 1.0:
        raise CliError("--bottom-audio-volume must be between 0 and 1.")
    if args.subtitle_words <= 0:
        raise CliError("--subtitle-words must be greater than zero.")
    if not 100 <= args.subtitle_scale <= 250:
        raise CliError("--subtitle-scale must be between 100 and 250.")
    if args.duration is not None and args.duration <= 0:
        raise CliError("--duration must be greater than zero.")


def resolve_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise CliError(f"{label} video does not exist: {resolved}")
    return resolved


def atempo_chain(speed: float) -> str:
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def nested_time_expression(values: Sequence[int], interval: float) -> str:
    if not values:
        return "0"
    expression = str(values[-1])
    for index in range(len(values) - 2, -1, -1):
        threshold = (index + 1) * interval
        expression = f"if(lt(t\\,{threshold:.3f})\\,{values[index]}\\,{expression})"
    return expression


def jumping_positions(
    *,
    duration: float,
    width: int,
    size: int,
    margin_x: int,
    margin_y: int,
    interval: float,
    offset: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    count = max(1, math.ceil(duration / interval))
    rng = random.Random(seed)
    x_values: list[int] = []
    y_values: list[int] = []
    left_base = margin_x
    right_base = width - size - margin_x
    for index in range(count):
        base_x = right_base if index % 2 == 0 else left_base
        x_values.append(max(-size + 1, min(width - 1, base_x + rng.randint(-offset, offset))))
        # Every jump starts from margin_y; negative offsets can only move the square upward.
        y_values.append(max(-size + 1, margin_y + rng.randint(-offset, 0)))
    return x_values, y_values


def normalize_token(text: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "", text.lower().replace("ё", "е"))


def transcript_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", re.sub(r"\s+", " ", text.strip()))


def parse_whisper_words(path: Path) -> list[TimedWord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[TimedWord] = []
    for segment in data.get("segments", []):
        for item in segment.get("words") or []:
            text = str(item.get("word") or "").strip()
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if text and end > start:
                words.append(TimedWord(text, start, end))
    return words


def align_transcript_words(text: str, source_words: Sequence[TimedWord]) -> list[TimedWord]:
    targets = transcript_tokens(text)
    if not targets or not source_words:
        return []
    target_normalized = [normalize_token(token) for token in targets]
    source_normalized = [normalize_token(word.text) for word in source_words]
    matcher = difflib.SequenceMatcher(None, target_normalized, source_normalized, autojunk=False)
    mapping: dict[int, int] = {}
    for tag, target_start, target_end, source_start, source_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(target_end - target_start):
                mapping[target_start + offset] = source_start + offset

    aligned: list[TimedWord] = []
    source_last = len(source_words) - 1
    target_last = max(1, len(targets) - 1)
    previous_index = 0
    for index, token in enumerate(targets):
        source_index = mapping.get(index)
        if source_index is None:
            source_index = round(index / target_last * source_last)
        source_index = max(previous_index, min(source_last, source_index))
        source = source_words[source_index]
        aligned.append(TimedWord(token, source.start, source.end))
        previous_index = source_index
    return aligned


def run_whisper(
    *,
    main_video: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    source_duration: float,
    dry_run: bool,
) -> Path:
    if shutil.which(args.whisper_bin) is None:
        raise CliError("Whisper CLI is required for word-level subtitle animation.")
    command = [
        args.whisper_bin,
        str(main_video),
        "--model",
        args.whisper_model,
        "--device",
        args.whisper_device,
        "--output_format",
        "json",
        "--output_dir",
        str(temp_dir),
        "--word_timestamps",
        "True",
        "--verbose",
        "False",
        "--clip_timestamps",
        f"0,{source_duration:.6f}",
    ]
    if args.subtitle_language:
        command.extend(["--language", args.subtitle_language])
    if args.whisper_threads > 0:
        command.extend(["--threads", str(args.whisper_threads)])
    if args.whisper_device == "cpu":
        command.extend(["--fp16", "False"])
    run_cmd(command, dry_run=dry_run)
    output = temp_dir / f"{main_video.stem}.json"
    if dry_run:
        return output
    if not output.exists():
        candidates = sorted(temp_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise CliError("Whisper completed without producing JSON output.")
        return candidates[-1]
    return output


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    cs = centiseconds % 100
    total_seconds = centiseconds // 100
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def group_words(words: Sequence[TimedWord], maximum: int) -> list[list[TimedWord]]:
    groups: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for word in words:
        current.append(word)
        if len(current) >= maximum or re.search(r"[.!?…]$", word.text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def animated_ass_text(
    words: Sequence[TimedWord],
    *,
    cue_start: float,
    style_name: str,
    scale: int,
) -> str:
    parts: list[str] = []
    for word in words:
        start_ms = max(0, round((word.start - cue_start) * 1000))
        end_ms = max(start_ms + 80, round((word.end - cue_start) * 1000))
        midpoint = max(start_ms + 40, round((start_ms + end_ms) / 2))
        parts.append(
            rf"{{\fscx100\fscy100"
            rf"\t({start_ms},{midpoint},1,\fscx{scale}\fscy{scale})"
            rf"\t({midpoint},{end_ms},1,\fscx100\fscy100)}}"
            f"{ass_escape(word.text)}"
            rf"{{\r{style_name}}}"
        )
    return " ".join(parts)


def write_animated_ass(
    words: Sequence[TimedWord],
    path: Path,
    *,
    width: int,
    height: int,
    junction_y: int,
    font: str,
    font_size: int | None,
    maximum_words: int,
    scale: int,
) -> Path:
    size = font_size or max(42, round(height * 0.034))
    outline = max(4, round(size * 0.09))
    y = junction_y
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
    ]
    for index, color in enumerate(DEFAULT_COLORS):
        lines.append(
            f"Style: Color{index},{font},{size},{color},{color},&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,{outline},2,5,40,40,0,1"
        )
    lines.extend(
        [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    groups = group_words(words, maximum_words)
    for index, group in enumerate(groups):
        style_name = f"Color{index % len(DEFAULT_COLORS)}"
        cue_start = max(0.0, group[0].start)
        cue_end = group[-1].end + 0.18
        if index + 1 < len(groups):
            cue_end = min(cue_end, groups[index + 1][0].start)
        text = animated_ass_text(
            group,
            cue_start=cue_start,
            style_name=style_name,
            scale=scale,
        )
        lines.append(
            f"Dialogue: 0,{ass_time(cue_start)},{ass_time(cue_end)},"
            f"{style_name},,0,0,0,,"
            rf"{{\an5\pos({width // 2},{y})}}{text}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def build_filter_graph(
    args: argparse.Namespace,
    *,
    duration: float,
    fps: float,
    main_height: int,
    bottom_height: int,
    left_y: int,
    subtitle_file: Path | None,
    bottom_has_audio: bool,
) -> str:
    x_values, y_values = jumping_positions(
        duration=duration,
        width=args.width,
        size=args.top_size,
        margin_x=args.top_horizontal_margin,
        margin_y=args.top_margin,
        interval=args.top_jump_interval,
        offset=args.top_random_offset,
        seed=args.top_jump_seed,
    )
    top_x = nested_time_expression(x_values, args.top_jump_interval)
    top_y = nested_time_expression(y_values, args.top_jump_interval)
    left_x = -args.left_overflow

    filters = [
        (
            f"[0:v:0]setpts=(PTS-STARTPTS)/{args.main_speed:.8f},"
            f"exposure=exposure={args.main_exposure:.4f},"
            f"scale={args.width}:{main_height}:flags=lanczos,setsar=1,"
            f"fps={fps:.6f},format=yuv420p[vmain]"
        ),
        (
            f"[1:v:0]setpts=PTS-STARTPTS,"
            f"scale={args.width}:{bottom_height}:flags=lanczos,setsar=1,"
            f"fps={fps:.6f},format=yuv420p[vbottom]"
        ),
        (
            f"[2:v:0]setpts=PTS-STARTPTS,"
            f"scale={args.left_size}:{args.left_size}:flags=lanczos,setsar=1,"
            f"fps={fps:.6f},format=yuva420p[vleft]"
        ),
        (
            f"[3:v:0]setpts=PTS-STARTPTS,"
            f"scale={args.top_size}:{args.top_size}:flags=lanczos,setsar=1,"
            f"fps={fps:.6f},format=yuva420p[vtop]"
        ),
        f"color=c=black:s={args.width}x{args.height}:r={fps:.6f}:d={duration:.6f}[canvas]",
        "[canvas][vmain]overlay=x=0:y=0:eof_action=pass[with_main]",
        f"[with_main][vbottom]overlay=x=0:y={main_height}:eof_action=repeat[with_bottom]",
        f"[with_bottom][vleft]overlay=x={left_x}:y={left_y}:eof_action=repeat[with_left]",
        (
            f"[with_left][vtop]overlay=x='{top_x}':y='{top_y}':"
            "eval=frame:eof_action=repeat[composed]"
        ),
    ]
    current = "composed"
    if subtitle_file is not None:
        filters.append(
            f"[{current}]subtitles=filename='{escape_filter_path(subtitle_file)}'[subtitled]"
        )
        current = "subtitled"
    filters.append(f"[{current}]trim=duration={duration:.6f},setpts=PTS-STARTPTS,format=yuv420p[vout]")

    main_audio = (
        f"[0:a:0]{atempo_chain(args.main_speed)},"
        f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[amain]"
    )
    filters.append(main_audio)
    if bottom_has_audio and args.bottom_audio_volume > 0:
        filters.extend(
            [
                (
                    f"[1:a:0]volume={args.bottom_audio_volume:.6f},"
                    f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[abottom]"
                ),
                (
                    "[amain][abottom]amix=inputs=2:duration=first:dropout_transition=0:"
                    "normalize=0,alimiter=limit=0.95[aout]"
                ),
            ]
        )
    else:
        filters.append("[amain]anull[aout]")
    return ";".join(filters)


def choose_encoder(args: argparse.Namespace, ffmpeg: str) -> str:
    if args.encoder != "auto":
        return args.encoder
    if sys.platform == "darwin":
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if "h264_videotoolbox" in result.stdout + result.stderr:
            return "videotoolbox"
    return "libx264"


def render(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    inputs: Sequence[Path],
    filter_graph: str,
    duration: float,
) -> None:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-hide_banner", "-i", str(inputs[0])]
    for path in inputs[1:]:
        command.extend(["-stream_loop", "-1", "-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{duration:.6f}",
        ]
    )
    encoder = choose_encoder(args, ffmpeg)
    if encoder == "videotoolbox":
        command.extend(
            [
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                args.video_bitrate,
                "-maxrate",
                args.video_bitrate,
                "-allow_sw",
                "1",
            ]
        )
    else:
        command.extend(
            ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf)]
        )
    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            args.audio_bitrate,
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
            "-shortest",
            str(output),
        ]
    )
    run_cmd(command, dry_run=args.dry_run)
    print(f"Done: {output}")


def run_pipeline(args: argparse.Namespace) -> int:
    validate_args(args)
    if not all((args.main_video, args.bottom_video, args.left_video, args.top_video)):
        raise CliError("Pass all four videos: main, bottom, left, and top.")
    ffmpeg, ffprobe = resolve_tools(args)
    inputs = [
        resolve_input(args.main_video, "Main"),
        resolve_input(args.bottom_video, "Bottom"),
        resolve_input(args.left_video, "Left"),
        resolve_input(args.top_video, "Top"),
    ]
    infos = [probe_media(path, ffprobe) for path in inputs]
    if not infos[0].has_audio:
        raise CliError("The main video must contain the primary audio track.")

    width = even(args.width)
    height = even(args.height)
    args.width = width
    args.height = height
    main_height = even(round(height * args.main_ratio))
    bottom_height = height - main_height
    if main_height < 2 or bottom_height < 2:
        raise CliError("--main-ratio leaves an invalid main or bottom area.")
    duration = args.duration or (infos[0].duration / args.main_speed)
    duration = min(duration, infos[0].duration / args.main_speed)
    fps = args.fps or infos[0].fps or 30.0
    left_y = round(main_height - args.left_size / 2 + args.left_y_offset)
    junction_y = main_height + args.subtitle_y_offset

    output = args.output.expanduser().resolve()
    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_dir = output.with_suffix("").with_name(f"{output.stem}_work")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="four_video_")
        temp_dir = Path(temp_owner.name)

    try:
        subtitle_file: Path | None = None
        whisper_json: Path | None = None
        if args.subtitles:
            if "subtitles" not in available_filters(ffmpeg):
                raise CliError(
                    "The selected ffmpeg has no subtitles/libass filter. "
                    "Install ffmpeg-full or pass --no-subtitles."
                )
            whisper_json = run_whisper(
                main_video=inputs[0],
                temp_dir=temp_dir,
                args=args,
                source_duration=duration * args.main_speed,
                dry_run=args.dry_run,
            )
            source_words = [] if args.dry_run else parse_whisper_words(whisper_json)
            if not args.dry_run and not source_words:
                raise CliError("Whisper produced no word timestamps.")
            if args.subtitle_text_file:
                transcript_path = args.subtitle_text_file.expanduser().resolve()
                if not transcript_path.exists():
                    raise CliError(f"Subtitle transcript does not exist: {transcript_path}")
                transcript = transcript_path.read_text(encoding="utf-8-sig")
                source_words = align_transcript_words(transcript, source_words)
            output_words = [
                TimedWord(word.text, word.start / args.main_speed, word.end / args.main_speed)
                for word in source_words
                if word.start / args.main_speed < duration
            ]
            subtitle_file = write_animated_ass(
                output_words,
                temp_dir / "animated_subtitles.ass",
                width=width,
                height=height,
                junction_y=junction_y,
                font=args.subtitle_font,
                font_size=args.subtitle_font_size,
                maximum_words=args.subtitle_words,
                scale=args.subtitle_scale,
            )
            print(f"Subtitle words: {len(output_words)}")

        filter_graph = build_filter_graph(
            args,
            duration=duration,
            fps=fps,
            main_height=main_height,
            bottom_height=bottom_height,
            left_y=left_y,
            subtitle_file=subtitle_file,
            bottom_has_audio=infos[1].has_audio,
        )
        print(
            f"Layout: {width}x{height}, main={main_height}px ({args.main_ratio:.0%}), "
            f"bottom={bottom_height}px, duration={duration:.3f}s, fps={fps:.3f}"
        )
        render(
            args,
            ffmpeg=ffmpeg,
            inputs=inputs,
            filter_graph=filter_graph,
            duration=duration,
        )

        if args.keep_subtitles and subtitle_file is not None and not args.dry_run:
            sidecar = output.with_suffix(".ass")
            shutil.copy2(subtitle_file, sidecar)
            if whisper_json is not None:
                shutil.copy2(whisper_json, output.with_suffix(".whisper.json"))
            print(f"Subtitle sidecars: {sidecar}")
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_dir}")
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()
    return 0


def run_self_test() -> int:
    assert atempo_chain(1.2) == "atempo=1.20000000"
    assert atempo_chain(4.0) == "atempo=2.00000000,atempo=2.00000000"
    x_values, y_values = jumping_positions(
        duration=16,
        width=1080,
        size=64,
        margin_x=18,
        margin_y=18,
        interval=5,
        offset=10,
        seed=7,
    )
    assert len(x_values) == 4
    assert x_values[0] > 500 and x_values[1] < 100
    assert all(y <= 18 for y in y_values)
    expression = nested_time_expression([100, 200, 300], 5)
    assert "lt(t\\,5.000)" in expression and expression.endswith("))")
    aligned = align_transcript_words(
        "один два три",
        [
            TimedWord("один", 0.0, 0.2),
            TimedWord("два", 0.3, 0.5),
            TimedWord("три", 0.6, 0.9),
        ],
    )
    assert [item.text for item in aligned] == ["один", "два", "три"]
    with tempfile.TemporaryDirectory() as directory:
        path = write_animated_ass(
            aligned,
            Path(directory) / "test.ass",
            width=1080,
            height=1920,
            junction_y=1152,
            font="Arial",
            font_size=64,
            maximum_words=4,
            scale=132,
        )
        contents = path.read_text(encoding="utf-8")
        assert "\\pos(540,1152)" in contents
        assert "\\t(" in contents
        assert "Style: Color7" in contents
    print("Self-test passed.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    try:
        return run_pipeline(args)
    except (CliError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
