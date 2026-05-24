#!/usr/bin/env python3
"""Normalize media audio loudness with ffmpeg.

By default this keeps the video stream unchanged and re-encodes the first audio
stream through loudnorm, targeting a stable average loudness for finished clips.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TOOL_FFMPEG = "ffmpeg"
TOOL_FFPROBE = "ffprobe"
FASTSTART_SUFFIXES = {".m4v", ".mov", ".mp4"}


class CliError(RuntimeError):
    """A user-facing command line error."""


@dataclass(frozen=True)
class MediaInfo:
    has_video: bool
    has_audio: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a media file's first audio stream. Video is copied "
            "unchanged unless --no-copy-video is passed."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input video or audio file.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output file. Defaults to INPUT_stem_normalized.INPUT_suffix.",
    )
    parser.add_argument(
        "--mode",
        choices=("loudnorm", "speech"),
        default="loudnorm",
        help=(
            "loudnorm targets overall LUFS; speech is kept as a compatibility alias."
        ),
    )
    parser.add_argument("--gain-db", type=float, default=0.0, help="Simple gain in dB after loudnorm.")
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--true-peak", type=float, default=-1.5)
    parser.add_argument("--lra", type=float, default=11.0)
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument(
        "--copy-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the first video stream unchanged when one exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the ffmpeg command only.")
    parser.add_argument("--ffmpeg-bin", help="ffmpeg binary. Defaults to FFMPEG_BIN or ffmpeg.")
    parser.add_argument("--ffprobe-bin", help="ffprobe binary. Defaults to FFPROBE_BIN or ffprobe.")
    parser.add_argument("--self-test", action="store_true", help="Run filter string tests.")
    return parser


def os_environ(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def configure_tools(args: argparse.Namespace) -> None:
    global TOOL_FFMPEG, TOOL_FFPROBE

    TOOL_FFMPEG = args.ffmpeg_bin or os_environ("FFMPEG_BIN") or "ffmpeg"
    if args.ffprobe_bin:
        TOOL_FFPROBE = args.ffprobe_bin
        return

    env_ffprobe = os_environ("FFPROBE_BIN")
    if env_ffprobe:
        TOOL_FFPROBE = env_ffprobe
        return

    ffmpeg_path = shutil.which(TOOL_FFMPEG) or TOOL_FFMPEG
    sibling = Path(ffmpeg_path).with_name("ffprobe")
    TOOL_FFPROBE = str(sibling) if sibling.exists() else "ffprobe"


def require_tool(name: str) -> None:
    if shutil.which(name) is None and not Path(name).exists():
        raise CliError(f"Required tool not found: {name}")


def available_ffmpeg_filters() -> set[str]:
    try:
        result = subprocess.run(
            [TOOL_FFMPEG, "-hide_banner", "-filters"],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    filters: set[str] = set()
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if "->" not in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            filters.add(parts[1])
    return filters


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_normalized{input_path.suffix}")


def validate_loudness_args(args: argparse.Namespace) -> None:
    if not -60.0 <= args.gain_db <= 60.0:
        raise CliError("--gain-db must be between -60 and 60.")
    if not -70.0 <= args.target_lufs <= -5.0:
        raise CliError("--target-lufs must be between -70 and -5.")
    if not -9.0 <= args.true_peak <= 0.0:
        raise CliError("--true-peak must be between -9 and 0.")
    if not 1.0 <= args.lra <= 50.0:
        raise CliError("--lra must be between 1 and 50.")


def validate_filter_support(args: argparse.Namespace) -> None:
    filters = available_ffmpeg_filters()
    if "loudnorm" not in filters:
        raise CliError("This ffmpeg build has no loudnorm filter.")


def ffprobe_media(path: Path) -> MediaInfo:
    try:
        result = subprocess.run(
            [
                TOOL_FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(path),
            ],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required tool not found: {TOOL_FFPROBE}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise CliError(f"ffprobe failed for {path}: {detail}") from exc

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    return MediaInfo(
        has_video=any(stream.get("codec_type") == "video" for stream in streams),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def audio_filter(
    *,
    mode: str,
    target_lufs: float,
    true_peak: float,
    lra: float,
    gain_db: float,
) -> str:
    filters = [
        "loudnorm="
        f"I={target_lufs:.1f}:"
        f"TP={true_peak:.1f}:"
        f"LRA={lra:.1f}:"
        "linear=false"
    ]
    if abs(gain_db) >= 0.001:
        filters.append(f"volume={gain_db:.2f}dB")
    filters.append("aresample=async=1:first_pts=0")
    return ",".join(filters)


def build_command(args: argparse.Namespace, input_path: Path, output_path: Path, info: MediaInfo) -> list[str]:
    cmd = [TOOL_FFMPEG, "-y", "-hide_banner", "-i", str(input_path)]
    if info.has_video and args.copy_video:
        cmd.extend(["-map", "0:v:0", "-c:v", "copy"])

    cmd.extend(
        [
            "-map",
            "0:a:0",
            "-filter:a",
            audio_filter(
                mode=args.mode,
                target_lufs=args.target_lufs,
                true_peak=args.true_peak,
                lra=args.lra,
                gain_db=args.gain_db,
            ),
            "-c:a",
            "aac",
            "-b:a",
            args.audio_bitrate,
        ]
    )
    if output_path.suffix.lower() in FASTSTART_SUFFIXES:
        cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))
    return cmd


def run_cmd(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("+", shlex.join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    try:
        subprocess.run(list(cmd), check=True)
    except FileNotFoundError as exc:
        raise CliError(f"Required tool not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise CliError(f"Command failed with exit code {exc.returncode}.") from exc


def run_pipeline(args: argparse.Namespace) -> int:
    configure_tools(args)
    require_tool(TOOL_FFMPEG)
    require_tool(TOOL_FFPROBE)
    validate_loudness_args(args)
    validate_filter_support(args)

    if args.input is None:
        raise CliError("Pass an input file.")
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else default_output_path(input_path)
    if not input_path.exists():
        raise CliError(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        raise CliError("Refusing to overwrite input file. Pass a different output path.")

    info = ffprobe_media(input_path)
    if not info.has_audio:
        raise CliError(f"No audio stream found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(build_command(args, input_path, output_path, info), dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run output path: {output_path}")
    else:
        print(f"Done: {output_path}")
    return 0


def run_self_test() -> int:
    loud = audio_filter(mode="loudnorm", target_lufs=-16.0, true_peak=-1.5, lra=11.0, gain_db=0.0)
    assert loud == "loudnorm=I=-16.0:TP=-1.5:LRA=11.0:linear=false,aresample=async=1:first_pts=0"
    speech = audio_filter(mode="speech", target_lufs=-18.0, true_peak=-2.0, lra=9.0, gain_db=11.0)
    assert "dynaudnorm" not in speech
    assert "loudnorm=I=-18.0:TP=-2.0:LRA=9.0:linear=false" in speech
    assert "volume=11.00dB" in speech
    print("Self-test passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            return run_self_test()
        return run_pipeline(args)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
