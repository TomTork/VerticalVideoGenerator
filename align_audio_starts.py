#!/usr/bin/env python3
"""Align two videos by their audio starts and pad the shorter tail with black.

The script is intentionally independent from main.py. It uses ffmpeg/ffprobe
for media IO and Python's standard library for audio-envelope correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TOOL_FFMPEG = "ffmpeg"
TOOL_FFPROBE = "ffprobe"


class CliError(RuntimeError):
    """A user-facing command line error."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_stream_index: int
    audio_stream_index: int | None


@dataclass(frozen=True)
class Alignment:
    lag_seconds: float
    trim_a: float
    trim_b: float
    score: float
    overlap_seconds: float
    target_duration: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize two camera videos by matching their audio tracks. "
            "The earlier-starting video is trimmed; the shorter aligned tail "
            "is padded with black video and silent audio."
        )
    )
    parser.add_argument("video_a", nargs="?", help="First camera video.")
    parser.add_argument("video_b", nargs="?", help="Second camera video.")
    parser.add_argument("--output-a", type=Path, help="Aligned output for the first video.")
    parser.add_argument("--output-b", type=Path, help="Aligned output for the second video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for default outputs when --output-a/--output-b are omitted.",
    )
    parser.add_argument("--suffix", default="_aligned", help="Default output suffix.")
    parser.add_argument(
        "--analysis-duration",
        type=float,
        default=160.0,
        help="Seconds from each input used for audio matching.",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=45.0,
        help="Maximum start offset to search in either direction, in seconds.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=20.0,
        help="Audio-envelope frame size for correlation.",
    )
    parser.add_argument(
        "--analysis-sample-rate",
        type=int,
        default=8000,
        help="Temporary mono WAV sample rate used for matching.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=20.0,
        help="Minimum overlapping audio seconds required for a candidate match.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF for aligned outputs.",
    )
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
    parser.add_argument("--audio-bitrate", default="192k", help="AAC audio bitrate.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without rendering.")
    parser.add_argument("--no-render", action="store_true", help="Only estimate alignment; do not write outputs.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary WAV files.")
    parser.add_argument("--ffmpeg-bin", help="ffmpeg binary. Defaults to FFMPEG_BIN or ffmpeg.")
    parser.add_argument("--ffprobe-bin", help="ffprobe binary. Defaults to FFPROBE_BIN or ffprobe.")
    parser.add_argument("--self-test", action="store_true", help="Run internal correlation tests.")
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


def run_cmd(
    cmd: Sequence[str],
    *,
    capture: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print("+", shlex.join(cmd), flush=True)
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    print("+", shlex.join(cmd), flush=True)
    try:
        return subprocess.run(
            list(cmd),
            text=True,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required tool not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            raise CliError(f"Command failed: {shlex.join(cmd)}\n{detail}") from exc
        raise CliError(f"Command failed: {shlex.join(cmd)}") from exc


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denom = float(denominator)
        if denom == 0:
            return None
        return float(numerator) / denom
    return float(value)


def ffprobe_media(path: Path) -> MediaInfo:
    result = run_cmd(
        [
            TOOL_FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    video = next(
        (
            stream
            for stream in video_streams
            if not stream.get("disposition", {}).get("attached_pic")
        ),
        video_streams[0] if video_streams else None,
    )
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not video:
        raise CliError(f"No video stream found: {path}")

    duration = None
    for raw in (
        data.get("format", {}).get("duration"),
        video.get("duration"),
        audio.get("duration") if audio else None,
    ):
        if raw not in (None, "N/A"):
            duration = float(raw)
            break
    if not duration or duration <= 0:
        raise CliError(f"Could not determine duration: {path}")

    return MediaInfo(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=parse_rate(video.get("avg_frame_rate")) or 30.0,
        has_audio=audio is not None,
        video_stream_index=int(video.get("index")),
        audio_stream_index=int(audio.get("index")) if audio else None,
    )


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.video_a or not args.video_b:
        raise CliError("Pass two input videos.")
    video_a = Path(args.video_a).expanduser().resolve()
    video_b = Path(args.video_b).expanduser().resolve()
    if not video_a.exists():
        raise CliError(f"Video does not exist: {video_a}")
    if not video_b.exists():
        raise CliError(f"Video does not exist: {video_b}")
    return video_a, video_b


def default_output_path(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    directory = output_dir.expanduser().resolve() if output_dir else input_path.parent
    return directory / f"{input_path.stem}{suffix}{input_path.suffix}"


def resolve_outputs(args: argparse.Namespace, video_a: Path, video_b: Path) -> tuple[Path, Path]:
    output_a = (
        args.output_a.expanduser().resolve()
        if args.output_a
        else default_output_path(video_a, args.output_dir, args.suffix)
    )
    output_b = (
        args.output_b.expanduser().resolve()
        if args.output_b
        else default_output_path(video_b, args.output_dir, args.suffix)
    )
    if output_a == video_a or output_b == video_b:
        raise CliError("Refusing to overwrite an input file. Choose different output paths.")
    return output_a, output_b


def fmt_seconds(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def extract_analysis_wav(
    media: MediaInfo,
    wav_path: Path,
    *,
    sample_rate: int,
    duration: float,
    dry_run: bool = False,
) -> Path:
    if not media.has_audio:
        raise CliError(f"Input has no audio stream: {media.path}")
    if media.audio_stream_index is None:
        raise CliError(f"Input has no audio stream: {media.path}")
    analysis_duration = min(media.duration, max(0.5, duration))
    run_cmd(
        [
            TOOL_FFMPEG,
            "-y",
            "-hide_banner",
            "-i",
            str(media.path),
            "-map",
            f"0:{media.audio_stream_index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-t",
            fmt_seconds(analysis_duration),
            "-f",
            "wav",
            str(wav_path),
        ],
        dry_run=dry_run,
    )
    return wav_path


def read_pcm16_mono(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1 or sample_width != 2:
            raise CliError(f"Expected mono 16-bit WAV for analysis: {path}")
        raw = wav.readframes(wav.getnframes())

    samples: list[int] = []
    for offset in range(0, len(raw) - 1, 2):
        value = int.from_bytes(raw[offset : offset + 2], byteorder="little", signed=True)
        samples.append(value)
    return samples, sample_rate


def moving_average(values: Sequence[float], radius: int) -> list[float]:
    if radius <= 0 or not values:
        return list(values)
    smoothed: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > radius * 2 + 1:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(variance, 1e-12))
    return [(value - mean) / std for value in values]


def audio_signature(samples: Sequence[int], sample_rate: int, frame_ms: float) -> list[float]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    rms_values: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        energy = sum(sample * sample for sample in frame) / len(frame)
        rms = math.sqrt(energy) / 32768.0
        rms_values.append(math.log1p(rms * 1000.0))

    envelope = zscore(moving_average(rms_values, radius=2))
    diffs = [0.0]
    diffs.extend(envelope[index] - envelope[index - 1] for index in range(1, len(envelope)))
    derivative = zscore(diffs)
    return [0.65 * env + 0.35 * diff for env, diff in zip(envelope, derivative)]


def correlation_for_lag(a: Sequence[float], b: Sequence[float], lag: int) -> tuple[float, int]:
    if lag >= 0:
        a_start = lag
        b_start = 0
    else:
        a_start = 0
        b_start = -lag
    count = min(len(a) - a_start, len(b) - b_start)
    if count <= 0:
        return -1e9, 0
    sum_a = 0.0
    sum_b = 0.0
    sum_a2 = 0.0
    sum_b2 = 0.0
    sum_ab = 0.0
    for index in range(count):
        value_a = a[a_start + index]
        value_b = b[b_start + index]
        sum_a += value_a
        sum_b += value_b
        sum_a2 += value_a * value_a
        sum_b2 += value_b * value_b
        sum_ab += value_a * value_b

    numerator = sum_ab - (sum_a * sum_b / count)
    variance_a = sum_a2 - (sum_a * sum_a / count)
    variance_b = sum_b2 - (sum_b * sum_b / count)
    denominator = math.sqrt(max(variance_a, 0.0) * max(variance_b, 0.0))
    if denominator <= 1e-12:
        return -1e9, count
    return numerator / denominator, count


def estimate_lag(
    signature_a: Sequence[float],
    signature_b: Sequence[float],
    *,
    frame_seconds: float,
    max_offset: float,
    min_overlap: float,
) -> tuple[float, float, float]:
    max_lag_frames = int(max_offset / frame_seconds)
    min_overlap_frames = max(1, int(min_overlap / frame_seconds))
    best_lag = 0
    best_score = -1e9
    best_count = 0

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        score, count = correlation_for_lag(signature_a, signature_b, lag)
        if count < min_overlap_frames:
            continue
        if score > best_score:
            best_lag = lag
            best_score = score
            best_count = count

    if best_score <= -1e8:
        raise CliError("Could not find an audio match. Increase --max-offset or lower --min-overlap.")

    lag_seconds = best_lag * frame_seconds
    overlap_seconds = best_count * frame_seconds
    return lag_seconds, best_score, overlap_seconds


def compute_alignment(
    info_a: MediaInfo,
    info_b: MediaInfo,
    signature_a: Sequence[float],
    signature_b: Sequence[float],
    *,
    frame_seconds: float,
    max_offset: float,
    min_overlap: float,
) -> Alignment:
    lag_seconds, score, overlap_seconds = estimate_lag(
        signature_a,
        signature_b,
        frame_seconds=frame_seconds,
        max_offset=max_offset,
        min_overlap=min_overlap,
    )

    # Positive lag means A contains the same audio later than B, so A started earlier.
    trim_a = max(0.0, lag_seconds)
    trim_b = max(0.0, -lag_seconds)
    remaining_a = max(0.0, info_a.duration - trim_a)
    remaining_b = max(0.0, info_b.duration - trim_b)
    target_duration = max(remaining_a, remaining_b)
    if target_duration <= 0.1:
        raise CliError("Aligned outputs would be empty.")

    return Alignment(
        lag_seconds=lag_seconds,
        trim_a=trim_a,
        trim_b=trim_b,
        score=score,
        overlap_seconds=overlap_seconds,
        target_duration=target_duration,
    )


def render_aligned_video(
    media: MediaInfo,
    output: Path,
    *,
    trim_start: float,
    target_duration: float,
    crf: int,
    preset: str,
    audio_bitrate: str,
    dry_run: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    content_duration = min(target_duration, max(0.0, media.duration - trim_start))
    pad_duration = max(0.0, target_duration - content_duration)
    width = max(2, media.width - media.width % 2)
    height = max(2, media.height - media.height % 2)
    fps = media.fps if media.fps > 0 else 30.0
    video_in = f"[0:{media.video_stream_index}]"
    if media.audio_stream_index is None:
        raise CliError(f"Input has no audio stream: {media.path}")
    audio_in = f"[0:{media.audio_stream_index}]"

    cmd = [TOOL_FFMPEG, "-y", "-hide_banner", "-i", str(media.path)]
    if pad_duration > 0.02:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-t",
                fmt_seconds(pad_duration),
                "-i",
                f"color=c=black:s={width}x{height}:r={fps:.3f}",
                "-f",
                "lavfi",
                "-t",
                fmt_seconds(pad_duration),
                "-i",
                "anullsrc=r=48000:cl=stereo",
            ]
        )
        filter_complex = (
            f"{video_in}trim=start={fmt_seconds(trim_start)}:duration={fmt_seconds(content_duration)},"
            f"setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v0];"
            "[1:v:0]format=yuv420p[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[vout];"
            f"{audio_in}atrim=start={fmt_seconds(trim_start)}:duration={fmt_seconds(content_duration)},"
            "asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo[a0];"
            "[2:a:0]aformat=sample_rates=48000:channel_layouts=stereo[a1];"
            "[a0][a1]concat=n=2:v=0:a=1[aout]"
        )
    else:
        filter_complex = (
            f"{video_in}trim=start={fmt_seconds(trim_start)}:duration={fmt_seconds(target_duration)},"
            f"setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[vout];"
            f"{audio_in}atrim=start={fmt_seconds(trim_start)}:duration={fmt_seconds(target_duration)},"
            "asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo[aout]"
        )

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-max_muxing_queue_size",
            "1024",
            str(output),
        ]
    )
    run_cmd(cmd, dry_run=dry_run)


def print_alignment_summary(alignment: Alignment, output_a: Path, output_b: Path) -> None:
    direction = "A starts earlier" if alignment.trim_a > 0 else "B starts earlier"
    if alignment.trim_a == 0 and alignment.trim_b == 0:
        direction = "starts already aligned"
    print(f"Estimated lag A-vs-B: {alignment.lag_seconds:+.3f}s ({direction})")
    print(f"Correlation score: {alignment.score:.4f}; overlap: {alignment.overlap_seconds:.3f}s")
    print(f"Trim A: {alignment.trim_a:.3f}s")
    print(f"Trim B: {alignment.trim_b:.3f}s")
    print(f"Aligned output duration: {alignment.target_duration:.3f}s")
    print(f"Output A: {output_a}")
    print(f"Output B: {output_b}")


def run_pipeline(args: argparse.Namespace) -> int:
    configure_tools(args)
    require_tool(TOOL_FFMPEG)
    require_tool(TOOL_FFPROBE)
    if args.analysis_duration <= 0:
        raise CliError("--analysis-duration must be greater than zero.")
    if args.max_offset < 0:
        raise CliError("--max-offset must be non-negative.")
    if args.frame_ms <= 0:
        raise CliError("--frame-ms must be greater than zero.")
    if args.analysis_sample_rate <= 0:
        raise CliError("--analysis-sample-rate must be greater than zero.")
    if args.min_overlap <= 0:
        raise CliError("--min-overlap must be greater than zero.")

    video_a, video_b = resolve_inputs(args)
    output_a, output_b = resolve_outputs(args, video_a, video_b)
    info_a = ffprobe_media(video_a)
    info_b = ffprobe_media(video_b)
    if not info_a.has_audio or not info_b.has_audio:
        raise CliError("Both input videos must contain an audio stream for alignment.")

    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_dir = output_a.with_suffix("")
        temp_dir = temp_dir.with_name(f"{temp_dir.name}_sync_work")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="audio_align_")
        temp_dir = Path(temp_owner.name)

    try:
        analysis_duration = args.analysis_duration + args.max_offset
        wav_a = extract_analysis_wav(
            info_a,
            temp_dir / "camera_a.wav",
            sample_rate=args.analysis_sample_rate,
            duration=analysis_duration,
            dry_run=args.dry_run,
        )
        wav_b = extract_analysis_wav(
            info_b,
            temp_dir / "camera_b.wav",
            sample_rate=args.analysis_sample_rate,
            duration=analysis_duration,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("Dry run stops after extraction commands; correlation needs actual WAV files.")
            return 0

        samples_a, sample_rate_a = read_pcm16_mono(wav_a)
        samples_b, sample_rate_b = read_pcm16_mono(wav_b)
        if sample_rate_a != sample_rate_b:
            raise CliError("Internal error: analysis WAV sample rates differ.")
        frame_seconds = args.frame_ms / 1000.0
        signature_a = audio_signature(samples_a, sample_rate_a, args.frame_ms)
        signature_b = audio_signature(samples_b, sample_rate_b, args.frame_ms)
        alignment = compute_alignment(
            info_a,
            info_b,
            signature_a,
            signature_b,
            frame_seconds=frame_seconds,
            max_offset=args.max_offset,
            min_overlap=min(args.min_overlap, args.analysis_duration),
        )
        print_alignment_summary(alignment, output_a, output_b)
        if args.no_render:
            return 0

        render_aligned_video(
            info_a,
            output_a,
            trim_start=alignment.trim_a,
            target_duration=alignment.target_duration,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            dry_run=args.dry_run,
        )
        render_aligned_video(
            info_b,
            output_b,
            trim_start=alignment.trim_b,
            target_duration=alignment.target_duration,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            dry_run=args.dry_run,
        )
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_dir}")
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()

    return 0


def run_self_test() -> int:
    base = [0.0] * 40
    pattern = [0.0, 1.0, 0.1, -0.7, 0.4, 0.0, 0.8, -0.2] * 8
    tail = [0.0] * 40
    signal_a = base + pattern + tail
    signal_b = pattern + tail
    lag, score, overlap = estimate_lag(
        signal_a,
        signal_b,
        frame_seconds=0.02,
        max_offset=2.0,
        min_overlap=0.5,
    )
    assert abs(lag - 0.8) < 1e-9, lag
    assert score > 0.1
    assert overlap > 0.5
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
