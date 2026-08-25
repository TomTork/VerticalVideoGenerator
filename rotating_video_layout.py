#!/usr/bin/env python3
"""Render rotating full-screen videos with a square talking-head overlay."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from four_video_layout import (
    CliError,
    MediaInfo,
    TimedWord,
    atempo_chain,
    audio_peak_protection_filter,
    available_filters,
    normalize_token,
    parse_rate,
    parse_whisper_words,
    resolve_tools,
    transcript_tokens,
    video_encoder_args,
    write_animated_ass,
)


@dataclass(frozen=True)
class Segment:
    source_index: int
    source_start: float
    source_duration: float
    output_duration: float


@dataclass(frozen=True)
class DrumSamples:
    kick: Path
    hihat: Path
    clap: Path
    bass_guitar: Path | None = None


@dataclass(frozen=True)
class BeatPattern:
    bpm: float
    first_beat: float
    beat_interval: float
    kick_steps: tuple[int, ...]
    pattern_bars: int


DRUM_SAMPLE_EXTENSIONS = {".aif", ".aiff", ".flac", ".wav"}
VIDEO_SUFFIXES = {".m4v", ".mov", ".mp4"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cycle screen1.MP4, soap1.mov, and carpet1.mov as a vertical "
            "background. Overlay center-cropped cat1.mov in the top-right "
            "corner and use its audio."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory containing the input videos and text.txt.",
    )
    parser.add_argument("--screen-video", type=Path, default=Path("screen1.MP4"))
    parser.add_argument("--soap-video", type=Path, default=Path("soap1.mov"))
    parser.add_argument("--carpet-video", type=Path, default=Path("carpet1.mov"))
    parser.add_argument("--cat-video", type=Path, default=Path("cat1.mov"))
    parser.add_argument("--text-file", type=Path, default=Path("text.txt"))
    parser.add_argument("-o", "--output", type=Path)

    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--switch-interval",
        type=float,
        default=6.0,
        help="Seconds of output time before switching the background video.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.25,
        help="Common playback speed for every video and the cat audio.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional output duration; defaults to the sped-up cat video duration.",
    )

    parser.add_argument(
        "--overlay-size",
        "--cat1-size",
        dest="overlay_size",
        type=int,
        default=400,
        help="Width and height of the center-cropped cat overlay in pixels.",
    )
    parser.add_argument("--overlay-margin-right", type=int, default=24)
    parser.add_argument("--overlay-margin-top", type=int, default=24)
    parser.add_argument(
        "--background-crop-percent",
        type=float,
        default=10.0,
        help="Zoom/crop background videos by this percent. Does not affect cat1.",
    )

    parser.add_argument(
        "--subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--subtitle-language", default="ru")
    parser.add_argument("--subtitle-font", default="Arial")
    parser.add_argument(
        "--subtitle-font-size",
        type=int,
        default=90,
        help="Base size; intentionally larger than four_video_layout.py.",
    )
    parser.add_argument("--subtitle-scale", type=int, default=145)
    parser.add_argument("--subtitle-words", type=int, default=4)
    parser.add_argument("--subtitle-side-margin", type=int, default=80)
    parser.add_argument("--subtitle-y", type=int, default=1160)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--whisper-threads", type=int, default=0)
    parser.add_argument("--whisper-bin", default="whisper")
    parser.add_argument(
        "--whisper-json",
        type=Path,
        help="Reuse an existing Whisper JSON file instead of transcribing cat1.mov.",
    )
    parser.add_argument(
        "--keep-subtitles",
        action="store_true",
        help="Keep the generated ASS and Whisper JSON beside the result.",
    )

    parser.add_argument("--audio-volume-percent", type=float, default=100.0)
    parser.add_argument(
        "--beat-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a repeating kick/hat/clap beat from Ableton samples.",
    )
    parser.add_argument(
        "--beat-bass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a random Ableton bass guitar sample as a generated bass line.",
    )
    parser.add_argument(
        "--beat-volume-percent",
        type=float,
        default=15.0,
        help="Generated beat volume in percent: 15 is 15%% of its rendered level.",
    )
    parser.add_argument(
        "--beat-sample-dir",
        type=Path,
        action="append",
        help=(
            "Additional sample directory to search. Ableton Core/User Library "
            "drum sample paths are searched automatically."
        ),
    )
    parser.add_argument(
        "--beat-pattern-bars",
        type=int,
        default=2,
        help="Length of the repeating drum pattern in 4/4 bars.",
    )
    parser.add_argument("--beat-min-bpm", type=float, default=80.0)
    parser.add_argument("--beat-max-bpm", type=float, default=180.0)
    parser.add_argument("--beat-analysis-sample-rate", type=int, default=22050)
    parser.add_argument(
        "--dereverb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress room echo with the nara_wpe WPE dereverberation processor.",
    )
    parser.add_argument("--dereverb-taps", type=int, default=20)
    parser.add_argument("--dereverb-delay", type=int, default=3)
    parser.add_argument("--dereverb-iterations", type=int, default=3)
    parser.add_argument("--dereverb-sample-rate", type=int, default=48000)
    parser.add_argument("--dereverb-fft-size", type=int, default=1024)
    parser.add_argument("--dereverb-hop-size", type=int, default=256)
    parser.add_argument(
        "--audio-peak-protection",
        choices=("limiter", "softclip", "off"),
        default="limiter",
    )
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument(
        "--encoder",
        choices=("auto", "videotoolbox", "libx264"),
        default="auto",
    )
    parser.add_argument("--video-bitrate", default="12M")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--ffprobe-bin")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def resolve_from(base: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else base / path
    return candidate.expanduser().resolve()


def resolve_video_from(base: Path, path: Path) -> Path:
    candidate = resolve_from(base, path)
    if candidate.exists() or path.is_absolute():
        return candidate

    try:
        children = sorted(base.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return candidate

    requested_name = path.name.lower()
    requested_stem = path.stem.lower()
    for child in children:
        if child.is_file() and child.name.lower() == requested_name:
            return child.resolve()
    for child in children:
        if (
            child.is_file()
            and child.stem.lower() == requested_stem
            and child.suffix.lower() in VIDEO_SUFFIXES
        ):
            return child.resolve()
    return candidate


def run_quiet(
    command: Sequence[str],
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        message = f"Command failed: {command[0]}"
        if detail:
            message += f"\n{detail}"
        raise CliError(message) from exc


def probe_media_quiet(path: Path, ffprobe: str) -> MediaInfo:
    result = run_quiet(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ]
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


def validate_args(args: argparse.Namespace) -> None:
    if args.width < 2 or args.height < 2:
        raise CliError("--width and --height must be at least 2.")
    if args.fps <= 0:
        raise CliError("--fps must be greater than zero.")
    if args.switch_interval <= 0:
        raise CliError("--switch-interval must be greater than zero.")
    if args.speed <= 0:
        raise CliError("--speed must be greater than zero.")
    if args.duration is not None and args.duration <= 0:
        raise CliError("--duration must be greater than zero.")
    if args.overlay_size < 2:
        raise CliError("--overlay-size must be at least 2.")
    if args.overlay_margin_right < 0 or args.overlay_margin_top < 0:
        raise CliError("Overlay margins cannot be negative.")
    if not 0 <= args.background_crop_percent <= 90:
        raise CliError("--background-crop-percent must be between 0 and 90.")
    if args.overlay_size + args.overlay_margin_right > args.width:
        raise CliError("The overlay does not fit inside the output width.")
    if args.overlay_size + args.overlay_margin_top > args.height:
        raise CliError("The overlay does not fit inside the output height.")
    if args.subtitle_font_size <= 0 or args.subtitle_words <= 0:
        raise CliError("Subtitle font size and word count must be greater than zero.")
    if not 100 <= args.subtitle_scale <= 250:
        raise CliError("--subtitle-scale must be between 100 and 250.")
    if not 0 <= args.subtitle_y <= args.height:
        raise CliError("--subtitle-y must be inside the output frame.")
    if args.subtitle_side_margin < 0 or args.subtitle_side_margin * 2 >= args.width:
        raise CliError("--subtitle-side-margin leaves no usable subtitle width.")
    if not 0 <= args.audio_volume_percent <= 1000:
        raise CliError("--audio-volume-percent must be between 0 and 1000.")
    if not 0 <= args.beat_volume_percent <= 1000:
        raise CliError("--beat-volume-percent must be between 0 and 1000.")
    if args.beat_pattern_bars <= 0:
        raise CliError("--beat-pattern-bars must be greater than zero.")
    if not 30 <= args.beat_min_bpm < args.beat_max_bpm <= 300:
        raise CliError("--beat-min-bpm and --beat-max-bpm must satisfy 30 <= min < max <= 300.")
    if args.beat_analysis_sample_rate < 8000:
        raise CliError("--beat-analysis-sample-rate must be at least 8000.")
    if args.dereverb_taps <= 0 or args.dereverb_delay <= 0:
        raise CliError("Dereverb taps and delay must be greater than zero.")
    if args.dereverb_iterations <= 0:
        raise CliError("--dereverb-iterations must be greater than zero.")
    if args.dereverb_sample_rate < 8000:
        raise CliError("--dereverb-sample-rate must be at least 8000.")
    if args.dereverb_fft_size < 128 or args.dereverb_fft_size % 2:
        raise CliError("--dereverb-fft-size must be an even number of at least 128.")
    if not 0 < args.dereverb_hop_size < args.dereverb_fft_size:
        raise CliError("--dereverb-hop-size must be between 1 and the FFT size.")


def build_segments(
    *,
    output_duration: float,
    interval: float,
    speed: float,
    source_durations: Sequence[float],
) -> list[Segment]:
    positions = [0.0] * len(source_durations)
    segments: list[Segment] = []
    output_position = 0.0
    segment_index = 0

    while output_position < output_duration - 1e-9:
        source_index = segment_index % len(source_durations)
        output_slice = min(interval, output_duration - output_position)
        source_slice = output_slice * speed
        source_length = source_durations[source_index]
        source_start = positions[source_index]

        # Start a complete slice from zero instead of showing a short tail.
        if source_start + source_slice > source_length + 1e-6:
            source_start = 0.0
        segments.append(
            Segment(
                source_index=source_index,
                source_start=source_start,
                source_duration=source_slice,
                output_duration=output_slice,
            )
        )
        positions[source_index] = source_start + source_slice
        output_position += output_slice
        segment_index += 1
    return segments


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def transcript_match_score(target: str, source: str) -> float:
    if target == source and target:
        return 4.0
    similarity = difflib.SequenceMatcher(
        None,
        target,
        source,
        autojunk=False,
    ).ratio()
    if similarity >= 0.80:
        return 2.5
    if similarity >= 0.55:
        return 0.8
    return -2.0


def align_spoken_transcript_words(
    text: str,
    source_words: Sequence[TimedWord],
) -> list[TimedWord]:
    """Align spoken words while skipping transcript passages absent from audio."""

    targets = transcript_tokens(text)
    if not targets or not source_words:
        return []
    target_tokens = [normalize_token(token) for token in targets]
    source_tokens = [normalize_token(word.text) for word in source_words]
    target_count = len(targets)
    source_count = len(source_words)
    gap_penalty = -0.8

    scores = [[0.0] * (source_count + 1) for _ in range(target_count + 1)]
    choices = [[0] * (source_count + 1) for _ in range(target_count + 1)]
    for target_index in range(1, target_count + 1):
        scores[target_index][0] = target_index * gap_penalty
        choices[target_index][0] = 1
    for source_index in range(1, source_count + 1):
        scores[0][source_index] = source_index * gap_penalty
        choices[0][source_index] = 2

    for target_index in range(1, target_count + 1):
        for source_index in range(1, source_count + 1):
            match_score = transcript_match_score(
                target_tokens[target_index - 1],
                source_tokens[source_index - 1],
            )
            position_difference = abs(
                (target_index - 1) / max(1, target_count - 1)
                - (source_index - 1) / max(1, source_count - 1)
            )
            candidates = (
                scores[target_index - 1][source_index - 1]
                + match_score
                - position_difference,
                scores[target_index - 1][source_index] + gap_penalty,
                scores[target_index][source_index - 1] + gap_penalty,
            )
            choice = max(range(3), key=candidates.__getitem__)
            scores[target_index][source_index] = candidates[choice]
            choices[target_index][source_index] = choice

    target_index = target_count
    source_index = source_count
    anchors: list[tuple[int, int]] = []
    while target_index or source_index:
        choice = choices[target_index][source_index]
        if target_index and source_index and choice == 0:
            if (
                transcript_match_score(
                    target_tokens[target_index - 1],
                    source_tokens[source_index - 1],
                )
                > -2.0
            ):
                anchors.append((target_index - 1, source_index - 1))
            target_index -= 1
            source_index -= 1
        elif target_index and (not source_index or choice == 1):
            target_index -= 1
        else:
            source_index -= 1
    anchors.reverse()

    # Fill short replacement spans, but leave long unspoken transcript passages out.
    aligned_pairs: list[tuple[int, int]] = []
    boundaries = [(-1, -1), *anchors, (target_count, source_count)]
    for index in range(len(boundaries) - 1):
        current_target, current_source = boundaries[index]
        next_target, next_source = boundaries[index + 1]
        if current_target >= 0:
            aligned_pairs.append((current_target, current_source))

        target_gap = list(range(current_target + 1, next_target))
        source_gap = list(range(current_source + 1, next_source))
        maximum_target_gap = max(len(source_gap) * 3, len(source_gap) + 3)
        if not target_gap or not source_gap or len(target_gap) > maximum_target_gap:
            continue
        for gap_index, source_word_index in enumerate(source_gap):
            target_position = round(
                (gap_index + 0.5) * len(target_gap) / len(source_gap) - 0.5
            )
            target_position = max(0, min(len(target_gap) - 1, target_position))
            aligned_pairs.append((target_gap[target_position], source_word_index))

    aligned_pairs.sort(key=lambda item: item[1])
    return [
        TimedWord(
            targets[target_index],
            source_words[source_index].start,
            source_words[source_index].end,
        )
        for target_index, source_index in aligned_pairs
    ]


def run_whisper_quiet(
    *,
    video: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    source_duration: float,
) -> Path:
    if shutil.which(args.whisper_bin) is None:
        raise CliError("Whisper CLI is required for word-level subtitle animation.")
    command = [
        args.whisper_bin,
        str(video),
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
    run_quiet(command, dry_run=args.dry_run)
    output = temp_dir / f"{video.stem}.json"
    if args.dry_run:
        return output
    if output.exists():
        return output
    candidates = sorted(temp_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise CliError("Whisper completed without producing JSON output.")
    return candidates[-1]


def prepare_dereverbed_audio(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    cat_video: Path,
    temp_dir: Path,
    source_duration: float,
) -> Path | None:
    if not args.dereverb:
        return None

    raw_audio = temp_dir / "cat_audio_for_dereverb.wav"
    output_audio = temp_dir / "cat_audio_dereverbed.wav"
    run_quiet(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(cat_video),
            "-map",
            "0:a:0",
            "-t",
            f"{source_duration:.6f}",
            "-ar",
            str(args.dereverb_sample_rate),
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            str(raw_audio),
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return output_audio

    try:
        import numpy as np
        import soundfile as sf
        from nara_wpe.utils import istft, stft
        from nara_wpe.wpe import wpe
    except ImportError as exc:
        raise CliError(
            "Dereverberation dependencies are missing. Run: "
            "python3 -m pip install nara_wpe scipy soundfile"
        ) from exc

    print("Suppressing room echo with WPE...")
    samples, sample_rate = sf.read(
        raw_audio,
        always_2d=True,
        dtype="float64",
    )
    original_length = samples.shape[0]
    input_channels = samples.T
    input_rms = float(np.sqrt(np.mean(np.square(input_channels))))
    spectrum = stft(
        input_channels,
        size=args.dereverb_fft_size,
        shift=args.dereverb_hop_size,
        fading=True,
        pad=True,
    )
    del samples, input_channels
    for frequency_index in range(spectrum.shape[-1]):
        spectrum[:, :, frequency_index] = wpe(
            spectrum[:, :, frequency_index],
            taps=args.dereverb_taps,
            delay=args.dereverb_delay,
            iterations=args.dereverb_iterations,
        )
    enhanced = istft(
        spectrum,
        size=args.dereverb_fft_size,
        shift=args.dereverb_hop_size,
        fading=True,
    )[:, :original_length]
    del spectrum
    if enhanced.shape[1] < original_length:
        enhanced = np.pad(
            enhanced,
            ((0, 0), (0, original_length - enhanced.shape[1])),
        )

    output_rms = float(np.sqrt(np.mean(np.square(enhanced))))
    gain = input_rms / output_rms if output_rms > 1e-12 else 1.0
    output_peak = float(np.max(np.abs(enhanced)))
    if output_peak * gain > 0.98:
        gain = 0.98 / output_peak
    enhanced *= gain
    sf.write(
        output_audio,
        enhanced.T,
        sample_rate,
        subtype="PCM_24",
    )
    return output_audio


def default_ableton_sample_dirs() -> list[Path]:
    roots: list[Path] = []
    for app in sorted(Path("/Applications").glob("Ableton Live*.app")):
        for relative in (
            Path("Contents/App-Resources/Core Library/Samples/One Shots/Drums"),
            Path("Contents/App-Resources/Core Library/Samples"),
        ):
            candidate = app / relative
            if candidate.exists():
                roots.append(candidate)

    home = Path.home()
    for candidate in (
        home / "Music/Ableton/User Library/Samples",
        home / "Music/Ableton/User Library",
        home / "Music/Ableton/Factory Packs",
        home / "Music/Ableton/Packs",
        home / "Library/Application Support/Ableton/Library",
    ):
        if candidate.exists():
            roots.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def classify_drum_sample(path: Path) -> str | None:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if "kick" in parts:
        return "kick"
    if "hihat" in parts:
        return "hihat"
    if "clap" in parts:
        return "clap"
    if "kick" in name:
        return "kick"
    if "hihat" in name or "hi hat" in name:
        return "hihat"
    if "clap" in name or "hand clap" in name:
        return "clap"
    return None


def is_bass_guitar_sample(path: Path) -> bool:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if "bass guitar" in name or "guitar bass" in name:
        return True
    return "bass" in parts and "guitar" in name


def iter_sample_files(root: Path) -> Sequence[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in DRUM_SAMPLE_EXTENSIONS else []
    try:
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in DRUM_SAMPLE_EXTENSIONS
        ]
    except OSError:
        return []


def find_drum_sample_candidates(args: argparse.Namespace) -> dict[str, list[Path]]:
    roots = default_ableton_sample_dirs()
    for raw_root in args.beat_sample_dir or []:
        roots.append(resolve_from(args.input_dir, raw_root))

    candidates: dict[str, list[Path]] = {
        "kick": [],
        "hihat": [],
        "clap": [],
        "bass_guitar": [],
    }
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in iter_sample_files(root):
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            kind = classify_drum_sample(resolved)
            if kind is None and is_bass_guitar_sample(resolved):
                kind = "bass_guitar"
            if kind is not None:
                candidates[kind].append(resolved)

    for kind in candidates:
        candidates[kind].sort(key=lambda item: str(item).lower())
    return candidates


def choose_random_drum_samples(args: argparse.Namespace) -> DrumSamples:
    candidates = find_drum_sample_candidates(args)
    required = ["kick", "hihat", "clap"]
    if args.beat_bass:
        required.append("bass_guitar")
    missing = [kind for kind in required if not candidates[kind]]
    if missing:
        labels = {"bass_guitar": "bass guitar"}
        disable_hint = (
            "Pass --beat-sample-dir or --no-beat-bass."
            if missing == ["bass_guitar"]
            else "Pass --beat-sample-dir or --no-beat-track."
        )
        raise CliError(
            "Could not find Ableton samples for: "
            f"{', '.join(labels.get(kind, kind) for kind in missing)}. "
            f"{disable_hint}"
        )

    chooser = random.SystemRandom()
    return DrumSamples(
        kick=chooser.choice(candidates["kick"]),
        hihat=chooser.choice(candidates["hihat"]),
        clap=chooser.choice(candidates["clap"]),
        bass_guitar=(
            chooser.choice(candidates["bass_guitar"])
            if args.beat_bass
            else None
        ),
    )


def extract_beat_analysis_audio(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    audio_source: Path,
    temp_dir: Path,
    source_duration: float,
    output_duration: float,
) -> Path:
    output = temp_dir / "beat_analysis.wav"
    filters = (
        f"atrim=duration={source_duration:.6f},"
        f"{atempo_chain(args.speed)},"
        f"atrim=duration={output_duration:.6f},"
        f"aformat=sample_rates={args.beat_analysis_sample_rate}:channel_layouts=mono"
    )
    run_quiet(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_source),
            "-map",
            "0:a:0",
            "-vn",
            "-filter:a",
            filters,
            "-c:a",
            "pcm_f32le",
            str(output),
        ],
        dry_run=args.dry_run,
    )
    return output


def onset_envelope(samples: object, sample_rate: int) -> tuple[object, int]:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32), 512

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    frame_size = 2048
    hop_size = 512
    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))

    window = np.hanning(frame_size).astype(np.float32)
    previous = np.zeros(frame_size // 2 + 1, dtype=np.float32)
    flux_values: list[float] = []
    energy_values: list[float] = []
    for start in range(0, audio.size - frame_size + 1, hop_size):
        frame = audio[start : start + frame_size]
        frame = frame - float(np.mean(frame))
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        flux_values.append(float(np.maximum(magnitude - previous, 0).sum()))
        energy_values.append(float(np.sqrt(np.mean(np.square(frame)))))
        previous = magnitude

    flux = np.asarray(flux_values, dtype=np.float32)
    energy = np.asarray(energy_values, dtype=np.float32)
    if flux.size == 0:
        return np.zeros(1, dtype=np.float32), hop_size

    energy_delta = np.maximum(np.diff(np.r_[energy[:1], energy]), 0)
    envelope = np.log1p(flux) + np.log1p(energy_delta * 100.0)
    if envelope.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        envelope = np.convolve(envelope, kernel, mode="same")
    envelope = np.maximum(envelope - float(np.median(envelope)), 0)
    maximum = float(np.max(envelope))
    if maximum > 0:
        envelope = envelope / maximum
    return envelope.astype(np.float32), hop_size


def estimate_bpm(
    envelope: object,
    *,
    frame_rate: float,
    min_bpm: float,
    max_bpm: float,
) -> float:
    import numpy as np

    values = np.asarray(envelope, dtype=np.float64)
    if values.size < 8 or float(np.max(values)) <= 1e-9:
        return 120.0
    values = values - float(np.mean(values))
    autocorrelation = np.correlate(values, values, mode="full")[values.size - 1 :]
    min_lag = max(1, int(math.floor(frame_rate * 60.0 / max_bpm)))
    max_lag = min(
        autocorrelation.size - 1,
        int(math.ceil(frame_rate * 60.0 / min_bpm)),
    )
    if max_lag <= min_lag:
        return 120.0

    best_lag = min_lag
    best_score = -math.inf
    for lag in range(min_lag, max_lag + 1):
        score = float(autocorrelation[lag])
        if lag * 2 < autocorrelation.size:
            score += 0.5 * float(autocorrelation[lag * 2])
        if lag * 3 < autocorrelation.size:
            score += 0.25 * float(autocorrelation[lag * 3])
        if score > best_score:
            best_score = score
            best_lag = lag

    if not math.isfinite(best_score) or best_score <= 0:
        return 120.0
    bpm = 60.0 / (best_lag / frame_rate)
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    return bpm


def estimate_first_beat(envelope: object, *, frame_rate: float, bpm: float) -> float:
    import numpy as np

    values = np.asarray(envelope, dtype=np.float32)
    beat_frames = max(1, int(round(frame_rate * 60.0 / bpm)))
    search_frames = min(beat_frames, values.size)
    if search_frames <= 1:
        return 0.0
    scores = [
        float(values[phase::beat_frames].sum())
        for phase in range(search_frames)
    ]
    return float(max(range(search_frames), key=scores.__getitem__) / frame_rate)


def choose_kick_steps(
    envelope: object,
    *,
    frame_rate: float,
    pattern: BeatPattern,
) -> tuple[int, ...]:
    import numpy as np

    values = np.asarray(envelope, dtype=np.float32)
    steps_per_bar = 16
    total_steps = pattern.pattern_bars * steps_per_bar
    step_interval = pattern.beat_interval / 4.0
    loop_duration = pattern.pattern_bars * 4.0 * pattern.beat_interval
    scores = np.zeros(total_steps, dtype=np.float32)
    counts = np.zeros(total_steps, dtype=np.float32)

    if step_interval <= 0 or loop_duration <= 0:
        return tuple(range(0, total_steps, steps_per_bar))

    for frame_index, value in enumerate(values):
        if value <= 0:
            continue
        time = frame_index / frame_rate
        relative = (time - pattern.first_beat) % loop_duration
        step = int(round(relative / step_interval)) % total_steps
        scores[step] += float(value)
        counts[step] += 1.0

    scores = scores / np.maximum(counts, 1.0)
    maximum = float(np.max(scores))
    if maximum > 0:
        scores = scores / maximum
    threshold = float(np.percentile(scores, 70)) if scores.size else 0.0

    selected: set[int] = set()
    for bar in range(pattern.pattern_bars):
        base = bar * steps_per_bar
        selected.add(base)
        extra_added = 0
        accent_steps = [2, 3, 6, 7, 8, 10, 11, 14, 15]
        ranked = sorted(
            (base + step for step in accent_steps),
            key=lambda step: float(scores[step]),
            reverse=True,
        )
        for step in ranked:
            if float(scores[step]) < max(0.15, threshold) and extra_added > 0:
                continue
            if any(abs(step - other) <= 1 for other in selected if base <= other < base + 16):
                continue
            selected.add(step)
            extra_added += 1
            if extra_added >= 2:
                break
        if not any(base < step < base + 16 for step in selected):
            selected.add(base + 8)

    return tuple(sorted(selected))


def analyze_audio_beat(args: argparse.Namespace, analysis_audio: Path) -> BeatPattern:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise CliError(
            "Beat generation requires numpy and soundfile. Run: "
            "python3 -m pip install numpy soundfile scipy"
        ) from exc

    samples, sample_rate = sf.read(
        analysis_audio,
        always_2d=True,
        dtype="float32",
    )
    mono = np.mean(samples, axis=1)
    envelope, hop_size = onset_envelope(mono, sample_rate)
    frame_rate = sample_rate / hop_size
    bpm = estimate_bpm(
        envelope,
        frame_rate=frame_rate,
        min_bpm=args.beat_min_bpm,
        max_bpm=args.beat_max_bpm,
    )
    first_beat = estimate_first_beat(envelope, frame_rate=frame_rate, bpm=bpm)
    pattern = BeatPattern(
        bpm=bpm,
        first_beat=first_beat,
        beat_interval=60.0 / bpm,
        kick_steps=(),
        pattern_bars=args.beat_pattern_bars,
    )
    kick_steps = choose_kick_steps(envelope, frame_rate=frame_rate, pattern=pattern)
    return BeatPattern(
        bpm=pattern.bpm,
        first_beat=pattern.first_beat,
        beat_interval=pattern.beat_interval,
        kick_steps=kick_steps,
        pattern_bars=pattern.pattern_bars,
    )


def resample_audio(samples: object, source_rate: int, target_rate: int) -> object:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return audio
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), int(target_rate))
        return resample_poly(
            audio,
            target_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32)
    except Exception:
        output_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
        old_positions = np.arange(audio.shape[0], dtype=np.float32)
        new_positions = np.linspace(0, audio.shape[0] - 1, output_length)
        channels = [
            np.interp(new_positions, old_positions, audio[:, channel])
            for channel in range(audio.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32)


def trim_and_normalize_sample(
    samples: object,
    *,
    sample_rate: int,
    max_seconds: float,
) -> object:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    level = np.max(np.abs(audio), axis=1)
    active = np.flatnonzero(level > 0.002)
    if active.size:
        start = max(0, int(active[0]) - int(0.002 * sample_rate))
        end = min(audio.shape[0], int(active[-1]) + int(0.015 * sample_rate))
        audio = audio[start:end]

    maximum_length = max(1, int(max_seconds * sample_rate))
    if audio.shape[0] > maximum_length:
        audio = audio[:maximum_length].copy()
        fade_length = min(int(0.02 * sample_rate), audio.shape[0] // 4)
        if fade_length > 1:
            audio[-fade_length:] *= np.linspace(1.0, 0.0, fade_length)[:, None]

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def load_drum_sample(path: Path, *, sample_rate: int, max_seconds: float) -> object:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise CliError(
            "Beat generation requires soundfile. Run: python3 -m pip install soundfile"
        ) from exc

    try:
        samples, source_rate = sf.read(path, always_2d=True, dtype="float32")
    except RuntimeError as exc:
        raise CliError(f"Could not read drum sample: {path}") from exc
    samples = resample_audio(samples, int(source_rate), sample_rate)
    return trim_and_normalize_sample(
        samples,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    )


def repeating_step_events(
    *,
    first_beat: float,
    beat_interval: float,
    duration: float,
    pattern_bars: int,
    steps: Sequence[int],
) -> list[tuple[float, int]]:
    step_interval = beat_interval / 4.0
    loop_duration = pattern_bars * 16 * step_interval
    if step_interval <= 0 or loop_duration <= 0:
        return []

    loop_start = first_beat
    while loop_start > 0:
        loop_start -= loop_duration

    events: list[tuple[float, int]] = []
    sorted_steps = sorted(set(steps))
    while loop_start < duration:
        for step in sorted_steps:
            time = loop_start + step * step_interval
            if 0 <= time < duration:
                events.append((time, step))
        loop_start += loop_duration
    return sorted(events)


def clap_pattern_steps(pattern_bars: int) -> list[int]:
    return [bar * 16 + 12 for bar in range(pattern_bars)]


def hihat_pattern_steps(pattern_bars: int) -> range:
    return range(0, pattern_bars * 16, 4)


def bass_pattern_notes(pattern_bars: int) -> list[tuple[int, int, float, int]]:
    phrases = (
        ((0, 0, 0.56, 7), (7, 0, 0.30, 4), (11, 7, 0.43, 5), (15, -2, 0.34, 4)),
        ((0, 0, 0.56, 7), (7, 3, 0.32, 4), (11, 7, 0.43, 5), (15, -2, 0.36, 4)),
    )
    notes: list[tuple[int, int, float, int]] = []
    for bar in range(pattern_bars):
        for step, semitones, velocity, duration_steps in phrases[bar % len(phrases)]:
            notes.append((bar * 16 + step, semitones, velocity, duration_steps))
    return notes


def transpose_sample(samples: object, semitones: int) -> object:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if semitones == 0 or audio.size == 0:
        return audio.copy()

    playback_rate = 2.0 ** (semitones / 12.0)
    output_length = max(1, int(round(audio.shape[0] / playback_rate)))
    source_positions = np.arange(output_length, dtype=np.float32) * playback_rate
    source_positions = np.minimum(source_positions, audio.shape[0] - 1)
    original_positions = np.arange(audio.shape[0], dtype=np.float32)
    channels = [
        np.interp(source_positions, original_positions, audio[:, channel])
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def fit_note_duration(samples: object, *, sample_rate: int, duration: float) -> object:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).copy()
    target_length = max(1, int(round(duration * sample_rate)))
    if audio.shape[0] > target_length:
        audio = audio[:target_length].copy()

    attack = min(int(0.004 * sample_rate), audio.shape[0] // 3)
    release = min(int(0.035 * sample_rate), audio.shape[0] // 3)
    if attack > 1:
        audio[:attack] *= np.linspace(0.0, 1.0, attack)[:, None]
    if release > 1:
        audio[-release:] *= np.linspace(1.0, 0.0, release)[:, None]
    return audio


def bass_line_events(
    *,
    pattern: BeatPattern,
    duration: float,
) -> list[tuple[float, int, float, float]]:
    step_interval = pattern.beat_interval / 4.0
    loop_duration = pattern.pattern_bars * 16 * step_interval
    if step_interval <= 0 or loop_duration <= 0:
        return []

    loop_start = pattern.first_beat
    while loop_start > 0:
        loop_start -= loop_duration

    events: list[tuple[float, int, float, float]] = []
    notes = bass_pattern_notes(pattern.pattern_bars)
    while loop_start < duration:
        for step, semitones, velocity, duration_steps in notes:
            time = loop_start + step * step_interval
            if 0 <= time < duration:
                note_duration = duration_steps * step_interval * 1.08
                events.append((time, semitones, velocity, note_duration))
        loop_start += loop_duration
    return sorted(events)


def add_sample_to_track(
    track: object,
    sample: object,
    *,
    time: float,
    velocity: float,
    sample_rate: int,
) -> None:
    import numpy as np

    output = np.asarray(track)
    sound = np.asarray(sample)
    start = int(round(time * sample_rate))
    if start >= output.shape[0]:
        return
    sound_start = 0
    if start < 0:
        sound_start = -start
        start = 0
    length = min(output.shape[0] - start, sound.shape[0] - sound_start)
    if length <= 0:
        return
    output[start : start + length] += sound[sound_start : sound_start + length] * velocity


def render_drum_beat(
    *,
    samples: DrumSamples,
    pattern: BeatPattern,
    output: Path,
    duration: float,
    sample_rate: int = 48000,
) -> None:
    import numpy as np
    import soundfile as sf

    loaded = {
        "kick": load_drum_sample(samples.kick, sample_rate=sample_rate, max_seconds=0.85),
        "hihat": load_drum_sample(samples.hihat, sample_rate=sample_rate, max_seconds=0.35),
        "clap": load_drum_sample(samples.clap, sample_rate=sample_rate, max_seconds=0.70),
    }
    if samples.bass_guitar is not None:
        loaded["bass"] = load_drum_sample(
            samples.bass_guitar,
            sample_rate=sample_rate,
            max_seconds=2.40,
        )
    tail = max(np.asarray(sample).shape[0] for sample in loaded.values())
    length = int(math.ceil(duration * sample_rate)) + tail + 1
    track = np.zeros((length, 2), dtype=np.float32)

    kick_events = repeating_step_events(
        first_beat=pattern.first_beat,
        beat_interval=pattern.beat_interval,
        duration=duration,
        pattern_bars=pattern.pattern_bars,
        steps=pattern.kick_steps,
    )
    clap_steps = clap_pattern_steps(pattern.pattern_bars)
    clap_events = repeating_step_events(
        first_beat=pattern.first_beat,
        beat_interval=pattern.beat_interval,
        duration=duration,
        pattern_bars=pattern.pattern_bars,
        steps=clap_steps,
    )
    hihat_events = repeating_step_events(
        first_beat=pattern.first_beat,
        beat_interval=pattern.beat_interval,
        duration=duration,
        pattern_bars=pattern.pattern_bars,
        steps=hihat_pattern_steps(pattern.pattern_bars),
    )

    for time, step in kick_events:
        velocity = 0.95 if step % 16 == 0 else 0.78
        add_sample_to_track(
            track,
            loaded["kick"],
            time=time,
            velocity=velocity,
            sample_rate=sample_rate,
        )
    for time, _step in clap_events:
        add_sample_to_track(
            track,
            loaded["clap"],
            time=time,
            velocity=0.68,
            sample_rate=sample_rate,
        )
    for time, step in hihat_events:
        velocity = 0.32 if step % 4 == 0 else 0.24
        if step % 16 in (4, 12):
            velocity *= 0.65
        add_sample_to_track(
            track,
            loaded["hihat"],
            time=time,
            velocity=velocity,
            sample_rate=sample_rate,
        )
    if "bass" in loaded:
        note_cache: dict[tuple[int, int], object] = {}
        for time, semitones, velocity, note_duration in bass_line_events(
            pattern=pattern,
            duration=duration,
        ):
            duration_key = int(round(note_duration * sample_rate))
            cache_key = (semitones, duration_key)
            if cache_key not in note_cache:
                note_cache[cache_key] = fit_note_duration(
                    transpose_sample(loaded["bass"], semitones),
                    sample_rate=sample_rate,
                    duration=note_duration,
                )
            add_sample_to_track(
                track,
                note_cache[cache_key],
                time=time,
                velocity=velocity,
                sample_rate=sample_rate,
            )

    track = track[: int(math.ceil(duration * sample_rate))]
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track *= 0.98 / peak
    sf.write(output, track, sample_rate, subtype="PCM_24")


def format_kick_steps(steps: Sequence[int]) -> str:
    return ", ".join(f"{step // 16 + 1}:{step % 16:02d}" for step in steps)


def prepare_beat_track(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    audio_source: Path,
    temp_dir: Path,
    source_duration: float,
    output_duration: float,
) -> Path | None:
    if not args.beat_track or args.beat_volume_percent <= 0:
        return None

    output = temp_dir / "drum_beat.wav"
    if args.dry_run:
        return output

    selected_samples = choose_random_drum_samples(args)
    analysis_audio = extract_beat_analysis_audio(
        args,
        ffmpeg=ffmpeg,
        audio_source=audio_source,
        temp_dir=temp_dir,
        source_duration=source_duration,
        output_duration=output_duration,
    )
    pattern = analyze_audio_beat(args, analysis_audio)
    render_drum_beat(
        samples=selected_samples,
        pattern=pattern,
        output=output,
        duration=output_duration,
    )
    print(
        f"Beat: {pattern.bpm:.1f} BPM, loop={pattern.pattern_bars} bars, "
        f"kick steps={format_kick_steps(pattern.kick_steps)}, "
        f"volume={args.beat_volume_percent:g}%"
    )
    print(
        "Ableton samples: "
        f"kick={selected_samples.kick.name}, "
        f"hihat={selected_samples.hihat.name}, "
        f"clap={selected_samples.clap.name}"
        + (
            f", bass={selected_samples.bass_guitar.name}"
            if selected_samples.bass_guitar is not None
            else ""
        )
    )
    return output


def zoomed_dimension(value: int, percent: float) -> int:
    return int(math.ceil(value * (1.0 + percent / 100.0) / 2.0) * 2)


def build_filter_graph(
    args: argparse.Namespace,
    *,
    segments: Sequence[Segment],
    output_duration: float,
    subtitle_file: Path | None,
    audio_input_index: int,
    beat_audio_input_index: int | None,
) -> str:
    filters: list[str] = []
    source_counts = Counter(segment.source_index for segment in segments)
    source_labels: dict[int, list[str]] = {}

    for source_index, count in sorted(source_counts.items()):
        labels = [f"src{source_index}_{index}" for index in range(count)]
        source_labels[source_index] = labels
        if count == 1:
            filters.append(f"[{source_index}:v:0]null[{labels[0]}]")
        else:
            outputs = "".join(f"[{label}]" for label in labels)
            filters.append(f"[{source_index}:v:0]split={count}{outputs}")

    used_per_source = Counter()
    segment_outputs: list[str] = []
    background_width = zoomed_dimension(args.width, args.background_crop_percent)
    background_height = zoomed_dimension(args.height, args.background_crop_percent)
    for index, segment in enumerate(segments):
        use_index = used_per_source[segment.source_index]
        used_per_source[segment.source_index] += 1
        input_label = source_labels[segment.source_index][use_index]
        output_label = f"segment{index}"
        segment_outputs.append(output_label)
        filters.append(
            f"[{input_label}]"
            f"trim=start={segment.source_start:.6f}:duration={segment.source_duration:.6f},"
            f"setpts=(PTS-STARTPTS)/{args.speed:.8f},"
            f"scale={background_width}:{background_height}:"
            "force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={args.width}:{args.height},setsar=1,"
            f"fps={args.fps:.6f},format=yuv420p[{output_label}]"
        )

    concat_inputs = "".join(f"[{label}]" for label in segment_outputs)
    filters.append(
        f"{concat_inputs}concat=n={len(segment_outputs)}:v=1:a=0[background]"
    )

    cat_source_duration = output_duration * args.speed
    filters.append(
        f"[3:v:0]trim=duration={cat_source_duration:.6f},"
        f"setpts=(PTS-STARTPTS)/{args.speed:.8f},"
        f"scale={args.overlay_size}:{args.overlay_size}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={args.overlay_size}:{args.overlay_size},setsar=1,"
        f"fps={args.fps:.6f},format=yuva420p[cat]"
    )
    overlay_x = args.width - args.overlay_size - args.overlay_margin_right
    filters.append(
        f"[background][cat]overlay=x={overlay_x}:y={args.overlay_margin_top}:"
        "eof_action=pass[composed]"
    )

    current = "composed"
    if subtitle_file is not None:
        filters.append(
            f"[{current}]subtitles=filename='{escape_filter_path(subtitle_file)}'[subtitled]"
        )
        current = "subtitled"
    filters.append(
        f"[{current}]trim=duration={output_duration:.6f},"
        "setpts=PTS-STARTPTS,format=yuv420p[vout]"
    )

    protection = audio_peak_protection_filter(args.audio_peak_protection)
    filters.append(
        f"[{audio_input_index}:a:0]atrim=duration={cat_source_duration:.6f},"
        f"{atempo_chain(args.speed)},"
        f"volume={args.audio_volume_percent / 100.0:.6f},"
        "asetpts=PTS-STARTPTS[amain]"
    )
    if beat_audio_input_index is not None:
        filters.append(
            f"[{beat_audio_input_index}:a:0]"
            f"atrim=duration={output_duration:.6f},"
            f"volume={args.beat_volume_percent / 100.0:.6f},"
            "asetpts=PTS-STARTPTS[abeat]"
        )
        filters.append(
            "[amain][abeat]"
            "amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            f"{protection or 'anull'}[aout]"
        )
    else:
        filters.append(f"[amain]{protection or 'anull'}[aout]")
    return ";".join(filters)


def create_subtitles(
    args: argparse.Namespace,
    *,
    cat_video: Path,
    text_file: Path,
    temp_dir: Path,
    output_duration: float,
) -> tuple[Path | None, Path | None]:
    if not args.subtitles:
        return None, None
    if not text_file.exists():
        raise CliError(f"Subtitle text file does not exist: {text_file}")

    if args.whisper_json:
        whisper_json = resolve_from(args.input_dir, args.whisper_json)
        if not whisper_json.exists():
            raise CliError(f"Whisper JSON does not exist: {whisper_json}")
    else:
        print("Transcribing subtitles...")
        whisper_json = run_whisper_quiet(
            video=cat_video,
            temp_dir=temp_dir,
            args=args,
            source_duration=output_duration * args.speed,
        )

    source_words = [] if args.dry_run else parse_whisper_words(whisper_json)
    if not args.dry_run and not source_words:
        raise CliError("Whisper produced no word timestamps.")
    if not args.dry_run:
        transcript = text_file.read_text(encoding="utf-8-sig")
        source_words = align_spoken_transcript_words(transcript, source_words)

    output_words = [
        TimedWord(word.text, word.start / args.speed, word.end / args.speed)
        for word in source_words
        if word.start / args.speed < output_duration
    ]
    subtitle_file = write_animated_ass(
        output_words,
        temp_dir / "animated_subtitles.ass",
        width=args.width,
        height=args.height,
        junction_y=args.subtitle_y,
        font=args.subtitle_font,
        font_size=args.subtitle_font_size,
        maximum_words=args.subtitle_words,
        scale=args.subtitle_scale,
        side_margin=args.subtitle_side_margin,
    )
    print(f"Subtitle words: {len(output_words)}")
    return subtitle_file, whisper_json


def render(
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    videos: Sequence[Path],
    processed_audio: Path | None,
    beat_audio: Path | None,
    filter_graph: str,
    output: Path,
    output_duration: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-hide_banner"]
    for video in videos[:3]:
        command.extend(["-stream_loop", "-1", "-i", str(video)])
    command.extend(["-i", str(videos[3])])
    if processed_audio is not None:
        command.extend(["-i", str(processed_audio)])
    if beat_audio is not None:
        command.extend(["-i", str(beat_audio)])
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{output_duration:.6f}",
            *video_encoder_args(args, ffmpeg),
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
    run_quiet(command, dry_run=args.dry_run)


def default_output_path(input_dir: Path) -> Path:
    project_dir = Path(__file__).resolve().parent
    return (
        project_dir
        / "results"
        / "02"
        / input_dir.name
        / "rotating_video_result.mp4"
    )


def run_pipeline(args: argparse.Namespace) -> int:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.width = even(args.width)
    args.height = even(args.height)
    args.overlay_size = even(args.overlay_size)
    validate_args(args)

    ffmpeg, ffprobe = resolve_tools(args)
    if args.subtitles and "subtitles" not in available_filters(ffmpeg):
        raise CliError(
            "The selected ffmpeg has no subtitles/libass filter. "
            "Install ffmpeg-full or use --no-subtitles."
        )

    videos = [
        resolve_video_from(args.input_dir, args.screen_video),
        resolve_video_from(args.input_dir, args.soap_video),
        resolve_video_from(args.input_dir, args.carpet_video),
        resolve_video_from(args.input_dir, args.cat_video),
    ]
    for path in videos:
        if not path.exists():
            raise CliError(f"Input video does not exist: {path}")
    text_file = resolve_from(args.input_dir, args.text_file)
    infos = [probe_media_quiet(path, ffprobe) for path in videos]
    if not infos[3].has_audio:
        raise CliError("cat1.mov must contain the output audio track.")

    maximum_duration = infos[3].duration / args.speed
    output_duration = min(args.duration or maximum_duration, maximum_duration)
    segments = build_segments(
        output_duration=output_duration,
        interval=args.switch_interval,
        speed=args.speed,
        source_durations=[info.duration for info in infos[:3]],
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(args.input_dir)
    )

    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_dir = output.with_suffix("").with_name(f"{output.stem}_work")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="rotating_video_")
        temp_dir = Path(temp_owner.name)

    try:
        processed_audio = prepare_dereverbed_audio(
            args,
            ffmpeg=ffmpeg,
            cat_video=videos[3],
            temp_dir=temp_dir,
            source_duration=output_duration * args.speed,
        )
        beat_audio = prepare_beat_track(
            args,
            ffmpeg=ffmpeg,
            audio_source=processed_audio or videos[3],
            temp_dir=temp_dir,
            source_duration=output_duration * args.speed,
            output_duration=output_duration,
        )
        subtitle_file, whisper_json = create_subtitles(
            args,
            cat_video=videos[3],
            text_file=text_file,
            temp_dir=temp_dir,
            output_duration=output_duration,
        )
        audio_input_index = 4 if processed_audio is not None else 3
        beat_audio_input_index = None
        if beat_audio is not None:
            beat_audio_input_index = 4 + (1 if processed_audio is not None else 0)
        filter_graph = build_filter_graph(
            args,
            segments=segments,
            output_duration=output_duration,
            subtitle_file=subtitle_file,
            audio_input_index=audio_input_index,
            beat_audio_input_index=beat_audio_input_index,
        )
        print(
            f"Layout: {args.width}x{args.height}, overlay={args.overlay_size}px, "
            f"switch={args.switch_interval:g}s, speed={args.speed:g}x, "
            f"duration={output_duration:.3f}s"
        )
        print(f"Background segments: {len(segments)}")
        render(
            args,
            ffmpeg=ffmpeg,
            videos=videos,
            processed_audio=processed_audio,
            beat_audio=beat_audio,
            filter_graph=filter_graph,
            output=output,
            output_duration=output_duration,
        )

        if args.keep_subtitles and subtitle_file is not None and not args.dry_run:
            shutil.copy2(subtitle_file, output.with_suffix(".ass"))
            if whisper_json is not None:
                shutil.copy2(whisper_json, output.with_suffix(".whisper.json"))
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_dir}")
        print(f"Done: {output}")
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()
    return 0


def run_self_test() -> int:
    segments = build_segments(
        output_duration=42.0,
        interval=6.0,
        speed=1.0,
        source_durations=[20.0, 100.0, 100.0],
    )
    assert [segment.source_index for segment in segments] == [0, 1, 2, 0, 1, 2, 0]
    assert [segment.source_start for segment in segments if segment.source_index == 0] == [
        0.0,
        6.0,
        12.0,
    ]

    reset_segments = build_segments(
        output_duration=60.0,
        interval=6.0,
        speed=1.0,
        source_durations=[15.0, 100.0, 100.0],
    )
    screen_starts = [
        segment.source_start
        for segment in reset_segments
        if segment.source_index == 0
    ]
    assert screen_starts == [0.0, 6.0, 0.0, 6.0]
    assert math.isclose(sum(item.output_duration for item in segments), 42.0)
    assert even(401) == 400
    aligned = align_spoken_transcript_words(
        "один два пропущенный большой фрагмент три человек четыре",
        [
            TimedWord("один", 0.0, 0.2),
            TimedWord("два", 0.3, 0.5),
            TimedWord("три", 0.6, 0.8),
            TimedWord("люди", 0.9, 1.1),
            TimedWord("четыре", 1.2, 1.4),
        ],
    )
    assert [word.text for word in aligned] == [
        "один",
        "два",
        "три",
        "человек",
        "четыре",
    ]
    assert default_output_path(Path("/tmp/moroz1")).parent.name == "moroz1"
    assert build_parser().parse_args([]).speed == 1.25
    assert build_parser().parse_args([]).dereverb is True
    assert build_parser().parse_args([]).background_crop_percent == 10.0
    assert build_parser().parse_args(["--cat1-size", "520"]).overlay_size == 520
    assert build_parser().parse_args([]).beat_volume_percent == 15.0
    assert build_parser().parse_args([]).beat_bass is True
    assert classify_drum_sample(Path("/x/Drums/Kick/Kick 909.wav")) == "kick"
    assert classify_drum_sample(Path("/x/Drums/Hihat/Hihat Closed.wav")) == "hihat"
    assert classify_drum_sample(Path("/x/Drums/Clap/Hand Clap 505.aif")) == "clap"
    assert classify_drum_sample(Path("/x/Drums/Hihat/Hihat Open Vinyl Kick.wav")) == "hihat"
    assert is_bass_guitar_sample(Path("/x/Instrument/Bass/Bass Guitar Note F.aif"))
    assert clap_pattern_steps(2) == [12, 28]
    assert list(hihat_pattern_steps(2)) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert bass_pattern_notes(2) == [
        (0, 0, 0.56, 7),
        (7, 0, 0.30, 4),
        (11, 7, 0.43, 5),
        (15, -2, 0.34, 4),
        (16, 0, 0.56, 7),
        (23, 3, 0.32, 4),
        (27, 7, 0.43, 5),
        (31, -2, 0.36, 4),
    ]
    test_bass_events = bass_line_events(
        pattern=BeatPattern(120.0, 0.0, 0.5, (), 2),
        duration=1.0,
    )
    assert test_bass_events[0][3] > 0.9
    repeated = repeating_step_events(
        first_beat=0.1,
        beat_interval=0.5,
        duration=2.0,
        pattern_bars=1,
        steps=[0, 8],
    )
    assert [(round(time, 3), step) for time, step in repeated] == [(0.1, 0), (1.1, 8)]
    graph = build_filter_graph(
        build_parser().parse_args([]),
        segments=[Segment(0, 0.0, 1.0, 1.0)],
        output_duration=1.0,
        subtitle_file=None,
        audio_input_index=3,
        beat_audio_input_index=4,
    )
    assert "amix=inputs=2" in graph
    assert "[4:a:0]atrim=duration=1.000000,volume=0.150000" in graph
    assert "scale=1188:2112:force_original_aspect_ratio=increase" in graph
    assert "scale=400:400:force_original_aspect_ratio=increase" in graph
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        (base / "screen1.mov").touch()
        assert resolve_video_from(base, Path("screen1.MP4")).name == "screen1.mov"
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
