#!/usr/bin/env python3
"""Two-camera video editor driven by pauses in the audio track.

The script intentionally depends only on the Python standard library. Video,
audio, denoise, VHS styling, subtitle burn-in, and splitting are delegated to
ffmpeg/ffprobe so the terminal tool remains portable and easy to inspect.
"""

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
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CAM_MAIN = "main"
CAM_SECOND = "second"
TOOL_FFMPEG = "ffmpeg"
TOOL_FFPROBE = "ffprobe"
SUBTITLE_VERTICAL_RAISE_PX = 150


class CliError(RuntimeError):
    """A user-facing command line error."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool


@dataclass(frozen=True)
class AudioSource:
    path: Path
    input_index: int
    label: str
    from_existing_video_input: bool


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return self.start + self.duration / 2.0


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    camera: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-camera montage with ffmpeg. The editor switches from "
            "the main camera to the second camera on long pauses or after 15s, "
            "and switches back on shorter pauses or after 10s."
        )
    )
    parser.add_argument("main_video", nargs="?", help="Main camera video.")
    parser.add_argument("second_video", nargs="?", help="Second camera video.")
    parser.add_argument("--main", dest="main_option", help="Main camera video path.")
    parser.add_argument("--second", dest="second_option", help="Second camera video path.")
    parser.add_argument("-o", "--output", default="result.mp4", help="Output video path.")

    parser.add_argument(
        "--audio",
        help=(
            "Optional audio source. Pass a file path, 1/main for the main "
            "camera audio, or 2/second for the second camera audio."
        ),
    )
    parser.add_argument(
        "--audio-from",
        choices=("1", "2", "main", "second"),
        help="Explicitly take audio from camera 1/main or 2/second.",
    )

    parser.add_argument("--sub", action="store_true", help="Generate or burn subtitles.")
    parser.add_argument("--sub-text", help="Ready transcript text to turn into subtitles.")
    parser.add_argument("--sub-text-file", type=Path, help="Text or SRT file with ready subtitles.")
    parser.add_argument("--sub-file", type=Path, help="Existing SRT or ASS subtitle file.")
    parser.add_argument(
        "--subtitle-lowercase",
        "--sub-lowercase",
        action="store_true",
        help="Render subtitle text in lowercase while preserving subtitle markup.",
    )
    parser.add_argument(
        "--strict-subtitles",
        action="store_true",
        help="Fail if ffmpeg cannot burn subtitles into the video.",
    )
    parser.add_argument(
        "--keyword-highlights",
        choices=("auto", "ollama", "heuristic", "off"),
        default="auto",
        help="Highlight subtitle keywords with bold/italic styling.",
    )
    parser.add_argument(
        "--subtitle-timing",
        choices=("auto", "whisper", "speech", "even"),
        default="auto",
        help=(
            "Timing source for ready plain text. auto uses Whisper timestamps "
            "when audio and whisper are available, otherwise detected speech pauses."
        ),
    )
    parser.add_argument(
        "--subtitle-words",
        type=int,
        default=6,
        help="Target maximum words per subtitle cue when plain text is supplied.",
    )
    parser.add_argument(
        "--subtitle-min-duration",
        type=float,
        default=2.8,
        help="Minimum display duration for generated subtitle cues.",
    )
    parser.add_argument(
        "--subtitle-max-duration",
        type=float,
        default=8.5,
        help="Maximum display duration for generated subtitle cues.",
    )
    parser.add_argument(
        "--subtitle-hold-extension",
        type=float,
        default=1.6,
        help="How much a subtitle may stay visible after its detected speech segment.",
    )
    parser.add_argument(
        "--subtitle-gap",
        type=float,
        default=0.08,
        help="Minimum gap between generated subtitle cue updates.",
    )
    parser.add_argument(
        "--transcriber",
        choices=("auto", "whisper", "none"),
        default="auto",
        help="Speech recognition backend when --sub is used without ready text.",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper CLI model name.")
    parser.add_argument(
        "--whisper-language",
        default=None,
        help="Language passed to Whisper, for example ru or en. Omit for auto.",
    )
    parser.add_argument(
        "--ollama-model",
        help=(
            "Optional local Ollama text model used for keyword highlighting. "
            "Ollama is not used for speech recognition."
        ),
    )
    parser.add_argument(
        "--cleanup-subtext-with-ollama",
        action="store_true",
        help="Let Ollama clean ready plain text before timing. Off by default to preserve alignment.",
    )
    parser.add_argument(
        "--ollama-pull",
        action="store_true",
        help="Run 'ollama pull MODEL' before using --ollama-model.",
    )

    parser.add_argument("--main-pause-threshold", type=float, default=2.5)
    parser.add_argument("--main-max-span", type=float, default=15.0)
    parser.add_argument("--second-pause-threshold", type=float, default=1.5)
    parser.add_argument("--second-max-span", type=float, default=10.0)
    parser.add_argument(
        "--min-transitions",
        type=int,
        default=4,
        help="Minimum camera transitions in the final timeline.",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=6,
        help="Maximum camera transitions in the final timeline.",
    )
    parser.add_argument(
        "--transition-duration",
        type=float,
        default=0.45,
        help="Smooth camera transition duration in seconds.",
    )
    parser.add_argument(
        "--transition-styles",
        default="smoothleft,smoothright,circleopen,fade",
        help="Comma-separated ffmpeg xfade transitions.",
    )
    parser.add_argument(
        "--transition-mode",
        choices=("cycle", "random"),
        default="cycle",
        help="How to choose transition styles from --transition-styles.",
    )
    parser.add_argument(
        "--transition-seed",
        type=int,
        help="Optional seed for repeatable random transition selection.",
    )
    parser.add_argument("--silence-noise", default="-35dB", help="ffmpeg silencedetect noise level.")
    parser.add_argument("--min-silence", type=float, default=0.35, help="Minimum silence for analysis.")

    parser.add_argument("--width", type=int, help="Output width. Defaults to 1080.")
    parser.add_argument("--height", type=int, help="Output height. Defaults to 1920.")
    parser.add_argument("--fps", type=float, help="Output FPS. Defaults to main video FPS or 30.")
    parser.add_argument(
        "--main-rotate",
        type=int,
        default=0,
        help="Rotate the main camera clockwise by 0, 90, 180, or 270 degrees before scaling.",
    )
    parser.add_argument(
        "--second-rotate",
        type=int,
        default=0,
        help="Rotate the second camera clockwise by 0, 90, 180, or 270 degrees before scaling.",
    )
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF.")
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
    parser.add_argument("--audio-bitrate", default="192k", help="AAC audio bitrate.")
    parser.add_argument(
        "--normalize-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize output audio loudness with ffmpeg loudnorm.",
    )
    parser.add_argument(
        "--audio-normalize-mode",
        choices=("loudnorm", "speech"),
        default="loudnorm",
        help=(
            "Audio normalization chain. speech is kept as a compatibility "
            "alias for loudnorm; use --audio-gain-db for simple level changes."
        ),
    )
    parser.add_argument(
        "--audio-gain-db",
        type=float,
        default=0.0,
        help="Simple output audio gain in dB, applied after optional loudnorm.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("filter", "copy"),
        default="filter",
        help=(
            "Audio handling mode. copy maps the selected audio stream without "
            "volume, loudness, bitrate, or codec changes."
        ),
    )
    parser.add_argument(
        "--copy-audio",
        dest="audio_mode",
        action="store_const",
        const="copy",
        help="Shortcut for --audio-mode copy.",
    )
    parser.add_argument(
        "--audio-target-lufs",
        type=float,
        default=-16.0,
        help="Target integrated loudness for --normalize-audio.",
    )
    parser.add_argument(
        "--audio-true-peak",
        type=float,
        default=-1.5,
        help="Target true peak in dBTP for --normalize-audio.",
    )
    parser.add_argument(
        "--audio-lra",
        type=float,
        default=11.0,
        help="Target loudness range for --normalize-audio.",
    )
    parser.add_argument(
        "--second-zoom",
        type=float,
        default=1.0,
        help="Small zoom applied to the second camera before crop.",
    )
    parser.add_argument(
        "--second-effect",
        choices=("off", "vhs", "glitch", "vhs-glitch"),
        default="vhs",
        help="Visual effect applied only to the second camera before subtitles.",
    )
    parser.add_argument(
        "--second-denoise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply a light hqdn3d denoise to the second camera. Off by default.",
    )

    parser.add_argument(
        "--max-part-duration",
        type=float,
        default=60.0,
        help="If the result is longer than this, split it into two near-equal parts.",
    )
    parser.add_argument("--no-split", action="store_true", help="Do not split long output.")
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional processing duration in seconds. Defaults to shared media duration.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary audio/subtitle files.")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without running them.")
    parser.add_argument(
        "--ffmpeg-bin",
        default=None,
        help="ffmpeg binary to use. Defaults to FFMPEG_BIN or ffmpeg.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=None,
        help="ffprobe binary to use. Defaults to FFPROBE_BIN, sibling ffprobe, or ffprobe.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run internal tests that do not need ffmpeg.")
    return parser


def run_cmd(
    cmd: Sequence[str],
    *,
    capture: bool = False,
    input_text: str | None = None,
    dry_run: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print("+", shlex.join(cmd), flush=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    print("+", shlex.join(cmd), flush=True)
    try:
        return subprocess.run(
            list(cmd),
            input=input_text,
            text=True,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required tool not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        detail = (stderr or stdout).strip()
        if detail:
            raise CliError(f"Command failed: {shlex.join(cmd)}\n{detail}") from exc
        raise CliError(f"Command failed: {shlex.join(cmd)}") from exc


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise CliError(
            f"'{name}' is not installed or is not in PATH. Install ffmpeg/ffprobe first."
        )


def configure_media_tools(args: argparse.Namespace) -> None:
    global TOOL_FFMPEG, TOOL_FFPROBE

    explicit_ffmpeg = args.ffmpeg_bin or os_environ("FFMPEG_BIN")
    TOOL_FFMPEG = explicit_ffmpeg or "ffmpeg"
    if not explicit_ffmpeg and not available_ffmpeg_filters(TOOL_FFMPEG).intersection(
        {"subtitles", "drawtext"}
    ):
        for candidate in (
            Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"),
            Path("/usr/local/opt/ffmpeg-full/bin/ffmpeg"),
        ):
            if candidate.exists() and available_ffmpeg_filters(str(candidate)).intersection(
                {"subtitles", "drawtext"}
            ):
                TOOL_FFMPEG = str(candidate)
                break

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


def os_environ(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def available_ffmpeg_filters(ffmpeg_bin: str | None = None) -> set[str]:
    """Return filter names exposed by the installed ffmpeg build."""

    try:
        result = subprocess.run(
            [ffmpeg_bin or TOOL_FFMPEG, "-hide_banner", "-filters"],
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
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    duration: float | None = None
    for raw in (
        data.get("format", {}).get("duration"),
        video.get("duration") if video else None,
        audio.get("duration") if audio else None,
    ):
        if raw not in (None, "N/A"):
            duration = float(raw)
            break

    return MediaInfo(
        path=path,
        duration=duration,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=parse_rate(video.get("avg_frame_rate") if video else None),
        has_audio=audio is not None,
    )


def even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value - 1


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    main_raw = args.main_option or args.main_video
    second_raw = args.second_option or args.second_video
    if not main_raw or not second_raw:
        raise CliError("Pass both camera videos: main_video second_video or --main/--second.")
    main = Path(main_raw).expanduser().resolve()
    second = Path(second_raw).expanduser().resolve()
    if not main.exists():
        raise CliError(f"Main video does not exist: {main}")
    if not second.exists():
        raise CliError(f"Second camera video does not exist: {second}")
    return main, second


def resolve_audio_source(
    args: argparse.Namespace,
    main: Path,
    second: Path,
    main_info: MediaInfo,
    second_info: MediaInfo,
) -> AudioSource | None:
    token = args.audio_from or args.audio
    if token:
        normalized = str(token).strip().lower()
        if normalized in {"1", "main", "primary"}:
            if not main_info.has_audio:
                raise CliError("Camera 1/main has no audio stream.")
            return AudioSource(main, 0, "main camera", True)
        if normalized in {"2", "second", "secondary"}:
            if not second_info.has_audio:
                raise CliError("Camera 2/second has no audio stream.")
            return AudioSource(second, 1, "second camera", True)

        audio_path = Path(str(token)).expanduser().resolve()
        if not audio_path.exists():
            raise CliError(f"Audio file does not exist: {audio_path}")
        return AudioSource(audio_path, 2, "external audio file", False)

    if main_info.has_audio:
        return AudioSource(main, 0, "main camera", True)
    if second_info.has_audio:
        return AudioSource(second, 1, "second camera", True)
    return None


def choose_timeline_duration(
    args: argparse.Namespace,
    main_info: MediaInfo,
    second_info: MediaInfo,
    audio_info: MediaInfo | None,
) -> float:
    if args.duration:
        if args.duration <= 0:
            raise CliError("--duration must be greater than zero.")
        return args.duration

    durations = [value for value in (main_info.duration, second_info.duration) if value]
    if audio_info and audio_info.duration:
        durations.append(audio_info.duration)
    if not durations:
        raise CliError("Could not determine media duration. Pass --duration explicitly.")
    return min(durations)


def extract_analysis_audio(
    audio_source: AudioSource,
    temp_dir: Path,
    duration: float,
    *,
    dry_run: bool = False,
) -> Path:
    wav_path = temp_dir / "analysis_audio.wav"
    cmd = [
        TOOL_FFMPEG,
        "-y",
        "-hide_banner",
        "-i",
        str(audio_source.path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-t",
        fmt_seconds(duration),
        str(wav_path),
    ]
    run_cmd(cmd, dry_run=dry_run)
    return wav_path


def detect_silences(
    audio_path: Path,
    duration: float,
    *,
    noise: str,
    min_silence: float,
    dry_run: bool = False,
) -> list[Silence]:
    if dry_run:
        return []

    cmd = [
        TOOL_FFMPEG,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise}:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    result = run_cmd(cmd, capture=True)
    text = (result.stderr or "") + "\n" + (result.stdout or "")
    starts: list[float] = []
    silences: list[Silence] = []
    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

    for line in text.splitlines():
        start_match = start_re.search(line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue

        end_match = end_re.search(line)
        if end_match and starts:
            start = starts.pop(0)
            end = float(end_match.group(1))
            if end > start:
                silences.append(Silence(start, min(end, duration)))

    for start in starts:
        if duration > start:
            silences.append(Silence(start, duration))

    silences.sort(key=lambda silence: silence.start)
    return silences


def find_pause_cut(
    current: float,
    deadline: float,
    threshold: float,
    silences: Iterable[Silence],
    consumed_silences: set[tuple[float, float]],
) -> tuple[float | None, Silence | None]:
    min_step = 0.15
    for silence in silences:
        silence_key = (round(silence.start, 3), round(silence.end, 3))
        if silence_key in consumed_silences:
            continue
        if silence.end <= current + min_step:
            continue
        if silence.duration < threshold:
            continue
        trigger = max(silence.start + threshold, current + min_step)
        if trigger <= silence.end and trigger <= deadline:
            return trigger, silence
    return None, None


def silence_score_for_cut(cut: float, silences: Sequence[Silence]) -> float:
    score = 0.0
    for silence in silences:
        if silence.start <= cut <= silence.end:
            score = max(score, silence.duration)
        else:
            distance = min(abs(cut - silence.start), abs(cut - silence.end))
            if distance <= 0.75:
                score = max(score, max(0.0, silence.duration - distance))
    return score


def unique_sorted_cuts(cuts: Iterable[float], duration: float) -> list[float]:
    normalized: list[float] = []
    for cut in sorted(cuts):
        if cut <= 0.25 or cut >= duration - 0.25:
            continue
        if normalized and abs(cut - normalized[-1]) < 0.25:
            continue
        normalized.append(cut)
    return normalized


def select_transition_cuts(
    candidate_cuts: Sequence[float],
    duration: float,
    silences: Sequence[Silence],
    *,
    min_transitions: int,
    max_transitions: int | None,
) -> list[float]:
    min_transitions = max(0, min_transitions)
    candidates = unique_sorted_cuts(candidate_cuts, duration)
    if max_transitions is None:
        if len(candidates) >= min_transitions:
            return candidates
        max_transitions = min_transitions
    max_transitions = max(min_transitions, max_transitions)
    if max_transitions == 0:
        return []

    target_count = min(max(len(candidates), min_transitions), max_transitions)
    if target_count <= 0:
        return []

    even_cuts = [duration * (index + 1) / (target_count + 1) for index in range(target_count)]
    if len(candidates) < target_count:
        candidates = unique_sorted_cuts([*candidates, *even_cuts], duration)

    selected: list[float] = []
    available = candidates[:]
    min_spacing = max(1.0, duration / max(target_count * 3.0, 1.0))
    target_window = max(2.0, duration / (target_count + 1) * 0.65)

    for target in even_cuts:
        pool = [cut for cut in available if abs(cut - target) <= target_window]
        if not pool:
            pool = available
        if not pool:
            selected.append(target)
            continue

        def rank(cut: float) -> tuple[float, float]:
            spacing_penalty = 0.0
            if selected:
                nearest = min(abs(cut - existing) for existing in selected)
                spacing_penalty = max(0.0, min_spacing - nearest) * 3.0
            pause_bonus = min(3.0, silence_score_for_cut(cut, silences)) * 0.35
            return abs(cut - target) + spacing_penalty - pause_bonus, cut

        cut = min(pool, key=rank)
        selected.append(cut)
        available.remove(cut)

    return unique_sorted_cuts(selected, duration)


def segments_from_cuts(cuts: Sequence[float], duration: float) -> list[Segment]:
    segments: list[Segment] = []
    start = 0.0
    camera = CAM_MAIN
    for cut in cuts:
        if cut > start + 0.05:
            segments.append(Segment(start, cut, camera))
            camera = CAM_SECOND if camera == CAM_MAIN else CAM_MAIN
            start = cut
    if duration > start + 0.05:
        segments.append(Segment(start, duration, camera))
    return segments or [Segment(0.0, duration, CAM_MAIN)]


def build_segments(
    duration: float,
    silences: Sequence[Silence],
    *,
    main_pause_threshold: float,
    main_max_span: float,
    second_pause_threshold: float,
    second_max_span: float,
    min_transitions: int = 0,
    max_transitions: int | None = None,
) -> list[Segment]:
    if duration <= 0:
        raise CliError("Timeline duration must be greater than zero.")

    segments: list[Segment] = []
    current = 0.0
    camera = CAM_MAIN
    guard = 0
    consumed_silences: set[tuple[float, float]] = set()

    while current < duration - 0.05:
        guard += 1
        if guard > 10000:
            raise CliError("Timeline builder exceeded the safety iteration limit.")

        if camera == CAM_MAIN:
            max_span = main_max_span
            threshold = main_pause_threshold
            next_camera = CAM_SECOND
        else:
            max_span = second_max_span
            threshold = second_pause_threshold
            next_camera = CAM_MAIN

        deadline = min(duration, current + max_span)
        cut, used_silence = find_pause_cut(
            current,
            deadline,
            threshold,
            silences,
            consumed_silences,
        )
        if cut is None:
            cut = deadline
        elif used_silence is not None:
            consumed_silences.add((round(used_silence.start, 3), round(used_silence.end, 3)))
        cut = min(max(cut, current + 0.15), duration)

        if cut <= current + 0.05:
            break

        segments.append(Segment(current, cut, camera))
        current = cut
        camera = next_camera

    if not segments:
        return [Segment(0.0, duration, CAM_MAIN)]

    candidate_cuts = [segment.end for segment in segments if segment.end < duration - 0.05]
    selected_cuts = select_transition_cuts(
        candidate_cuts,
        duration,
        silences,
        min_transitions=min_transitions,
        max_transitions=max_transitions,
    )
    return segments_from_cuts(selected_cuts, duration)


def speech_intervals(duration: float, silences: Sequence[Silence]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for silence in silences:
        start = max(0.0, min(duration, silence.start))
        end = max(0.0, min(duration, silence.end))
        if start > cursor + 0.25:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 0.25:
        intervals.append((cursor, duration))
    return intervals or [(0.0, duration)]


def split_text_chunks(text: str, max_words: int = 6) -> list[str]:
    max_words = max(1, max_words)
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    for sentence in sentence_parts:
        words = sentence.split()
        while words:
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
    return chunks


def cues_from_plain_text(
    text: str,
    duration: float,
    silences: Sequence[Silence],
    *,
    max_words: int = 6,
    min_duration: float = 2.8,
    max_duration: float = 8.5,
    gap: float = 0.08,
    use_speech_intervals: bool = True,
) -> list[SubtitleCue]:
    chunks = split_text_chunks(text, max_words=max_words)
    if not chunks:
        return []

    intervals = speech_intervals(duration, silences) if use_speech_intervals else [(0.0, duration)]
    total_available = sum(end - start for start, end in intervals)
    if total_available <= 0:
        total_available = duration
        intervals = [(0.0, duration)]

    weights = [max(1, len(chunk.split())) for chunk in chunks]
    total_weight = sum(weights)
    max_duration = max(min_duration, max_duration)
    cue_durations = [
        max(min_duration, min(max_duration, total_available * weight / total_weight))
        for weight in weights
    ]

    cues: list[SubtitleCue] = []
    interval_index = 0
    cursor = intervals[0][0]
    for chunk, cue_duration in zip(chunks, cue_durations):
        while interval_index < len(intervals) and cursor >= intervals[interval_index][1] - 0.2:
            interval_index += 1
            if interval_index < len(intervals):
                cursor = intervals[interval_index][0]

        if interval_index >= len(intervals):
            start = min(duration - 0.25, cues[-1].end + 0.05 if cues else 0.0)
            end = min(duration, start + max(0.5, cue_duration))
        else:
            interval_start, interval_end = intervals[interval_index]
            start = max(cursor, interval_start)
            end = min(interval_end, start + cue_duration)
            if end - start < 0.45 and interval_index + 1 < len(intervals):
                interval_index += 1
                start = intervals[interval_index][0]
                end = min(intervals[interval_index][1], start + cue_duration)
            cursor = end + gap

        if end > start:
            cues.append(SubtitleCue(start, end, chunk))

    return cues


def parse_timecode(raw: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", raw.strip())
    if not match:
        raise ValueError(f"Invalid timecode: {raw}")
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis[:3].ljust(3, "0")) / 1000.0
    )


def parse_srt(path: Path) -> list[SubtitleCue]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[SubtitleCue] = []
    timing_re = re.compile(
        r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)"
    )

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next(
            (index for index, line in enumerate(lines) if timing_re.search(line)),
            None,
        )
        if timing_index is None:
            continue
        match = timing_re.search(lines[timing_index])
        if not match:
            continue
        cue_text = "\n".join(lines[timing_index + 1 :]).strip()
        if cue_text:
            cues.append(
                SubtitleCue(
                    parse_timecode(match.group(1)),
                    parse_timecode(match.group(2)),
                    cue_text,
                )
            )
    return cues


SUBTITLE_MARKUP_RE = re.compile(r"(</?b>|</?i>|\{\\[^}]*\}|\\[Nnh])")


def lowercase_subtitle_text(text: str) -> str:
    parts = SUBTITLE_MARKUP_RE.split(text)
    return "".join(
        part if SUBTITLE_MARKUP_RE.fullmatch(part) else part.lower()
        for part in parts
    )


def lowercase_subtitle_cues(cues: Sequence[SubtitleCue]) -> list[SubtitleCue]:
    return [
        SubtitleCue(cue.start, cue.end, lowercase_subtitle_text(cue.text))
        for cue in cues
    ]


def lowercase_ass_dialogue_line(line: str, text_index: int | None = None) -> str:
    fallback_text_index = 9
    index = text_index if text_index is not None and text_index >= 0 else fallback_text_index
    parts = line.split(",", index)
    if len(parts) <= index:
        if index == fallback_text_index:
            return line
        parts = line.split(",", fallback_text_index)
        index = fallback_text_index
        if len(parts) <= index:
            return line
    parts[index] = lowercase_subtitle_text(parts[index])
    return ",".join(parts)


def write_lowercase_ass_copy(source: Path, target: Path) -> Path:
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    output: list[str] = []
    in_events = False
    dialogue_text_index: int | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.lower() == "[events]"
            output.append(line)
            continue

        if in_events and stripped.lower().startswith("format:"):
            raw_fields = stripped.split(":", 1)[1]
            fields = [field.strip().lower() for field in raw_fields.split(",")]
            dialogue_text_index = fields.index("text") if "text" in fields else None
            output.append(line)
            continue

        if in_events and line.lstrip().lower().startswith("dialogue:"):
            output.append(lowercase_ass_dialogue_line(line, dialogue_text_index))
            continue

        output.append(line)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return target


def normalize_alignment_token(text: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", "", text.lower().replace("ё", "е"))


def subtitle_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", text) if word.strip("-")]


def adjust_subtitle_cues(
    cues: Sequence[SubtitleCue],
    *,
    duration: float,
    min_duration: float,
    max_duration: float,
    hold_extension: float,
    gap: float,
) -> list[SubtitleCue]:
    normalized: list[SubtitleCue] = []
    for cue in sorted(cues, key=lambda item: (item.start, item.end)):
        text = cue.text.strip()
        if not text:
            continue
        start = max(0.0, min(duration, cue.start))
        end = max(0.0, min(duration, cue.end))
        if end <= start:
            continue
        normalized.append(SubtitleCue(start, end, text))

    if not normalized:
        return []

    min_duration = max(0.05, min_duration)
    max_duration = max(min_duration, max_duration)
    hold_extension = max(0.0, hold_extension)
    gap = max(0.0, gap)

    adjusted: list[SubtitleCue] = []
    for index, cue in enumerate(normalized):
        next_start = normalized[index + 1].start if index + 1 < len(normalized) else duration
        latest_end = max(cue.start, min(duration, next_start - gap))
        target_end = max(cue.end + hold_extension, cue.start + min_duration)
        target_end = min(target_end, cue.start + max_duration, latest_end)
        if target_end <= cue.start:
            target_end = min(cue.end, latest_end)
        if target_end > cue.start:
            adjusted.append(SubtitleCue(cue.start, target_end, cue.text))
    return adjusted


def timed_words_to_cues(
    words: Sequence[TimedWord],
    *,
    max_words: int,
) -> list[SubtitleCue]:
    max_words = max(1, max_words)
    clean_words = [word for word in words if word.text.strip() and word.end > word.start]
    cues: list[SubtitleCue] = []
    cursor = 0
    while cursor < len(clean_words):
        end_index = min(len(clean_words), cursor + max_words)
        for candidate in range(cursor + 1, end_index + 1):
            if re.search(r"[.!?。！？…]$", clean_words[candidate - 1].text.strip()):
                end_index = candidate
                break
        chunk_words = clean_words[cursor:end_index]
        text = " ".join(word.text.strip() for word in chunk_words)
        cues.append(SubtitleCue(chunk_words[0].start, chunk_words[-1].end, text))
        cursor = end_index
    return cues


def text_chunks_with_token_ranges(text: str, max_words: int) -> list[tuple[str, int, int]]:
    chunks = split_text_chunks(text, max_words=max_words)
    ranges: list[tuple[str, int, int]] = []
    cursor = 0
    for chunk in chunks:
        count = max(1, len(subtitle_words(chunk)))
        ranges.append((chunk, cursor, cursor + count))
        cursor += count
    return ranges


def target_to_source_word_map(
    target_tokens: Sequence[str],
    source_tokens: Sequence[str],
) -> dict[int, int]:
    matcher = difflib.SequenceMatcher(None, list(target_tokens), list(source_tokens), autojunk=False)
    mapping: dict[int, int] = {}
    for tag, target_start, target_end, source_start, source_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(target_end - target_start):
                mapping[target_start + offset] = source_start + offset
    return mapping


def ready_text_with_timed_words_to_cues(
    text: str,
    words: Sequence[TimedWord],
    *,
    max_words: int,
) -> list[SubtitleCue]:
    chunks = text_chunks_with_token_ranges(text, max_words=max_words)
    if not chunks:
        return []
    if not words:
        return []

    source_tokens = [normalize_alignment_token(word.text) for word in words]
    target_tokens = [normalize_alignment_token(word) for word in subtitle_words(text)]
    source_tokens = [token for token in source_tokens if token]
    target_tokens = [token for token in target_tokens if token]
    token_map = target_to_source_word_map(target_tokens, source_tokens)

    cues: list[SubtitleCue] = []
    last_source_end = -1
    total_target = max(1, len(target_tokens))
    total_source = len(words)

    for chunk, target_start, target_end in chunks:
        mapped = [
            token_map[index]
            for index in range(target_start, target_end)
            if index in token_map and 0 <= token_map[index] < total_source
        ]
        if mapped:
            source_start = min(mapped)
            source_end = max(mapped)
            if source_start <= last_source_end:
                source_start = min(total_source - 1, last_source_end + 1)
                source_end = max(source_start, source_end)
        else:
            source_start = round(target_start / total_target * max(0, total_source - 1))
            source_end = max(
                source_start,
                round(target_end / total_target * max(0, total_source - 1)),
            )
            if source_start <= last_source_end:
                source_start = min(total_source - 1, last_source_end + 1)
                source_end = max(source_start, source_end)

        source_end = min(total_source - 1, source_end)
        if source_start >= total_source:
            break
        cues.append(SubtitleCue(words[source_start].start, words[source_end].end, chunk))
        last_source_end = source_end

    return cues


def parse_whisper_json(path: Path) -> tuple[list[TimedWord], list[SubtitleCue]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[TimedWord] = []
    segment_cues: list[SubtitleCue] = []

    for segment in data.get("segments", []):
        text = str(segment.get("text") or "").strip()
        try:
            segment_start = float(segment.get("start"))
            segment_end = float(segment.get("end"))
        except (TypeError, ValueError):
            segment_start = segment_end = 0.0
        if text and segment_end > segment_start:
            segment_cues.append(SubtitleCue(segment_start, segment_end, text))

        for item in segment.get("words") or []:
            word_text = str(item.get("word") or "").strip()
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                continue
            if word_text and end > start:
                words.append(TimedWord(word_text, start, end))

    return words, segment_cues


def run_whisper(
    audio_path: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    *,
    output_format: str = "srt",
    word_timestamps: bool = False,
    dry_run: bool = False,
) -> Path:
    if shutil.which("whisper") is None:
        raise CliError(
            "Whisper CLI is not installed. Install openai-whisper or pass "
            "--sub-text, --sub-text-file, or --sub-file."
        )

    cmd = [
        "whisper",
        str(audio_path),
        "--model",
        args.whisper_model,
        "--output_format",
        output_format,
        "--output_dir",
        str(temp_dir),
    ]
    if args.whisper_language:
        cmd.extend(["--language", args.whisper_language])
    if word_timestamps:
        cmd.extend(["--word_timestamps", "True"])
    if output_format in {"srt", "vtt"}:
        cmd.extend(["--max_words_per_line", str(max(1, args.subtitle_words))])
    run_cmd(cmd, dry_run=dry_run)

    suffix = output_format if output_format != "all" else "srt"
    if dry_run:
        return temp_dir / f"{audio_path.stem}.{suffix}"

    candidates = sorted(temp_dir.glob(f"*.{suffix}"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise CliError(f"Whisper completed but did not create a {suffix.upper()} file.")
    return candidates[-1]


def maybe_refine_text_with_ollama(
    text: str,
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> str:
    if not args.cleanup_subtext_with_ollama or not args.ollama_model:
        return text
    if shutil.which("ollama") is None:
        print(
            "Warning: 'ollama' is not installed; using subtitle text without model cleanup.",
            file=sys.stderr,
        )
        return text

    try:
        if args.ollama_pull:
            run_cmd(["ollama", "pull", args.ollama_model], dry_run=dry_run)

        prompt = (
            "Clean this transcript for subtitles. Keep the original language, do not "
            "add facts, do not summarize, and return only the cleaned transcript.\n\n"
            f"{text.strip()}"
        )
        result = run_cmd(
            ["ollama", "run", args.ollama_model],
            capture=True,
            input_text=prompt,
            dry_run=dry_run,
            timeout=300,
        )
    except CliError as exc:
        print(
            f"Warning: Ollama cleanup failed; using subtitle text as-is: {exc}",
            file=sys.stderr,
        )
        return text

    cleaned = (result.stdout or "").strip()
    return cleaned or text


STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was",
    "were", "have", "has", "not", "but", "from", "they", "their", "there",
    "what", "when", "where", "why", "how", "can", "could", "would", "should",
    "into", "about", "just", "then", "than", "them", "will", "all", "our",
}


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9-]{2,}")


def strip_text_markup(text: str) -> str:
    return re.sub(r"</?[bi]>", "", text)


def cue_keywords_heuristic(text: str, cue_index: int) -> dict[str, str]:
    counts: dict[str, tuple[str, float]] = {}
    for match in WORD_RE.finditer(strip_text_markup(text)):
        word = match.group(0)
        key = word.lower().replace("ё", "е")
        if key in STOPWORDS or len(key) < 4:
            continue
        score = len(key)
        if word[:1].isupper():
            score += 1.5
        if key not in counts or score > counts[key][1]:
            counts[key] = (word, score)

    ranked = sorted(counts.values(), key=lambda item: item[1], reverse=True)[:3]
    styles: dict[str, str] = {}
    for index, (word, _) in enumerate(ranked):
        if index == 0 and cue_index % 3 != 1:
            styles[word] = "bold"
        else:
            styles[word] = "italic"
    return styles


def parse_ollama_keyword_json(raw: str) -> dict[int, dict[str, str]]:
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise ValueError("No JSON array found.")
    data = json.loads(match.group(0))
    styles: dict[int, dict[str, str]] = {}
    for item in data:
        index = int(item.get("index"))
        styles[index] = {}
        for word in item.get("bold", [])[:2]:
            if isinstance(word, str) and word.strip():
                styles[index][word.strip()] = "bold"
        for word in item.get("italic", [])[:3]:
            if isinstance(word, str) and word.strip():
                styles[index].setdefault(word.strip(), "italic")
    return styles


def cue_keywords_ollama(
    cues: Sequence[SubtitleCue],
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> dict[int, dict[str, str]]:
    if not args.ollama_model:
        raise CliError("--keyword-highlights=ollama requires --ollama-model.")
    if shutil.which("ollama") is None:
        raise CliError("'ollama' is not installed but Ollama keyword highlights were requested.")

    cue_lines = "\n".join(
        f"{index}: {strip_text_markup(cue.text)}" for index, cue in enumerate(cues)
    )
    prompt = (
        "For each subtitle cue, choose only the most meaningful words in the "
        "same language. Return strict JSON only, no markdown, as an array of "
        "objects: {\"index\": number, \"bold\": [words], \"italic\": [words]}. "
        "Use at most one bold word and at most two italic words per cue. "
        "Do not rewrite cue text.\n\n"
        f"{cue_lines}"
    )
    result = run_cmd(
        ["ollama", "run", args.ollama_model],
        capture=True,
        input_text=prompt,
        dry_run=dry_run,
        timeout=300,
    )
    return parse_ollama_keyword_json(result.stdout or "")


def apply_style_to_first_match(text: str, word: str, style: str) -> str:
    tag = "b" if style == "bold" else "i"
    pattern = re.compile(rf"(?<![\w<])({re.escape(word)})(?![\w>])", re.IGNORECASE)
    return pattern.sub(lambda match: f"<{tag}>{match.group(1)}</{tag}>", text, count=1)


def apply_keyword_highlights_to_cues(
    cues: Sequence[SubtitleCue],
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> list[SubtitleCue]:
    if args.keyword_highlights == "off" or not cues:
        return list(cues)

    styles_by_index: dict[int, dict[str, str]] = {}
    use_ollama = args.keyword_highlights == "ollama" or (
        args.keyword_highlights == "auto" and bool(args.ollama_model)
    )
    if use_ollama:
        try:
            styles_by_index = cue_keywords_ollama(cues, args, dry_run=dry_run)
        except (CliError, ValueError, json.JSONDecodeError) as exc:
            if args.keyword_highlights == "ollama":
                raise CliError(f"Ollama keyword highlighting failed: {exc}") from exc
            print(
                f"Warning: Ollama keyword highlighting failed; using heuristic fallback: {exc}",
                file=sys.stderr,
            )

    highlighted: list[SubtitleCue] = []
    for index, cue in enumerate(cues):
        styles = styles_by_index.get(index) or cue_keywords_heuristic(cue.text, index)
        text = cue.text
        for word, style in sorted(styles.items(), key=lambda item: len(item[0]), reverse=True):
            text = apply_style_to_first_match(text, word, style)
        highlighted.append(SubtitleCue(cue.start, cue.end, text))
    return highlighted


def cues_from_whisper_json(
    audio_path: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    *,
    ready_text: str | None = None,
    dry_run: bool = False,
) -> list[SubtitleCue]:
    json_path = run_whisper(
        audio_path,
        temp_dir,
        args,
        output_format="json",
        word_timestamps=True,
        dry_run=dry_run,
    )
    if dry_run:
        return []

    words, segment_cues = parse_whisper_json(json_path)
    if ready_text is not None:
        if words:
            return ready_text_with_timed_words_to_cues(
                ready_text,
                words,
                max_words=args.subtitle_words,
            )
        if segment_cues:
            return ready_text_against_reference_cues(
                ready_text,
                segment_cues,
                max_words=args.subtitle_words,
            )
        raise CliError("Whisper did not produce word timestamps for ready subtitle text.")

    if words:
        return timed_words_to_cues(words, max_words=args.subtitle_words)
    if segment_cues:
        return segment_cues
    raise CliError("Whisper did not produce subtitle timestamps.")


def ready_text_against_reference_cues(
    text: str,
    reference_cues: Sequence[SubtitleCue],
    *,
    max_words: int,
) -> list[SubtitleCue]:
    chunks = split_text_chunks(text, max_words=max_words)
    if not chunks or not reference_cues:
        return []
    cue_start = reference_cues[0].start
    cue_end = reference_cues[-1].end
    total_duration = max(0.1, cue_end - cue_start)
    weights = [max(1, len(subtitle_words(chunk))) for chunk in chunks]
    total_weight = max(1, sum(weights))
    cursor = cue_start
    cues: list[SubtitleCue] = []
    for chunk, weight in zip(chunks, weights):
        chunk_duration = total_duration * weight / total_weight
        end = min(cue_end, cursor + chunk_duration)
        if end > cursor:
            cues.append(SubtitleCue(cursor, end, chunk))
        cursor = end
    return cues


def cues_from_ready_text(
    text: str,
    args: argparse.Namespace,
    audio_path: Path | None,
    temp_dir: Path,
    duration: float,
    silences: Sequence[Silence],
    *,
    dry_run: bool = False,
) -> list[SubtitleCue]:
    mode = args.subtitle_timing
    if mode in {"auto", "whisper"} and audio_path and shutil.which("whisper"):
        try:
            return cues_from_whisper_json(
                audio_path,
                temp_dir,
                args,
                ready_text=text,
                dry_run=dry_run,
            )
        except CliError as exc:
            if mode == "whisper":
                raise
            print(
                f"Warning: Whisper timing failed; using speech-interval subtitle timing: {exc}",
                file=sys.stderr,
            )

    if mode == "whisper":
        if not audio_path:
            raise CliError("--subtitle-timing=whisper requires an audio source.")
        raise CliError("Whisper CLI is not installed.")

    return cues_from_plain_text(
        text,
        duration,
        silences,
        max_words=args.subtitle_words,
        min_duration=args.subtitle_min_duration,
        max_duration=args.subtitle_max_duration,
        gap=args.subtitle_gap,
        use_speech_intervals=(mode != "even"),
    )


def load_subtitle_cues(
    args: argparse.Namespace,
    audio_path: Path | None,
    temp_dir: Path,
    duration: float,
    silences: Sequence[Silence],
    *,
    dry_run: bool = False,
) -> tuple[list[SubtitleCue] | None, Path | None]:
    if not args.sub:
        return None, None

    if args.sub_file:
        subtitle_path = args.sub_file.expanduser().resolve()
        if not subtitle_path.exists():
            raise CliError(f"Subtitle file does not exist: {subtitle_path}")
        if subtitle_path.suffix.lower() == ".ass":
            if args.subtitle_lowercase:
                return None, write_lowercase_ass_copy(
                    subtitle_path,
                    temp_dir / f"{subtitle_path.stem}_lowercase.ass",
                )
            return None, subtitle_path
        return parse_srt(subtitle_path), None

    if args.sub_text:
        text = maybe_refine_text_with_ollama(args.sub_text, args, dry_run=dry_run)
        return cues_from_ready_text(
            text,
            args,
            audio_path,
            temp_dir,
            duration,
            silences,
            dry_run=dry_run,
        ), None

    if args.sub_text_file:
        text_path = args.sub_text_file.expanduser().resolve()
        if not text_path.exists():
            raise CliError(f"Subtitle text file does not exist: {text_path}")
        if text_path.suffix.lower() == ".ass":
            if args.subtitle_lowercase:
                return None, write_lowercase_ass_copy(
                    text_path,
                    temp_dir / f"{text_path.stem}_lowercase.ass",
                )
            return None, text_path
        if text_path.suffix.lower() == ".srt":
            return parse_srt(text_path), None
        text = text_path.read_text(encoding="utf-8-sig")
        text = maybe_refine_text_with_ollama(text, args, dry_run=dry_run)
        return cues_from_ready_text(
            text,
            args,
            audio_path,
            temp_dir,
            duration,
            silences,
            dry_run=dry_run,
        ), None

    if args.transcriber == "none":
        raise CliError("--sub was passed, but no subtitle text/file was provided.")
    if not audio_path:
        raise CliError("Automatic subtitles need an audio source.")

    if args.transcriber in {"auto", "whisper"}:
        return cues_from_whisper_json(
            audio_path,
            temp_dir,
            args,
            dry_run=dry_run,
        ), None

    raise CliError(f"Unsupported transcriber: {args.transcriber}")


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    cs = centiseconds % 100
    total_seconds = centiseconds // 100
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"


def ass_escape_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", "\n")
        .replace("\n", r"\N")
    )


def ass_escape_markup(text: str) -> str:
    chunks = re.split(r"(</?b>|</?i>)", text)
    output: list[str] = []
    for chunk in chunks:
        if chunk == "<b>":
            output.append(r"{\b1}")
        elif chunk == "</b>":
            output.append(r"{\b0}")
        elif chunk == "<i>":
            output.append(r"{\i1}")
        elif chunk == "</i>":
            output.append(r"{\i0}")
        elif chunk:
            output.append(ass_escape_text(chunk))
    return "".join(output)


def write_rotating_ass(
    cues: Sequence[SubtitleCue],
    path: Path,
    *,
    width: int,
    height: int,
    font: str = "Arial",
    font_size: int | None = None,
) -> Path:
    size = font_size or max(34, int(height * 0.038))
    outline = max(2, int(size * 0.09))
    side_margin = max(48, int(width * 0.06))
    bottom_margin = max(28, int(height * 0.09)) + SUBTITLE_VERTICAL_RAISE_PX

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&HCC000000,"
            f"&H99000000,1,0,0,0,100,100,0,0,1,{outline},1,2,"
            f"{side_margin},{side_margin},{bottom_margin},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in cues:
        if cue.end <= cue.start:
            continue
        lines.append(
            "Dialogue: 0,"
            f"{ass_time(cue.start)},{ass_time(cue.end)},Default,,0,0,0,,"
            rf"{{\fad(80,180)}}{ass_escape_markup(cue.text)}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def default_font_file() -> Path | None:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    return next((path for path in candidates if path.exists()), None)


def wrap_subtitle_line(text: str, width: int = 32) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return ""
    return "\n".join(
        textwrap.wrap(
            normalized,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def drawtext_option(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(";", r"\;")
    )


def write_drawtext_subtitle_filters(
    cues: Sequence[SubtitleCue],
    temp_dir: Path,
    *,
    width: int,
    height: int,
) -> list[str]:
    font_file = default_font_file()
    if font_file is None:
        raise CliError("Could not find a local font file for drawtext subtitle fallback.")

    font_size = max(34, int(height * 0.038))
    border = max(3, int(font_size * 0.10))
    box_border = max(14, int(font_size * 0.28))
    bottom_margin = max(28, int(height * 0.09)) + SUBTITLE_VERTICAL_RAISE_PX
    wrap_chars = max(14, int(width / (font_size * 0.62)))

    filters: list[str] = []
    text_dir = temp_dir / "drawtext_subtitles"
    text_dir.mkdir(parents=True, exist_ok=True)

    def append_filter(text: str, start: float, end: float, index: int) -> None:
        if not text or end <= start:
            return
        text_file = text_dir / f"subtitle_{index:04d}.txt"
        text_file.write_text(
            wrap_subtitle_line(strip_text_markup(text), width=wrap_chars),
            encoding="utf-8",
        )
        filters.append(
            "drawtext="
            f"fontfile='{drawtext_option(str(font_file))}':"
            f"textfile='{drawtext_option(str(text_file))}':"
            f"x=(w-text_w)/2:"
            f"y=h-text_h-{bottom_margin}:"
            f"fontsize={font_size}:"
            "fontcolor=white:"
            f"borderw={border}:"
            "bordercolor=black@0.90:"
            "box=1:"
            "boxcolor=black@0.38:"
            f"boxborderw={box_border}:"
            "line_spacing=8:"
            f"enable='between(t,{fmt_seconds(start)},{fmt_seconds(end)})'"
        )

    for index, cue in enumerate(cues):
        append_filter(cue.text, cue.start, cue.end, index)

    return filters


def copy_subtitle_sidecar(subtitle_file: Path, output: Path) -> Path:
    suffix = subtitle_file.suffix if subtitle_file.suffix else ".ass"
    sidecar = output.with_suffix(suffix)
    if subtitle_file.resolve() != sidecar.resolve():
        shutil.copyfile(subtitle_file, sidecar)
    return sidecar


def prepare_subtitle_filters(
    args: argparse.Namespace,
    *,
    output: Path,
    subtitle_file: Path | None,
    cues: Sequence[SubtitleCue] | None,
    temp_dir: Path,
    width: int,
    height: int,
) -> tuple[Path | None, list[str]]:
    if subtitle_file is None:
        return None, []

    filters = available_ffmpeg_filters()
    if "subtitles" in filters:
        return subtitle_file, []

    if cues and "drawtext" in filters:
        print(
            "ffmpeg has no subtitles/libass filter; using drawtext fallback for generated cues.",
            file=sys.stderr,
        )
        return (
            None,
            write_drawtext_subtitle_filters(
                cues,
                temp_dir,
                width=width,
                height=height,
            ),
        )

    sidecar = copy_subtitle_sidecar(subtitle_file, output)
    message = (
        "This ffmpeg build has neither 'subtitles' nor 'drawtext' filters, "
        f"so subtitles cannot be burned in. Video will be rendered without "
        f"burned subtitles; sidecar saved: {sidecar}"
    )
    if args.strict_subtitles:
        raise CliError(message)
    print(f"Warning: {message}", file=sys.stderr)
    return None, []


def escape_filter_path(path: Path) -> str:
    value = str(path)
    return value.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def fmt_seconds(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def scale_crop_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )


def rotation_filters(degrees: int) -> list[str]:
    normalized = degrees % 360
    if normalized == 0:
        return []
    if normalized == 90:
        return ["transpose=1"]
    if normalized == 180:
        return ["hflip", "vflip"]
    if normalized == 270:
        return ["transpose=2"]
    raise CliError("--main-rotate and --second-rotate must be multiples of 90 degrees.")


def second_camera_effect_filters(effect: str) -> list[str]:
    if effect == "off":
        return []

    filters: list[str] = []
    if effect in {"vhs", "vhs-glitch"}:
        filters.extend(
            [
                "eq=contrast=1.07:saturation=0.86:brightness=0.012",
                "noise=alls=8:allf=t+u",
                "drawgrid=width=iw:height=4:thickness=1:color=black@0.10",
            ]
        )
    if effect in {"glitch", "vhs-glitch"}:
        filters.extend(
            [
                "rgbashift=rh=2:bh=-2:edge=wrap",
                "chromashift=cbh=1:crh=-1:edge=wrap",
            ]
        )
    return filters


def audio_normalization_filters(
    *,
    mode: str,
    target_lufs: float,
    true_peak: float,
    loudness_range: float,
) -> list[str]:
    loudnorm = (
        "loudnorm="
        f"I={target_lufs:.1f}:"
        f"TP={true_peak:.1f}:"
        f"LRA={loudness_range:.1f}:"
        "linear=false"
    )
    return [loudnorm, "aresample=async=1:first_pts=0"]


def audio_gain_filters(gain_db: float) -> list[str]:
    if abs(gain_db) < 0.001:
        return []
    return [f"volume={gain_db:.2f}dB"]


def segment_filter(
    segment: Segment,
    index: int,
    *,
    width: int,
    height: int,
    fps: float,
    second_zoom: float,
    second_effect: str,
    second_denoise: bool,
    main_rotate: int,
    second_rotate: int,
    source_start: float | None = None,
    source_end: float | None = None,
) -> str:
    input_index = 0 if segment.camera == CAM_MAIN else 1
    rotate = main_rotate if segment.camera == CAM_MAIN else second_rotate
    start = segment.start if source_start is None else source_start
    end = segment.end if source_end is None else source_end
    trim = (
        f"[{input_index}:v]trim=start={fmt_seconds(start)}:"
        f"end={fmt_seconds(end)},setpts=PTS-STARTPTS,settb=AVTB"
    )

    if segment.camera == CAM_MAIN:
        filters = [
            *rotation_filters(rotate),
            scale_crop_filter(width, height),
            f"fps={fps:.3f}",
            "format=yuv420p",
        ]
    else:
        zoom = max(1.0, second_zoom)
        zoom_width = even(math.ceil(width * zoom))
        zoom_height = even(math.ceil(height * zoom))
        filters = [
            *rotation_filters(rotate),
            scale_crop_filter(width, height),
            *(["hqdn3d=1.5:1.5:3:3"] if second_denoise else []),
            "unsharp=5:5:0.55:3:3:0.20",
            f"scale={zoom_width}:{zoom_height}",
            f"crop={width}:{height}",
            "setsar=1",
            *second_camera_effect_filters(second_effect),
            f"fps={fps:.3f}",
            "format=yuv420p",
        ]

    return f"{trim},{','.join(filters)}[v{index}]"


def parse_transition_styles(raw: str) -> list[str]:
    styles = [style.strip() for style in raw.split(",") if style.strip()]
    safe = [style for style in styles if re.fullmatch(r"[a-zA-Z0-9_]+", style)]
    return safe or ["fade"]


def choose_transition_styles(
    count: int,
    styles: Sequence[str],
    *,
    mode: str,
    seed: int | None,
) -> list[str]:
    if count <= 0:
        return []
    safe_styles = list(styles) or ["fade"]
    if mode == "random":
        rng = random.Random(seed) if seed is not None else random.SystemRandom()
        return [rng.choice(safe_styles) for _ in range(count)]
    return [safe_styles[index % len(safe_styles)] for index in range(count)]


def effective_transition_duration(segments: Sequence[Segment], requested: float) -> float:
    if len(segments) < 2 or requested <= 0:
        return 0.0
    min_segment = min(segment.duration for segment in segments)
    return max(0.0, min(requested, min_segment * 0.45))


def segment_source_bounds(
    segments: Sequence[Segment],
    *,
    duration: float,
    transition_duration: float,
) -> list[tuple[float, float]]:
    if transition_duration <= 0 or len(segments) < 2:
        return [(segment.start, segment.end) for segment in segments]
    overlap = transition_duration / 2.0
    bounds: list[tuple[float, float]] = []
    for index, segment in enumerate(segments):
        start = segment.start - overlap if index > 0 else segment.start
        end = segment.end + overlap if index < len(segments) - 1 else segment.end
        bounds.append((max(0.0, start), min(duration, end)))
    return bounds


def build_filter_complex(
    segments: Sequence[Segment],
    *,
    width: int,
    height: int,
    fps: float,
    second_zoom: float,
    second_effect: str,
    audio_source: AudioSource | None,
    duration: float,
    subtitle_file: Path | None,
    second_denoise: bool = False,
    subtitle_draw_filters: Sequence[str] | None = None,
    transition_duration: float = 0.0,
    transition_styles: Sequence[str] | None = None,
    transition_mode: str = "cycle",
    transition_seed: int | None = None,
    normalize_audio: bool = False,
    audio_normalize_mode: str = "loudnorm",
    audio_target_lufs: float = -16.0,
    audio_true_peak: float = -1.5,
    audio_lra: float = 11.0,
    audio_gain_db: float = 0.0,
    copy_audio: bool = False,
    main_rotate: int = 0,
    second_rotate: int = 0,
) -> str:
    transition_duration = effective_transition_duration(segments, transition_duration)
    source_bounds = segment_source_bounds(
        segments,
        duration=duration,
        transition_duration=transition_duration,
    )
    filter_lines = [
        segment_filter(
            segment,
            index,
            width=width,
            height=height,
            fps=fps,
            second_zoom=second_zoom,
            second_effect=second_effect,
            second_denoise=second_denoise,
            main_rotate=main_rotate,
            second_rotate=second_rotate,
            source_start=source_bounds[index][0],
            source_end=source_bounds[index][1],
        )
        for index, segment in enumerate(segments)
    ]

    if len(segments) == 1:
        video_label = "v0"
    elif transition_duration > 0:
        styles = list(transition_styles or ["fade"])
        selected_styles = choose_transition_styles(
            len(segments) - 1,
            styles,
            mode=transition_mode,
            seed=transition_seed,
        )
        video_label = "v0"
        accumulated = source_bounds[0][1] - source_bounds[0][0]
        for index in range(1, len(segments)):
            out_label = f"xf{index}"
            style = selected_styles[index - 1]
            offset = max(0.0, accumulated - transition_duration)
            filter_lines.append(
                f"[{video_label}][v{index}]xfade=transition={style}:"
                f"duration={fmt_seconds(transition_duration)}:"
                f"offset={fmt_seconds(offset)}[{out_label}]"
            )
            video_label = out_label
            accumulated += (source_bounds[index][1] - source_bounds[index][0]) - transition_duration
    else:
        concat_inputs = "".join(f"[v{index}]" for index in range(len(segments)))
        filter_lines.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vcat]")
        video_label = "vcat"

    current_label = video_label
    if subtitle_file:
        escaped = escape_filter_path(subtitle_file)
        filter_lines.append(f"[{current_label}]subtitles=filename='{escaped}'[vsub]")
        current_label = "vsub"
    elif subtitle_draw_filters:
        filter_lines.append(f"[{current_label}]{','.join(subtitle_draw_filters)}[vsub]")
        current_label = "vsub"

    filter_lines.append(f"[{current_label}]format=yuv420p[vout]")

    if audio_source and not copy_audio:
        audio_filters = [
            f"[{audio_source.input_index}:a:0]atrim=start=0:end={fmt_seconds(duration)}",
            "asetpts=PTS-STARTPTS",
        ]
        if normalize_audio:
            audio_filters.extend(
                audio_normalization_filters(
                    mode=audio_normalize_mode,
                    target_lufs=audio_target_lufs,
                    true_peak=audio_true_peak,
                    loudness_range=audio_lra,
                )
            )
        audio_filters.extend(audio_gain_filters(audio_gain_db))
        filter_lines.append(f"{','.join(audio_filters)}[aout]")

    return ";".join(filter_lines)


def render_montage(
    args: argparse.Namespace,
    *,
    main: Path,
    second: Path,
    output: Path,
    segments: Sequence[Segment],
    audio_source: AudioSource | None,
    duration: float,
    subtitle_file: Path | None,
    subtitle_draw_filters: Sequence[str] | None,
    width: int,
    height: int,
    fps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [TOOL_FFMPEG, "-y", "-hide_banner", "-i", str(main), "-i", str(second)]
    if audio_source and not audio_source.from_existing_video_input:
        cmd.extend(["-i", str(audio_source.path)])

    filter_complex = build_filter_complex(
        segments,
        width=width,
        height=height,
        fps=fps,
        second_zoom=args.second_zoom,
        second_effect=args.second_effect,
        second_denoise=args.second_denoise,
        audio_source=audio_source,
        duration=duration,
        subtitle_file=subtitle_file,
        subtitle_draw_filters=subtitle_draw_filters,
        transition_duration=args.transition_duration,
        transition_styles=parse_transition_styles(args.transition_styles),
        transition_mode=args.transition_mode,
        transition_seed=args.transition_seed,
        normalize_audio=args.normalize_audio,
        audio_normalize_mode=args.audio_normalize_mode,
        audio_target_lufs=args.audio_target_lufs,
        audio_true_peak=args.audio_true_peak,
        audio_lra=args.audio_lra,
        audio_gain_db=args.audio_gain_db,
        copy_audio=(args.audio_mode == "copy"),
        main_rotate=args.main_rotate,
        second_rotate=args.second_rotate,
    )
    cmd.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
    if audio_source:
        if args.audio_mode == "copy":
            cmd.extend(
                [
                    "-map",
                    f"{audio_source.input_index}:a:0",
                    "-c:a",
                    "copy",
                    "-shortest",
                ]
            )
        else:
            cmd.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", args.audio_bitrate])
    else:
        cmd.append("-an")

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata:s:v:0",
            "rotate=0",
            "-max_muxing_queue_size",
            "1024",
            str(output),
        ]
    )
    run_cmd(cmd, dry_run=args.dry_run)


def choose_split_point(duration: float, silences: Sequence[Silence]) -> float:
    target = duration / 2.0
    middle = [
        silence
        for silence in silences
        if duration * 0.35 <= silence.midpoint <= duration * 0.65
    ]
    if not middle:
        middle = [
            silence
            for silence in silences
            if duration * 0.25 <= silence.midpoint <= duration * 0.75
        ]
    if not middle:
        return target

    best = max(middle, key=lambda silence: (silence.duration, -abs(silence.midpoint - target)))
    return min(max(best.midpoint, 1.0), duration - 1.0)


def split_output(
    args: argparse.Namespace,
    output: Path,
    *,
    duration: float,
    silences: Sequence[Silence],
) -> tuple[Path, Path] | None:
    if args.no_split or duration <= args.max_part_duration:
        return None

    split_at = choose_split_point(duration, silences)
    part1 = output.with_name(f"{output.stem}_part1{output.suffix}")
    part2 = output.with_name(f"{output.stem}_part2{output.suffix}")

    run_cmd(
        [
            TOOL_FFMPEG,
            "-y",
            "-hide_banner",
            "-i",
            str(output),
            "-t",
            fmt_seconds(split_at),
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-c:a",
            "copy" if args.audio_mode == "copy" else "aac",
            *( [] if args.audio_mode == "copy" else ["-b:a", args.audio_bitrate] ),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(part1),
        ],
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            TOOL_FFMPEG,
            "-y",
            "-hide_banner",
            "-i",
            str(output),
            "-ss",
            fmt_seconds(split_at),
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-c:a",
            "copy" if args.audio_mode == "copy" else "aac",
            *( [] if args.audio_mode == "copy" else ["-b:a", args.audio_bitrate] ),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(part2),
        ],
        dry_run=args.dry_run,
    )
    return part1, part2


def print_timeline_summary(segments: Sequence[Segment], silences: Sequence[Silence]) -> None:
    print(f"Detected pauses: {len(silences)}")
    print(f"Timeline segments: {len(segments)}")
    for segment in segments[:12]:
        print(
            f"  {segment.camera:6s} {fmt_seconds(segment.start)} -> "
            f"{fmt_seconds(segment.end)} ({fmt_seconds(segment.duration)}s)"
        )
    if len(segments) > 12:
        print(f"  ... {len(segments) - 12} more segments")


def output_dimensions(
    args: argparse.Namespace,
    main_info: MediaInfo,
) -> tuple[int, int, float]:
    width = even(args.width or 1080)
    height = even(args.height or 1920)
    fps = args.fps or main_info.fps or 30.0
    if width <= 0 or height <= 0 or fps <= 0:
        raise CliError("Invalid output dimensions or FPS.")
    return width, height, fps


def validate_subtitle_options(args: argparse.Namespace) -> None:
    if args.subtitle_words <= 0:
        raise CliError("--subtitle-words must be greater than zero.")
    if args.subtitle_min_duration <= 0:
        raise CliError("--subtitle-min-duration must be greater than zero.")
    if args.subtitle_max_duration <= 0:
        raise CliError("--subtitle-max-duration must be greater than zero.")
    if args.subtitle_max_duration < args.subtitle_min_duration:
        raise CliError("--subtitle-max-duration cannot be smaller than --subtitle-min-duration.")
    if args.subtitle_hold_extension < 0:
        raise CliError("--subtitle-hold-extension must be non-negative.")
    if args.subtitle_gap < 0:
        raise CliError("--subtitle-gap must be non-negative.")


def validate_audio_normalization_options(args: argparse.Namespace) -> None:
    if args.audio_mode == "copy":
        return
    if not -60.0 <= args.audio_gain_db <= 60.0:
        raise CliError("--audio-gain-db must be between -60 and 60.")
    if not args.normalize_audio:
        return
    if not -70.0 <= args.audio_target_lufs <= -5.0:
        raise CliError("--audio-target-lufs must be between -70 and -5.")
    if not -9.0 <= args.audio_true_peak <= 0.0:
        raise CliError("--audio-true-peak must be between -9 and 0.")
    if not 1.0 <= args.audio_lra <= 50.0:
        raise CliError("--audio-lra must be between 1 and 50.")

    filters = available_ffmpeg_filters()
    if "loudnorm" not in filters:
        raise CliError("This ffmpeg build has no loudnorm filter.")


def validate_rotation_options(args: argparse.Namespace) -> None:
    for option_name, value in (
        ("--main-rotate", args.main_rotate),
        ("--second-rotate", args.second_rotate),
    ):
        if value % 90 != 0:
            raise CliError(f"{option_name} must be a multiple of 90 degrees.")


def run_pipeline(args: argparse.Namespace) -> int:
    configure_media_tools(args)
    main, second = resolve_input_paths(args)

    require_tool(TOOL_FFMPEG)
    require_tool(TOOL_FFPROBE)
    if args.min_transitions < 0 or args.max_transitions < 0:
        raise CliError("--min-transitions and --max-transitions must be non-negative.")
    if args.min_transitions > args.max_transitions:
        raise CliError("--min-transitions cannot be greater than --max-transitions.")
    if args.transition_duration < 0:
        raise CliError("--transition-duration must be non-negative.")
    validate_subtitle_options(args)
    validate_audio_normalization_options(args)
    validate_rotation_options(args)

    output = Path(args.output).expanduser().resolve()

    main_info = ffprobe_media(main)
    second_info = ffprobe_media(second)
    audio_source = resolve_audio_source(args, main, second, main_info, second_info)
    audio_info = None
    if audio_source and not audio_source.from_existing_video_input:
        audio_info = ffprobe_media(audio_source.path)

    duration = choose_timeline_duration(args, main_info, second_info, audio_info)
    width, height, fps = output_dimensions(args, main_info)

    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_dir = output.with_suffix("")
        temp_dir = temp_dir.with_name(f"{temp_dir.name}_work")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="video2_")
        temp_dir = Path(temp_owner.name)

    try:
        analysis_audio: Path | None = None
        silences: list[Silence] = []
        if audio_source:
            print(f"Using audio from {audio_source.label}: {audio_source.path}")
            analysis_audio = extract_analysis_audio(
                audio_source,
                temp_dir,
                duration,
                dry_run=args.dry_run,
            )
            silences = detect_silences(
                analysis_audio,
                duration,
                noise=args.silence_noise,
                min_silence=args.min_silence,
                dry_run=args.dry_run,
            )
        else:
            print("No audio stream was provided; camera switches will use only timers.")

        segments = build_segments(
            duration,
            silences,
            main_pause_threshold=args.main_pause_threshold,
            main_max_span=args.main_max_span,
            second_pause_threshold=args.second_pause_threshold,
            second_max_span=args.second_max_span,
            min_transitions=args.min_transitions,
            max_transitions=args.max_transitions,
        )
        print_timeline_summary(segments, silences)

        cues, ready_subtitle_file = load_subtitle_cues(
            args,
            analysis_audio,
            temp_dir,
            duration,
            silences,
            dry_run=args.dry_run,
        )
        subtitle_file = ready_subtitle_file
        if cues is not None:
            if not cues and args.sub and not args.dry_run:
                raise CliError("No subtitle cues were generated.")
            if args.subtitle_lowercase:
                cues = lowercase_subtitle_cues(cues)
            cues = adjust_subtitle_cues(
                cues,
                duration=duration,
                min_duration=args.subtitle_min_duration,
                max_duration=args.subtitle_max_duration,
                hold_extension=args.subtitle_hold_extension,
                gap=args.subtitle_gap,
            )
            cues = apply_keyword_highlights_to_cues(cues, args, dry_run=args.dry_run)
            subtitle_file = write_rotating_ass(
                cues,
                temp_dir / "rotating_subtitles.ass",
                width=width,
                height=height,
            )
            print(f"Subtitle cues: {len(cues)}")

        subtitle_draw_filters: list[str] = []
        subtitle_file, subtitle_draw_filters = prepare_subtitle_filters(
            args,
            output=output,
            subtitle_file=subtitle_file,
            cues=cues,
            temp_dir=temp_dir,
            width=width,
            height=height,
        )

        render_montage(
            args,
            main=main,
            second=second,
            output=output,
            segments=segments,
            audio_source=audio_source,
            duration=duration,
            subtitle_file=subtitle_file,
            subtitle_draw_filters=subtitle_draw_filters,
            width=width,
            height=height,
            fps=fps,
        )
        split_paths = split_output(args, output, duration=duration, silences=silences)
        print(f"Done: {output}")
        if split_paths:
            print(f"Split parts: {split_paths[0]} and {split_paths[1]}")
        if args.keep_temp:
            print(f"Temporary files kept in: {temp_dir}")
    finally:
        if temp_owner is not None:
            temp_owner.cleanup()

    return 0


def run_self_test() -> int:
    silences = [Silence(8.0, 11.0), Silence(25.0, 27.0), Silence(40.0, 44.5)]
    segments = build_segments(
        58.0,
        silences,
        main_pause_threshold=2.5,
        main_max_span=15.0,
        second_pause_threshold=1.5,
        second_max_span=10.0,
    )
    assert segments[0] == Segment(0.0, 10.5, CAM_MAIN)
    assert segments[1].camera == CAM_SECOND
    assert all(segment.end > segment.start for segment in segments)

    long_pause_segments = build_segments(
        30.0,
        [Silence(5.0, 20.0)],
        main_pause_threshold=2.5,
        main_max_span=15.0,
        second_pause_threshold=1.5,
        second_max_span=10.0,
    )
    assert len(long_pause_segments) <= 4
    assert long_pause_segments[0] == Segment(0.0, 7.5, CAM_MAIN)
    assert long_pause_segments[1] == Segment(7.5, 17.5, CAM_SECOND)

    cues = cues_from_plain_text(
        "First subtitle phrase. Second subtitle phrase appears next.",
        12.0,
        [Silence(4.0, 4.8)],
    )
    assert cues
    assert cues[0].start >= 0
    assert cues[-1].end <= 12.0
    adjusted_cues = adjust_subtitle_cues(
        cues,
        duration=12.0,
        min_duration=2.8,
        max_duration=8.5,
        hold_extension=1.6,
        gap=0.08,
    )
    assert adjusted_cues[0].duration >= cues[0].duration
    timed_words = [
        TimedWord("First", 0.2, 0.5),
        TimedWord("subtitle", 0.55, 0.95),
        TimedWord("phrase", 1.0, 1.35),
        TimedWord("Second", 3.0, 3.35),
        TimedWord("subtitle", 3.4, 3.8),
        TimedWord("phrase", 3.85, 4.2),
    ]
    aligned_cues = ready_text_with_timed_words_to_cues(
        "First subtitle phrase. Second subtitle phrase.",
        timed_words,
        max_words=3,
    )
    assert aligned_cues[0].start == 0.2
    assert aligned_cues[1].start == 3.0
    lowered_cues = lowercase_subtitle_cues(
        [SubtitleCue(0.0, 1.0, "HELLO <b>МИР</b>\nNEXT")]
    )
    assert lowered_cues[0].text == "hello <b>мир</b>\nnext"
    rotated_width, rotated_height, _ = output_dimensions(
        argparse.Namespace(width=None, height=None, fps=None, main_rotate=90),
        MediaInfo(Path("main.mov"), 10.0, 1920, 1080, 30.0, True),
    )
    assert (rotated_width, rotated_height) == (1080, 1920)

    with tempfile.TemporaryDirectory(prefix="video2_self_test_") as temp_raw:
        ass_path = Path(temp_raw) / "test.ass"
        write_rotating_ass(adjusted_cues, ass_path, width=1080, height=1920)
        ass_text = ass_path.read_text(encoding="utf-8")
        assert "Style: Default" in ass_text
        assert "Dialogue:" in ass_text
        assert ",64,64,322,1" in ass_text
        upper_ass_path = Path(temp_raw) / "upper.ass"
        lower_ass_path = Path(temp_raw) / "lower.ass"
        write_rotating_ass(
            [SubtitleCue(0.0, 1.0, "HELLO\nМИР")],
            upper_ass_path,
            width=1080,
            height=1920,
        )
        write_lowercase_ass_copy(upper_ass_path, lower_ass_path)
        lower_ass_text = lower_ass_path.read_text(encoding="utf-8")
        assert r"{\fad(80,180)}hello\Nмир" in lower_ass_text

        filter_text = build_filter_complex(
            segments[:2],
            width=1080,
            height=1920,
            fps=30.0,
            second_zoom=1.045,
            second_effect="vhs",
            audio_source=AudioSource(Path("speech.wav"), 2, "external", False),
            duration=20.0,
            subtitle_file=ass_path,
            second_denoise=True,
            second_rotate=90,
        )
        assert "concat=n=2:v=1:a=0" in filter_text
        assert "hqdn3d" in filter_text
        assert "noise=alls=8" in filter_text
        assert "transpose=1" in filter_text
        assert "[2:a:0]atrim" in filter_text
        assert "subtitles=filename=" in filter_text
        normalized_filter_text = build_filter_complex(
            segments[:2],
            width=1080,
            height=1920,
            fps=30.0,
            second_zoom=1.045,
            second_effect="vhs",
            audio_source=AudioSource(Path("speech.wav"), 2, "external", False),
            duration=20.0,
            subtitle_file=None,
            normalize_audio=True,
            audio_normalize_mode="speech",
            audio_gain_db=11.0,
        )
        assert "dynaudnorm" not in normalized_filter_text
        assert "loudnorm=I=-16.0:TP=-1.5:LRA=11.0:linear=false" in normalized_filter_text
        assert "volume=11.00dB" in normalized_filter_text
        random_filter_text = build_filter_complex(
            segments[:4],
            width=1080,
            height=1920,
            fps=30.0,
            second_zoom=1.08,
            second_effect="vhs-glitch",
            audio_source=None,
            duration=30.0,
            subtitle_file=None,
            transition_duration=0.45,
            transition_styles=["fade", "smoothleft"],
            transition_mode="random",
            transition_seed=7,
        )
        assert "rgbashift" in random_filter_text
        assert random_filter_text.count("xfade=transition=") == 3

    split_at = choose_split_point(80.0, [Silence(38.0, 43.0), Silence(10.0, 16.0)])
    assert 38.0 <= split_at <= 43.0
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
