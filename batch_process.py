#!/usr/bin/env python3
"""Run the video2 pipeline for many source folders.

The script finds folders containing the repeated input set:

  cat1.* or cat.*
  cat2.*
  text.txt

For every matching folder it runs the two commands from EXECUTION.md with
paths adjusted for that folder. Temporary synchronized videos are kept outside
the results tree; only final montage outputs are written under results/.
Optional per-folder overrides can be stored in help.json.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SKIP_DIR_NAMES = {".git", ".hg", ".svn", ".idea", ".venv", "__pycache__"}
VIDEO_EXTENSIONS = {
    ".3g2",
    ".3gp",
    ".avi",
    ".dv",
    ".hevc",
    ".m2t",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".webm",
}


class ConfigError(RuntimeError):
    """A user-facing configuration error."""


@dataclass(frozen=True)
class FolderConfig:
    video_a_name: str | None = None
    video_b_name: str | None = None
    text_name: str | None = None
    output_name: str | None = None
    align_args: tuple[str, ...] = ()
    main_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Job:
    source_dir: Path
    relative_dir: Path
    video_a: Path
    video_b: Path
    text_file: Path | None
    output_file: Path
    align_args: tuple[str, ...]
    main_args: tuple[str, ...]


@dataclass
class Summary:
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively process folders with cat1.*, cat.* / cat2.* and text.txt, "
            "writing final outputs to results/ with the same hierarchy."
        )
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="Folder or folders to scan recursively.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root folder for final outputs. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--cat1-name",
        help=(
            "Exact first camera filename. By default the scanner uses "
            "--cat1-patterns instead."
        ),
    )
    parser.add_argument(
        "--cat2-name",
        help=(
            "Exact second camera filename. By default the scanner uses "
            "--cat2-patterns instead."
        ),
    )
    parser.add_argument(
        "--cat1-patterns",
        default="cat1.*,cat.*",
        help=(
            "Comma-separated first camera filename patterns. Matching is "
            "case-insensitive and limited to known video extensions."
        ),
    )
    parser.add_argument(
        "--cat2-patterns",
        default="cat2.*",
        help=(
            "Comma-separated second camera filename patterns. Matching is "
            "case-insensitive and limited to known video extensions."
        ),
    )
    parser.add_argument("--text-name", default="text.txt", help="Subtitle text filename.")
    parser.add_argument(
        "--output-name",
        default="cat_result.m4v",
        help="Final montage filename created inside each results subfolder.",
    )
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        help="Process folders without text.txt by falling back to Whisper transcription.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a job when its final output or split outputs already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining folders after a failed job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running ffmpeg or writing outputs.",
    )
    parser.add_argument(
        "--normalize-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass audio loudness normalization options to main.py.",
    )
    parser.add_argument(
        "--audio-normalize-mode",
        choices=("loudnorm", "speech"),
        default="loudnorm",
        help="Normalization mode passed to main.py.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("filter", "copy"),
        default="filter",
        help="Audio mode passed to main.py. copy disables all audio filtering and re-encoding.",
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
        help="Target LUFS passed to main.py when audio normalization is enabled.",
    )
    parser.add_argument(
        "--audio-gain-db",
        type=float,
        default=0.0,
        help="Simple audio gain in dB passed to main.py.",
    )
    parser.add_argument(
        "--audio-true-peak",
        type=float,
        default=-1.5,
        help="Target true peak passed to main.py when audio normalization is enabled.",
    )
    parser.add_argument(
        "--audio-lra",
        type=float,
        default=11.0,
        help="Target loudness range passed to main.py when audio normalization is enabled.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run align_audio_starts.py and main.py.",
    )
    return parser


def resolve_path(path: Path, *, base: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def relative_output_dir(source_dir: Path, scan_root: Path, project_root: Path) -> Path:
    if is_relative_to(source_dir, project_root):
        return source_dir.relative_to(project_root)
    if is_relative_to(source_dir, scan_root):
        return Path(scan_root.name) / source_dir.relative_to(scan_root)
    return Path(source_dir.name)


def should_descend(path: Path, results_root: Path) -> bool:
    if path.name in SKIP_DIR_NAMES:
        return False
    if path.name.startswith("."):
        return False
    if path.resolve() == results_root or is_relative_to(path.resolve(), results_root):
        return False
    return True


def split_patterns(raw: str) -> tuple[str, ...]:
    return tuple(pattern.strip() for pattern in raw.split(",") if pattern.strip())


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def pattern_matches(name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(name.lower(), pattern.lower())


def resolve_config_path(source_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    return path.resolve()


def path_config_value(data: object, *keys: str) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def scalar_to_args(flag_name: str, value: object) -> list[str]:
    flag = "--" + flag_name.strip().replace("_", "-").lstrip("-")
    if isinstance(value, bool):
        return [flag] if value else []
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [flag, str(value)]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            if isinstance(item, bool):
                if item:
                    output.append(flag)
            elif item is not None:
                output.extend([flag, str(item)])
        return output
    raise ConfigError(f"Unsupported value for {flag}: {value!r}")


def options_to_args(options: object) -> list[str]:
    if options is None:
        return []
    if not isinstance(options, dict):
        raise ConfigError("help.json options must be objects.")

    output: list[str] = []
    for key, value in options.items():
        output.extend(scalar_to_args(str(key), value))
    return output


def raw_args(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return [str(item) for item in value]
    raise ConfigError(f"help.json {field_name} must be a string or a list of strings.")


def camera_rotation_args(camera: str, value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise ConfigError(f"help.json main.{camera} must be an object.")
    rotate = value.get("rotate")
    if rotate is None:
        return []
    flag = "--main-rotate" if camera in {"video1", "cat1", "first"} else "--second-rotate"
    return [flag, str(rotate)]


def audio_section_args(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise ConfigError("help.json audio must be an object.")

    output: list[str] = []
    for key, raw in value.items():
        normalized = key.strip().replace("_", "-")
        if normalized == "db":
            output.extend(["--audio-gain-db", str(raw)])
        elif normalized in {"gain", "gain-db", "audio-gain-db"}:
            output.extend(["--audio-gain-db", str(raw)])
        elif normalized in {"normalize", "normalize-audio"}:
            if not isinstance(raw, bool):
                raise ConfigError("help.json audio.normalize must be true or false.")
            output.append("--normalize-audio" if raw else "--no-normalize-audio")
        elif normalized in {"normalize-mode", "mode"}:
            mode = str(raw)
            if mode in {"copy", "filter"}:
                output.extend(["--audio-mode", mode])
            else:
                output.extend(["--audio-normalize-mode", mode])
        elif normalized in {"copy", "copy-audio"}:
            if not isinstance(raw, bool):
                raise ConfigError("help.json audio.copy must be true or false.")
            if raw:
                output.append("--copy-audio")
            else:
                output.extend(["--audio-mode", "filter"])
        elif normalized in {"target-lufs", "true-peak", "lra"}:
            output.extend([f"--audio-{normalized}", str(raw)])
        else:
            raise ConfigError(f"Unsupported help.json audio option: {key}")
    return output


def section_to_args(section: object, *, section_name: str) -> list[str]:
    if section is None:
        return []
    if isinstance(section, (str, list)):
        return raw_args(section, section_name)
    if not isinstance(section, dict):
        raise ConfigError(f"help.json {section_name} must be an object, string, or list.")

    reserved = {
        "args",
        "options",
        "video1",
        "video2",
        "cat1",
        "cat2",
        "first",
        "second",
        "audio",
    }
    output = raw_args(section.get("args"), f"{section_name}.args")
    output.extend(options_to_args(section.get("options")))
    for camera in ("video1", "cat1", "first", "video2", "cat2", "second"):
        output.extend(camera_rotation_args(camera, section.get(camera)))
    output.extend(audio_section_args(section.get("audio")))
    for key, value in section.items():
        if key in reserved:
            continue
        if section_name == "main" and key == "db":
            output.extend(["--audio-gain-db", str(value)])
            continue
        if isinstance(value, (dict, tuple)):
            raise ConfigError(f"Unsupported nested value in help.json {section_name}.{key}.")
        output.extend(scalar_to_args(key, value))
    return output


def load_folder_config(source_dir: Path) -> FolderConfig:
    config_path = source_dir / "help.json"
    if not config_path.exists():
        return FolderConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a JSON object.")

    inputs = data.get("inputs")
    if inputs is not None and not isinstance(inputs, dict):
        raise ConfigError(f"{config_path}: inputs must be an object.")

    video_a_name = path_config_value(data, "video1", "cat1", "main_video")
    video_b_name = path_config_value(data, "video2", "cat2", "second_video")
    text_name = path_config_value(data, "text", "text_file", "sub_text_file")
    if isinstance(inputs, dict):
        video_a_name = path_config_value(inputs, "main", "video1", "cat1", "first") or video_a_name
        video_b_name = path_config_value(inputs, "second", "video2", "cat2") or video_b_name
        text_name = path_config_value(inputs, "text", "subtitles", "sub_text_file") or text_name

    output_name = path_config_value(data, "output", "output_name")
    if isinstance(data.get("output"), dict):
        output_name = path_config_value(data["output"], "name", "file", "path") or output_name

    align_args = [
        *raw_args(data.get("align_args"), "align_args"),
        *section_to_args(data.get("align"), section_name="align"),
    ]
    main_args = [
        *raw_args(data.get("main_args"), "main_args"),
        *audio_section_args(data.get("audio")),
        *section_to_args(data.get("main"), section_name="main"),
    ]
    return FolderConfig(
        video_a_name=video_a_name,
        video_b_name=video_b_name,
        text_name=text_name,
        output_name=output_name,
        align_args=tuple(align_args),
        main_args=tuple(main_args),
    )


def find_exact_file(source_dir: Path, name: str) -> Path | None:
    path = resolve_config_path(source_dir, name)
    return path if path.is_file() else None


def find_video_by_patterns(source_dir: Path, patterns: Sequence[str]) -> Path | None:
    matches = [
        child.resolve()
        for child in source_dir.iterdir()
        if is_video_file(child) and any(pattern_matches(child.name, pattern) for pattern in patterns)
    ]
    matches = sorted(set(matches), key=lambda path: path.name.lower())
    if len(matches) > 1:
        print(
            f"Warning: multiple video matches in {source_dir}; using {matches[0].name}. "
            "Set inputs in help.json to choose explicitly.",
            file=sys.stderr,
        )
    return matches[0] if matches else None


def resolve_video_input(
    source_dir: Path,
    *,
    configured_name: str | None,
    exact_name: str | None,
    patterns: Sequence[str],
) -> Path | None:
    if configured_name:
        return find_exact_file(source_dir, configured_name)
    if exact_name:
        return find_exact_file(source_dir, exact_name)
    return find_video_by_patterns(source_dir, patterns)


def resolve_text_file(source_dir: Path, text_name: str, configured_name: str | None) -> Path | None:
    raw_name = configured_name or text_name
    path = resolve_config_path(source_dir, raw_name)
    return path if path.is_file() else None


def output_path_for_job(results_root: Path, relative_dir: Path, output_name: str) -> Path:
    output_path = Path(output_name)
    if output_path.is_absolute():
        raise ConfigError("help.json output path must be relative.")
    return results_root / relative_dir / output_path


def find_jobs(
    scan_roots: Sequence[Path],
    *,
    project_root: Path,
    results_root: Path,
    cat1_name: str | None,
    cat2_name: str | None,
    cat1_patterns: Sequence[str],
    cat2_patterns: Sequence[str],
    text_name: str,
    output_name: str,
    allow_missing_text: bool,
) -> list[Job]:
    jobs: list[Job] = []
    seen: set[Path] = set()

    for scan_root in scan_roots:
        for current, dir_names, file_names in os.walk(scan_root):
            current_dir = Path(current).resolve()
            dir_names[:] = sorted(
                name
                for name in dir_names
                if should_descend((current_dir / name).resolve(), results_root)
            )

            try:
                folder_config = load_folder_config(current_dir)
            except ConfigError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                continue

            video_a = resolve_video_input(
                current_dir,
                configured_name=folder_config.video_a_name,
                exact_name=cat1_name,
                patterns=cat1_patterns,
            )
            video_b = resolve_video_input(
                current_dir,
                configured_name=folder_config.video_b_name,
                exact_name=cat2_name,
                patterns=cat2_patterns,
            )
            text_file = resolve_text_file(current_dir, text_name, folder_config.text_name)
            if video_a is None or video_b is None or (text_file is None and not allow_missing_text):
                continue

            if current_dir in seen:
                continue
            seen.add(current_dir)

            relative_dir = relative_output_dir(current_dir, scan_root, project_root)
            try:
                output_file = output_path_for_job(
                    results_root,
                    relative_dir,
                    folder_config.output_name or output_name,
                )
            except ConfigError as exc:
                print(f"Warning: {current_dir}: {exc}", file=sys.stderr)
                continue
            jobs.append(
                Job(
                    source_dir=current_dir,
                    relative_dir=relative_dir,
                    video_a=video_a,
                    video_b=video_b,
                    text_file=text_file,
                    output_file=output_file,
                    align_args=folder_config.align_args,
                    main_args=folder_config.main_args,
                )
            )

    return sorted(jobs, key=lambda job: str(job.relative_dir))


def command_env(python_executable: str) -> dict[str, str]:
    env = os.environ.copy()
    resolved = shutil.which(python_executable) or python_executable
    python_path = Path(resolved).expanduser()
    if python_path.parent != Path("."):
        env["PATH"] = f"{python_path.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    print("+", shlex.join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=cwd, env=env, check=True)


def existing_outputs(output_file: Path) -> list[Path]:
    part1 = output_file.with_name(f"{output_file.stem}_part1{output_file.suffix}")
    part2 = output_file.with_name(f"{output_file.stem}_part2{output_file.suffix}")
    return [path for path in (output_file, part1, part2) if path.exists()]


def build_align_command(args: argparse.Namespace, job: Job, synced_a: Path, synced_b: Path) -> list[str]:
    cmd = [
        args.python,
        str(PROJECT_ROOT / "align_audio_starts.py"),
        str(job.video_a),
        str(job.video_b),
        "--output-a",
        str(synced_a),
        "--output-b",
        str(synced_b),
        "--analysis-duration",
        "160",
        "--max-offset",
        "45",
        "--min-overlap",
        "20",
    ]
    cmd.extend(job.align_args)
    return cmd


def build_main_command(args: argparse.Namespace, job: Job, synced_a: Path, synced_b: Path) -> list[str]:
    cmd = [
        args.python,
        str(PROJECT_ROOT / "main.py"),
        str(synced_a),
        str(synced_b),
        "-o",
        str(job.output_file),
        "--audio",
        "1",
        "--sub",
        "--transcriber",
        "whisper",
        "--whisper-model",
        "base",
        "--whisper-language",
        "ru",
        "--subtitle-lowercase",
        "--main-max-span",
        "8",
        "--second-max-span",
        "6",
        "--min-transitions",
        "8",
        "--max-transitions",
        "12",
        "--transition-mode",
        "random",
        "--transition-styles",
        "smoothleft,smoothright,circleopen,fade,wipeleft,wiperight",
        "--second-effect",
        "vhs-glitch",
        "--second-zoom",
        "1.0",
        "--no-second-denoise",
    ]
    if args.normalize_audio:
        cmd.extend(
            [
                "--normalize-audio",
                "--audio-normalize-mode",
                args.audio_normalize_mode,
                "--audio-target-lufs",
                str(args.audio_target_lufs),
                "--audio-true-peak",
                str(args.audio_true_peak),
                "--audio-lra",
                str(args.audio_lra),
            ]
        )
    if abs(args.audio_gain_db) >= 0.001:
        cmd.extend(["--audio-gain-db", str(args.audio_gain_db)])
    if args.audio_mode == "copy":
        cmd.append("--copy-audio")
    if job.text_file is not None:
        cmd.extend(["--sub-text-file", str(job.text_file)])
    cmd.extend(job.main_args)
    return cmd


def process_job(args: argparse.Namespace, job: Job, env: dict[str, str]) -> bool:
    existing = existing_outputs(job.output_file)
    if args.skip_existing and existing:
        print(f"Skipping existing result: {job.relative_dir}", flush=True)
        return False

    print(f"\n=== Processing {job.relative_dir} ===", flush=True)
    if not args.dry_run:
        job.output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video2_batch_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        synced_a = temp_dir / "cat1_synced.m4v"
        synced_b = temp_dir / "cat2_synced.m4v"

        run_cmd(
            build_align_command(args, job, synced_a, synced_b),
            cwd=PROJECT_ROOT,
            env=env,
            dry_run=args.dry_run,
        )
        run_cmd(
            build_main_command(args, job, synced_a, synced_b),
            cwd=PROJECT_ROOT,
            env=env,
            dry_run=args.dry_run,
        )

    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = PROJECT_ROOT
    results_root = resolve_path(args.results_root, base=project_root)
    scan_roots = [resolve_path(folder, base=Path.cwd()) for folder in args.folders]

    missing = [path for path in scan_roots if not path.is_dir()]
    if missing:
        parser.error("Folders do not exist: " + ", ".join(str(path) for path in missing))

    jobs = find_jobs(
        scan_roots,
        project_root=project_root,
        results_root=results_root,
        cat1_name=args.cat1_name,
        cat2_name=args.cat2_name,
        cat1_patterns=split_patterns(args.cat1_patterns),
        cat2_patterns=split_patterns(args.cat2_patterns),
        text_name=args.text_name,
        output_name=args.output_name,
        allow_missing_text=args.allow_missing_text,
    )
    if not jobs:
        print("No matching folders found.", file=sys.stderr)
        return 1

    print(f"Found jobs: {len(jobs)}")
    print(f"Results root: {results_root}")

    env = command_env(args.python)
    summary = Summary()
    for job in jobs:
        try:
            processed = process_job(args, job, env)
            if processed:
                summary.completed += 1
            else:
                summary.skipped += 1
        except subprocess.CalledProcessError as exc:
            summary.failed += 1
            print(f"Failed: {job.relative_dir} (exit code {exc.returncode})", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(
        "\nSummary: "
        f"completed={summary.completed}, skipped={summary.skipped}, failed={summary.failed}"
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
