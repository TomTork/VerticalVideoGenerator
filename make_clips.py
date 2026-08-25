#!/usr/bin/env python3
"""Render a list of vertical shorts from one source video.

The clip list is a JSON array of {"name", "start", "end"} objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vertical_speaker_layout import build_parser as layout_parser
from vertical_speaker_layout import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path)
    parser.add_argument("clips_json", type=Path)
    parser.add_argument("--words-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("clips"))
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    clips = json.loads(args.clips_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, clip in enumerate(clips, start=1):
        output = args.output_dir / f"{index:02d}_{clip['name']}.mp4"
        layout_args = layout_parser().parse_args(
            [
                str(args.input_video),
                "--output",
                str(output),
                "--words-json",
                str(args.words_json),
                "--start",
                str(clip["start"]),
                "--duration",
                str(round(float(clip["end"]) - float(clip["start"]), 3)),
                *args.extra,
            ]
        )
        run_pipeline(layout_args)
    print(f"Rendered {len(clips)} clips into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
