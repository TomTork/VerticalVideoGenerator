#!/usr/bin/env python3
"""Split a Whisper transcript into consecutive clips that respect sentence boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min", type=float, default=30.0, help="Minimum output length.")
    parser.add_argument("--max", type=float, default=45.0, help="Maximum output length.")
    args = parser.parse_args()

    segments = json.loads(args.words_json.read_text(encoding="utf-8"))["segments"]
    minimum = args.min * args.speed
    maximum = args.max * args.speed

    groups: list[list[dict]] = []
    current: list[dict] = []
    for segment in segments:
        if current and segment["end"] - current[0]["start"] > maximum:
            groups.append(current)
            current = []
        current.append(segment)
        length = segment["end"] - current[0]["start"]
        sentence_end = bool(re.search(r"[.!?…]$", segment["text"].strip()))
        if length >= minimum and sentence_end:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    # A short tail reads as a cut-off thought: pull sentences back from its neighbour.
    while len(groups) > 1 and (groups[-1][-1]["end"] - groups[-1][0]["start"]) < minimum:
        donor = groups[-2]
        if len(donor) < 2:
            break
        moved = donor.pop()
        candidate = groups[-1][-1]["end"] - moved["start"]
        if candidate > maximum:
            donor.append(moved)
            break
        groups[-1].insert(0, moved)

    clips = [
        {
            "name": f"part{index:02d}",
            "start": round(group[0]["start"], 2),
            "end": round(group[-1]["end"], 2),
            "output_seconds": round((group[-1]["end"] - group[0]["start"]) / args.speed, 1),
            "text": " ".join(segment["text"].strip() for segment in group),
        }
        for index, group in enumerate(groups, start=1)
    ]

    args.output.write_text(
        json.dumps(clips, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lengths = [clip["output_seconds"] for clip in clips]
    print(f"{len(clips)} clips, output seconds min={min(lengths)} max={max(lengths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
