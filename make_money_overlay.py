#!/usr/bin/env python3
"""Render a transparent overlay of falling money bills for ffmpeg."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

BILL_COLORS = (
    ((46, 160, 88), (18, 92, 48)),
    ((60, 190, 110), (24, 110, 60)),
    ((32, 140, 76), (12, 78, 40)),
)


def make_bill(width: int, height: int, palette: int) -> Image.Image:
    face, edge = BILL_COLORS[palette % len(BILL_COLORS)]
    bill = Image.new("RGBA", (width, height), (*face, 255))
    draw = ImageDraw.Draw(bill)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(*edge, 255), width=max(2, height // 22))
    inset = max(4, height // 9)
    draw.rectangle(
        [inset, inset, width - inset - 1, height - inset - 1],
        outline=(210, 245, 220, 200),
        width=max(1, height // 40),
    )
    radius = height // 4
    center = (width // 2, height // 2)
    draw.ellipse(
        [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
        outline=(225, 250, 230, 220),
        width=max(2, height // 30),
    )
    bar = max(2, height // 26)
    for offset in (-radius // 2, 0, radius // 2):
        draw.line(
            [center[0] - radius // 2, center[1] + offset, center[0] + radius // 2, center[1] + offset],
            fill=(232, 252, 236, 210),
            width=bar,
        )
    draw.line(
        [center[0], center[1] - radius, center[0], center[1] + radius],
        fill=(232, 252, 236, 230),
        width=bar,
    )
    return bill


def render(
    output: Path,
    *,
    width: int,
    height: int,
    duration: float,
    fps: int,
    count: int,
    seed: int,
    ffmpeg: str,
) -> Path:
    random.seed(seed)
    bill_w = round(width * 0.13)
    bill_h = round(bill_w * 0.46)
    sprites = [make_bill(bill_w, bill_h, index) for index in range(len(BILL_COLORS))]
    bills = [
        {
            "x": random.uniform(-0.05, 1.05) * width,
            "y": random.uniform(-1.4, 1.0) * height,
            "speed": random.uniform(0.32, 0.72) * height,
            "sway": random.uniform(0.02, 0.06) * width,
            "phase": random.uniform(0, math.tau),
            "spin": random.uniform(-70, 70),
            "angle": random.uniform(0, 360),
            "scale": random.uniform(0.65, 1.25),
            "sprite": sprites[index % len(sprites)],
        }
        for index in range(count)
    ]

    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "qtrle",
            "-pix_fmt",
            "argb",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame_index in range(round(duration * fps)):
        t = frame_index / fps
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for bill in bills:
            y = bill["y"] + bill["speed"] * t
            y = (y % (height + 2 * bill_h)) - bill_h
            x = bill["x"] + bill["sway"] * math.sin(bill["phase"] + t * 2.1)
            sprite = bill["sprite"]
            size = (round(sprite.width * bill["scale"]), round(sprite.height * bill["scale"]))
            rotated = sprite.resize(size).rotate(
                bill["angle"] + bill["spin"] * t, expand=True, resample=Image.Resampling.BICUBIC
            )
            frame.alpha_composite(rotated, (round(x - rotated.width / 2), round(y)))
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the money overlay.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--count", type=int, default=34)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        bill = make_bill(120, 56, 0)
        assert bill.size == (120, 56) and bill.mode == "RGBA"
        print("Self-test passed.")
        return 0
    render(
        args.output,
        width=args.width,
        height=args.height,
        duration=args.duration,
        fps=args.fps,
        count=args.count,
        seed=args.seed,
        ffmpeg=args.ffmpeg_bin,
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
