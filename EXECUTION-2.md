# Four-video vertical layout

The renderer accepts four videos in this order:

1. Main video: top 60% by default, stretched to the area without cropping.
2. Bottom video: bottom 40%, stretched to the area without cropping.
3. Left video: 90x90 by default, centered on the 60/40 junction and shifted
   70 pixels downward, and 10 pixels beyond the left edge.
4. Top video: 64x64 by default. It alternates between the top-right and
   top-left corners every 5 seconds, with deterministic random offsets.

Concrete example using the existing `02/cat` files:

```bash
.venv/bin/python four_video_layout.py \
  02/cat/cat1.m4v \
  02/cat/cat2.m4v \
  02/cat/cat2.m4v \
  02/cat/cat2.m4v \
  --subtitle-text-file 02/cat/text.txt \
  --subtitle-language ru \
  --main-ratio 0.60 \
  --main-speed 1.20 \
  --main-exposure 0.12 \
  --main-audio-volume-percent 100 \
  --bottom-audio-volume 0.05 \
  --left-size 90 \
  --left-overflow 10 \
  --top-size 64 \
  --top-jump-interval 5 \
  --top-random-offset 10 \
  --encoder auto \
  --keep-subtitles \
  -o results/02/cat/cat_four_video_result.m4v
```

`--encoder auto` uses Apple VideoToolbox on macOS when available. Only the
main audio is used at full level. Bottom audio is mixed at 5% by default;
left and top audio tracks are ignored.

The bottom, left, and top inputs are looped automatically when they are
shorter than the accelerated main video. Results longer than 60 seconds are
also split into `_part1` and `_part2`. The split point is selected from pauses
near the middle using the same rules as `main.py`.

Useful overrides:

```bash
--main-ratio 0.65
--swap-main-bottom
--main-speed 1.10
--main-exposure 0.18
--main-audio-volume-percent 300
--audio-peak-protection limiter
--bottom-audio-volume 0.08
--left-size 110
--left-overflow 14
--left-y-offset 70
--top-size 72
--top-margin 20
--top-horizontal-margin 24
--top-jump-interval 4
--top-random-offset 8
--top-jump-seed 123
--subtitle-words 5
--subtitle-scale 140
--subtitle-side-margin 80
--subtitle-y-offset -10
--max-part-duration 60
--no-split
--silence-noise -35dB
--min-silence 0.35
```

`--bottom-audio-volume` is a linear multiplier: `0.05` means 5%, `1.0`
means 100%, and `2.05` means 205%. Values from 0 through 10 are accepted.

`--swap-main-bottom` moves the main-ratio video to the lower part of the
frame and the secondary video to the upper part. The main video keeps its
configured share, duration, speed, exposure, subtitles, and primary audio.
The subtitle junction and left overlay follow the moved boundary.

For very high gain, the default limiter prevents digital clipping. This means
that values such as `800` apply an 8x pre-gain, but the encoded signal cannot
grow beyond the digital ceiling:

```bash
# Clean peak limiting, default.
--main-audio-volume-percent 800 --audio-peak-protection limiter

# Denser saturation instead of a conventional limiter.
--main-audio-volume-percent 800 --audio-peak-protection softclip

# Literal pre-gain with possible severe clipping and distortion.
--main-audio-volume-percent 800 --audio-peak-protection off
```

For a quick layout check without Whisper:

```bash
.venv/bin/python four_video_layout.py \
  02/cat/cat1.m4v 02/cat/cat2.m4v 02/cat/cat2.m4v 02/cat/cat2.m4v \
  --no-subtitles --duration 10 \
  -o results/02/cat/cat_four_video_preview.m4v
```
