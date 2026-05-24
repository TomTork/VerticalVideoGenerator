  python3 align_audio_starts.py 02/cat/cat1.m4v 02/cat/cat2.m4v \
    --output-a 02/cat/cat1_synced.m4v \
    --output-b 02/cat/cat2_synced.m4v \
    --analysis-duration 160 \
    --max-offset 45 \
    --min-overlap 20

  python3 main.py 02/cat/cat1_synced.m4v 02/cat/cat2_synced.m4v \
    -o 02/cat/cat_result.m4v \
    --audio 1 \
    --sub \
    --transcriber whisper \
    --whisper-model base \
    --whisper-language ru \
    --subtitle-lowercase \
    --copy-audio \
    --main-rotate 0 \
    --second-rotate 0 \
    --main-max-span 8 \
    --second-max-span 6 \
    --min-transitions 8 \
    --max-transitions 12 \
    --transition-mode random \
    --transition-styles smoothleft,smoothright,circleopen,fade,wipeleft,wiperight \
    --second-effect vhs-glitch \
    --no-second-denoise \
    --second-zoom 1.0

  python3 normalize_audio.py 02/cat/cat_result.m4v 02/cat/cat_result_normalized.m4v \
    --mode loudnorm \
    --gain-db 0 \
    --target-lufs -16 \
    --true-peak -1.5 \
    --lra 11

  python3 batch_process.py 02 --copy-audio

  # Batch output:
  # results/02/cat/cat_result.m4v
  # results/02/cat/cat_result_part1.m4v
  # results/02/cat/cat_result_part2.m4v

  # batch_process.py auto-detects cat1.*, cat.* and cat2.* video files,
  # including uppercase extensions such as cat.MOV and cat2.MOV.
  # When --main-rotate is 90 or 270, the default output width/height are
  # swapped before rendering, so sideways vertical footage stays vertical.
  # Per-folder overrides can be stored in help.json:
  #
  # {
  #   "inputs": {
  #     "main": "cat.MOV",
  #     "second": "cat2.MOV",
  #     "text": "text.txt"
  #   },
  #   "align": {
  #     "max-offset": 60
  #   },
  #   "main": {
  #     "video2": {
  #       "rotate": 90
  #     },
  #     "audio": {
  #       "copy": true
  #     }
  #   }
  # }
