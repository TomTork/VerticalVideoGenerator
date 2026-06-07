# Two-camera video editor

Терминальная Python-утилита для автоматического монтажа двух синхронных камер через `ffmpeg`.

Программа принимает видео с основной камеры, видео со второй камеры и опциональный звук. Затем она строит единый таймлайн: начинает с основной камеры, переключается на вторую по паузам или таймеру, улучшает вторую камеру, накладывает VHS-эффект, добавляет субтитры при флаге `--sub` и при необходимости делит длинный результат на две примерно равные части.

## Требования

Обязательно:

```bash
brew install ffmpeg
```

Нужно, чтобы в `PATH` были доступны обе команды:

```bash
ffmpeg -version
ffprobe -version
```

Для прожига субтитров `ffmpeg` должен быть собран с фильтром `subtitles` или `drawtext`. PNG/SVG-оверлеи не используются: сначала выбирается ASS через `subtitles`, при отсутствии libass используется текстовый fallback через `drawtext`.

Опционально для автоматического распознавания речи:

```bash
python3 -m pip install -U openai-whisper
```

После этого программа сможет использовать команду `whisper`. Модель Whisper скачивается самим Whisper при первом запуске, если ее еще нет локально.

Опционально для очистки уже готового текста и выделения ключевых слов:

```bash
brew install ollama
ollama pull llama3.2
```

Важно: Ollama используется для текстовой правки готового текста и для попытки выделить ключевые слова. Распознавание речи выполняется через Whisper CLI.

## Быстрый старт

Взять звук из основной камеры:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 1
```

Взять звук из второй камеры:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 2
```

Передать отдельный аудиофайл:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio speech.wav
```

Добавить субтитры из готового текста:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 1 --sub --sub-text-file transcript.txt
```

Автоматически распознать речь через Whisper и прожечь субтитры:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 1 --sub --transcriber whisper --whisper-model small --whisper-language ru
```

Передать готовый SRT:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 1 --sub --sub-file subtitles.srt
```

## Синхронизация двух камер по аудио

Если камеры запущены не одновременно, сначала выровняйте исходники отдельной утилитой:

```bash
python3 align_audio_starts.py 02/cat/cat1.m4v 02/cat/cat2.m4v \
  --output-a 02/cat/cat1_synced.m4v \
  --output-b 02/cat/cat2_synced.m4v \
  --analysis-duration 160 \
  --max-offset 45 \
  --min-overlap 20
```

Скрипт сравнивает аудио двух камер, обрезает только ранний старт и делает оба результата одинаковой длины. Если после выравнивания один исходник закончился раньше, его хвост дополняется черным видео и тишиной.

Для проверки найденного смещения без рендера:

```bash
python3 align_audio_starts.py 02/cat/cat1.m4v 02/cat/cat2.m4v --no-render
```

После этого в основной монтаж передавайте уже синхронизированные файлы:

```bash
python3 main.py 02/cat/cat1_synced.m4v 02/cat/cat2_synced.m4v \
  -o 02/cat/cat_result.m4v \
  --audio 1 \
  --sub \
  --transcriber whisper \
  --whisper-model base \
  --whisper-language ru \
  --main-max-span 8 \
  --second-max-span 6 \
  --transition-mode random \
  --transition-styles smoothleft,smoothright,circleopen,fade,wipeleft,wiperight \
  --second-effect vhs-glitch \
  --second-zoom 1.0
```

Если установленный `ffmpeg` собран без фильтров `subtitles` и `drawtext`, программа сохранит ASS-файл рядом с результатом, например `result.ass`, и выведет предупреждение. Для обязательного прожига включите `--strict-subtitles`, тогда такая сборка `ffmpeg` будет считаться ошибкой.

Если нужно, чтобы программа падала с ошибкой при невозможности прожечь субтитры:

```bash
python3 main.py main.mp4 second.mp4 -o result.mp4 --audio 1 --sub --sub-text-file transcript.txt --strict-subtitles
```

## Логика монтажа

Монтаж всегда стартует с основной камеры.

Переход с основной камеры на вторую происходит:

- если в аудио найдена пауза дольше `2.5` секунд;
- или если основная камера уже показывалась `15` секунд.

Переход со второй камеры на основную происходит:

- если в аудио найдена пауза дольше `1.5` секунд;
- или если вторая камера уже показывалась `10` секунд.

Пороговые значения можно менять:

```bash
python3 main.py main.mp4 second.mp4 \
  --main-pause-threshold 2.5 \
  --main-max-span 15 \
  --second-pause-threshold 1.5 \
  --second-max-span 10
```

Паузы ищутся через `ffmpeg`-фильтр `silencedetect`. Настройка чувствительности:

```bash
python3 main.py main.mp4 second.mp4 --silence-noise -35dB --min-silence 0.35
```

Количество переходов ограничено. По умолчанию минимум `4`, максимум `6`:

```bash
python3 main.py main.mp4 second.mp4 --min-transitions 4 --max-transitions 6
```

Переходы плавные и строятся через `xfade`. Длительность и стили можно менять:

```bash
python3 main.py main.mp4 second.mp4 \
  --transition-duration 0.45 \
  --transition-styles smoothleft,smoothright,circleopen,fade \
  --transition-mode random
```

Чаще всего частота переключений настраивается через `--main-max-span` и `--second-max-span`: чем меньше значения, тем чаще камера принудительно сменится даже без паузы. `--min-transitions` и `--max-transitions` задают итоговый диапазон количества переходов.

## Обработка второй камеры

Для второй камеры в `ffmpeg`-графе применяется отдельная цепочка:

- нормализация размера под основной кадр;
- опциональный легкий `hqdn3d` только при `--second-denoise`;
- `unsharp` для аккуратного восстановления резкости;
- небольшой zoom-in, по умолчанию `1.045`;
- выбранный режим эффекта: `off`, `vhs`, `glitch` или `vhs-glitch`.

Зум и эффект применяются только к видео второй камеры до наложения субтитров, поэтому размер и позиция субтитров от них не меняются.

```bash
python3 main.py main.mp4 second.mp4 --second-zoom 1.08 --second-effect vhs-glitch --no-second-denoise
```

## Субтитры

Субтитры включаются флагом:

```bash
--sub
```

Источники субтитров:

- `--sub-text "готовый текст"` - текст прямо в команде;
- `--sub-text-file transcript.txt` - готовый текст из файла;
- `--sub-file subtitles.srt` - готовый SRT;
- `--sub-file subtitles.ass` - готовый ASS, будет прожжен как есть;
- `--transcriber whisper` - автоматическое распознавание через Whisper CLI.

Чтобы вывести весь текст субтитров в нижнем регистре:

```bash
--subtitle-lowercase
```

Если передан обычный текст, режим `--subtitle-timing auto` сначала пытается получить word-level таймкоды через Whisper и привязать готовый текст к этим словам. Если Whisper недоступен, используется раскладка по найденным речевым участкам. Если передан SRT, берутся готовые таймкоды.

Сгенерированные ASS-субтитры рисуются одной текстовой репликой внизу кадра на весь рассчитанный интервал. По умолчанию каждая реплика дополнительно удерживается до `1.6` секунды, но не налезает на следующую.

Ключевые слова в сгенерированных субтитрах выделяются курсивом и иногда жирным. По умолчанию режим `auto`: если передан `--ollama-model`, программа пробует Ollama, иначе использует локальную эвристику.

```bash
python3 main.py main.mp4 second.mp4 \
  --audio 1 \
  --sub \
  --sub-text-file transcript.txt \
  --keyword-highlights auto \
  --ollama-model llama3.2
```

Можно принудительно выбрать режим:

```bash
--keyword-highlights ollama
--keyword-highlights heuristic
--keyword-highlights off
```

Очистить готовый текст через локальную Ollama-модель:

```bash
python3 main.py main.mp4 second.mp4 \
  --audio 1 \
  --sub \
  --sub-text-file transcript.txt \
  --ollama-model llama3.2 \
  --cleanup-subtext-with-ollama
```

Если модель еще не скачана:

```bash
python3 main.py main.mp4 second.mp4 \
  --audio 1 \
  --sub \
  --sub-text-file transcript.txt \
  --ollama-model llama3.2 \
  --cleanup-subtext-with-ollama \
  --ollama-pull
```

## Длина результата и разбиение

По умолчанию программа монтирует общий доступный участок исходников: минимальную длительность основной камеры, второй камеры и внешнего аудио, если оно передано отдельным файлом.

Если итоговая длительность больше `60` секунд, программа сохраняет полный файл и дополнительно создает две части:

- `result_part1.mp4`;
- `result_part2.mp4`.

Точка разреза выбирается около середины, но с приоритетом на самую выраженную паузу в аудио в центральной области ролика.

Отключить разбиение:

```bash
python3 main.py main.mp4 second.mp4 --no-split
```

Изменить порог разбиения:

```bash
python3 main.py main.mp4 second.mp4 --max-part-duration 90
```

Ограничить обрабатываемый участок вручную:

```bash
python3 main.py main.mp4 second.mp4 --duration 60
```

## Основные флаги

```text
main_video second_video          входные видео, можно передать позиционно
--main PATH                      путь к основной камере вместо позиционного аргумента
--second PATH                    путь ко второй камере вместо позиционного аргумента
-o, --output PATH                итоговый файл, по умолчанию result.mp4
--audio VALUE                    аудио: путь к файлу, 1/main или 2/second
--audio-from 1|2|main|second     явный выбор аудио из камеры
--sub                            включить субтитры
--sub-text TEXT                  готовый текст
--sub-text-file PATH             текстовый файл или SRT
--sub-file PATH                  готовый SRT или ASS
--strict-subtitles               ошибка, если ffmpeg не умеет прожечь субтитры
--keyword-highlights MODE        auto, ollama, heuristic или off
--subtitle-timing MODE           auto, whisper, speech или even для готового текста
--subtitle-words N               целевое число слов в реплике, по умолчанию 6
--subtitle-min-duration SEC      минимум показа реплики, по умолчанию 2.8
--subtitle-max-duration SEC      максимум показа реплики, по умолчанию 8.5
--subtitle-hold-extension SEC    продление реплики после речи, по умолчанию 1.6
--subtitle-gap SEC               минимальный зазор между репликами, по умолчанию 0.08
--subtitle-lowercase             вывести весь текст субтитров в нижнем регистре
--transcriber auto|whisper|none  распознавание речи при отсутствии готового текста
--whisper-model MODEL            модель Whisper, по умолчанию base
--whisper-language LANG          язык Whisper, например ru
--ollama-model MODEL             локальная модель для очистки текста и ключевых слов
--cleanup-subtext-with-ollama    разрешить Ollama менять готовый текст перед таймингом
--main-pause-threshold SEC       пауза для ухода с основной камеры
--main-max-span SEC              максимальное время на основной камере
--second-pause-threshold SEC     пауза для возврата со второй камеры
--second-max-span SEC            максимальное время на второй камере
--min-transitions N              минимум переходов, по умолчанию 4
--max-transitions N              максимум переходов, по умолчанию 6
--transition-duration SEC        длительность плавного перехода
--transition-styles LIST         список xfade-стилей через запятую
--transition-mode cycle|random   порядок выбора переходов
--transition-seed N              seed для повторяемых случайных переходов
--silence-noise LEVEL            уровень тишины для silencedetect
--min-silence SEC                минимальная пауза для анализа
--width, --height, --fps         параметры выходного видео, по умолчанию 1080x1920
--main-rotate DEG                поворот основной камеры до масштабирования
--second-rotate DEG              поворот второй камеры до масштабирования
--copy-audio                     скопировать выбранное аудио без фильтров и перекодирования
--audio-gain-db DB               простое изменение громкости в dB
--crf, --preset                  параметры libx264
--second-zoom VALUE              приближение второй камеры
--second-effect MODE             off, vhs, glitch или vhs-glitch
--second-denoise                 включить легкий шумодав второй камеры
--max-part-duration SEC          порог разбиения длинного результата
--no-split                       не делить длинный результат
--duration SEC                   обработать только заданную длительность
--keep-temp                      сохранить временные WAV/ASS/SRT файлы
--dry-run                        показать команды ffmpeg без запуска
--self-test                      запустить внутреннюю проверку без ffmpeg
```

## Проверка установки

Встроенная проверка не требует реального видео и не запускает `ffmpeg`:

```bash
python3 main.py --self-test
```

Она проверяет построение таймлайна, генерацию таймингов субтитров, создание ASS-файла и выбор точки разбиения.

## Замечания

Исходники с двух камер должны быть синхронизированы по началу. Программа переключает камеры по одинаковым временным меткам, поэтому если одна камера стартовала позже, сначала выровняйте файлы через `align_audio_starts.py`.

Если в видео нет аудио и отдельный аудиофайл не передан, программа смонтирует переключения только по таймерам: 15 секунд на основной камере и 10 секунд на второй.

Если `--sub` включен без готового текста, нужен установленный `whisper`. Если `whisper` не найден, передайте `--sub-text`, `--sub-text-file` или `--sub-file`.

## Компоновка четырех видео

`four_video_layout.py` собирает вертикальный кадр 1080x1920 из четырех видео:

- основное видео занимает верхние 60%, ускоряется до 1.2x и получает небольшое повышение экспозиции;
- нижнее видео занимает оставшиеся 40%;
- левое видео масштабируется до 90x90 и частично выходит за левый край;
- видео 64x64 каждые 5 секунд переключается между верхними углами.

Размеры, пропорции, смещения, скорость, экспозиция и параметры прыжков
настраиваются аргументами. Все входы растягиваются до целевой области без
кропа. Основной звук используется с уровнем 100%, звук нижнего видео
подмешивается с уровнем 5%, звук остальных источников игнорируется.

Цветные ASS-субтитры располагаются на стыке областей. Цвет меняется с каждой
строкой, вокруг букв используется черная обводка без прямоугольной подложки,
а произносимое слово увеличивается и возвращается к исходному размеру по
word timestamps Whisper.

Полная команда и параметры приведены в `EXECUTION-2.md`.
