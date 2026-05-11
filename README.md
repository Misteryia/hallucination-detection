# halu-detect

Программная утилита детектирования галлюцинаций в текстах больших языковых моделей. Реализация в рамках выпускной квалификационной работы бакалавра «Исследование математических моделей детектирования галлюцинаций в текстах, генерируемых большими языковыми моделями».

## Описание

Утилита реализует два класса классических методов детектирования:

- **Natural Language Inference (NLI)** — классификация противоречий через предобученную модель DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- **Embedding similarity** — поиск семантически близких фактов через модель all-mpnet-base-v2

Утилита halu-detect (CLI-команда `haludetect`) реализует только классические методы (NLI и embedding similarity). Подход LLM-as-judge с использованием Claude Haiku 4.5 (Anthropic) и GPT-5.4-mini (OpenAI) реализован отдельным экспериментальным скриптом `scripts/run_llm_baseline.py` и `scripts/run_llm_baseline_openai.py` для сопоставительной оценки в рамках исследовательской работы.

## Установка

```bash
git clone git@github.com:Misteryia/halu-detect.git
cd halu-detect
python -m venv venv
source venv/bin/activate

# Базовый набор: CLI + классические методы + статистический анализ (mcnemar, fisher).
pip install -e .

# Для запуска LLM-as-judge скриптов в scripts/ (run_llm_baseline*, auto_classify_*):
pip install -e ".[llm]"

python -m spacy download en_core_web_sm
```

После `pip install -e .` в `PATH` появляется команда `haludetect` (точка входа задана в `pyproject.toml`).

Для запуска экспериментальных скриптов LLM-as-judge в каталоге `scripts/` скопируйте `.env.example` в `.env` и подставьте свои API-ключи Anthropic и/или OpenAI. Для базовых команд `check` и `evaluate --method nli|embedding`, статистических скриптов (`mcnemar_tests.py`, `fisher_test.py`) и тестов API-ключи не требуются.

## Получение датасета

Эксперименты главы 5 ВКР опираются на датасет HaluEval-QA. Сам файл `qa_data.json` не входит в репозиторий — его нужно скачать из открытого репозитория [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) (лицензия MIT) и положить в `data/qa_data.json`:

```bash
mkdir -p data
curl -L -o data/qa_data.json \
    https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa/qa_data.json
```

Метрики глав 5–6 посчитаны на исходной 10 000-строчной версии датасета (формат JSONL, по одному JSON-объекту на строку с полями `knowledge`, `question`, `right_answer`, `hallucinated_answer`). После скачивания становятся работоспособны `haludetect evaluate`, скрипты в `scripts/` и тесты в `tests/`.

## Использование

### Проверка пользовательского текста

```bash
haludetect check \
    --input data/sample_text.json \
    --kb data/sample_kb.json \
    --output reports \
    --method nli
```

Формат входных файлов:

```json
// data/sample_text.json
{"text": "проверяемый текст ..."}

// data/sample_kb.json — допустимы оба варианта
{"facts": ["факт 1", "факт 2", ...]}
["факт 1", "факт 2", ...]
```

Утилита разбивает текст на отдельные утверждения средствами spaCy, для каждого ищет наиболее похожий факт в базе знаний и проверяет наличие противоречия. В каталоге `--output` создаются файлы `<--name>.json` и `<--name>.html` (по умолчанию `check_report.json` и `check_report.html`).

Доступные методы: `--method nli|embedding|combined` (для `combined` режим объединения задаётся `--mode or|and`). Пороги настраиваются флагами `--nli-threshold` и `--similarity-threshold`.

### Оценка на эталонном датасете

```bash
haludetect evaluate \
    --dataset data/qa_data.json \
    --limit 200 \
    --method nli \
    --output reports \
    --name evaluation_report
```

Прогоняет указанный метод по подмножеству HaluEval-QA (`--limit` задаёт число примеров, по умолчанию 100), считает метрики Precision/Recall/F1/Accuracy/Q_prod и записывает отчёты в `<--output>/<--name>.json` и `<--output>/<--name>.html` с цветовой индикацией соответствия целевым показателям и топ-10 ложноположительных/ложноотрицательных классификаций.

## Структура проекта

```
halu_detect/          — исходный код утилиты
  data_loader.py      — загрузка датасета HaluEval (load_halueval, load_custom)
  claim_extractor.py  — разбиение текста на утверждения через spaCy
  fact_checker.py     — верификация через NLI/embedding (FactChecker)
  aggregator.py       — агрегация вердиктов и подсчёт метрик
  report_generator.py — генерация отчётов JSON/HTML
  evaluation.py       — связующий модуль evaluate_on_halueval
  utils.py            — split_to_facts и прочие утилиты
  cli.py              — интерфейс командной строки
scripts/              — скрипты для экспериментов (LLM-as-judge, подбор порогов, bootstrap, McNemar и др.)
tests/                — модульные и smoke-тесты
splits/               — детерминированные разбиения данных
reports/              — JSON-отчёты экспериментов
```

## Воспроизводимость

Все эксперименты используют фиксированный seed = 42. Разбиение датасета HaluEval-QA сохранено в `splits/halueval_split.json` (50 + 7959 + 1991 = 10000 примеров, group-aware split по полю knowledge); файл закоммичен и пересоздаётся побитово через `python -m scripts.prepare_evaluation` (после того как датасет скачан в `data/qa_data.json`, см. раздел «Получение датасета»). Результаты ключевых экспериментов сохранены в `reports/*.json`.

## Лицензия

MIT License (см. LICENSE).
