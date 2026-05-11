# Скрипты экспериментов

Все скрипты запускаются из корня репозитория через `python -m scripts.<name>`. Папка `scripts/` содержит `__init__.py`, поэтому импортируется как пакет.

## Воспроизведение главных результатов

Скрипты ниже воспроизводят таблицу 3 (метрики методов на test) и анализ валидности разметки (глава 6 диплома).

| # | Скрипт | Назначение |
|---|--------|------------|
| 1 | `prepare_evaluation.py` | Sanity-check датасета + детерминированный group-aware split HaluEval-QA: 50 + 7959 + 1991 (seed=42) |
| 2a | `tune_thresholds.py` | Базовый подбор порогов: scoring + grid search по 3 стратегиям (q_prod, f1, q_prod_constrained); кеш скоров `reports/train_scores_n1000.json` |
| 2b | `fine_grid_thresholds.py` | Уточняющий поиск по плотной сетке на готовом кеше скоров |
| 3 | `run_test_evaluation.py` | Прогон классических методов на test (1991 пример) |
| 4 | `run_llm_baseline.py` | LLM-as-judge Claude Haiku 4.5 на test |
| 5 | `run_llm_baseline_openai.py` | LLM-as-judge GPT-5.4-mini на test |

Время полного прогона: ~7 часов (NLI+embedding 2:43 на CPU, Haiku 1:45, GPT 2:10).

## Статистический анализ

| # | Скрипт | Назначение |
|---|--------|------------|
| 6 | `bootstrap_main_paired.py` | Paired bootstrap CI на уровне примеров |
| 7 | `mcnemar_tests.py` | Тест Макнемара (Haiku vs GPT, NLI vs LLM) |
| 7a | `fisher_test.py` | Точный тест Фишера для независимости расхождений двух арбитров (Утверждение 3, подраздел 6.2 ВКР) |
| 8 | `kappa_gpt.py` | Cohen's κ для двух классификаторов |
| 9 | `q_prod_sensitivity.py` | Устойчивость порядка методов к весу λ в Q_λ |

## Анализ валидности разметки (глава 6)

| # | Скрипт | Назначение |
|---|--------|------------|
| 10 | `auto_classify_disagreements.py` | Автоклассификация расхождений Haiku-classifier |
| 11 | `auto_classify_disagreements_openai.py` | Автоклассификация расхождений GPT-classifier |
| 12 | `compare_arbiters.py` | Пересечение и анализ Jaccard двух арбитров |
| 13 | `prepare_manual_review_v3.py` | Подготовка 200 случаев для ручного аудита |
| 14 | `analyze_manual_review.py` | Сводка ручного аудита + Wilson CI (требует `reports/manual_review_haiku_100.md` и `reports/manual_review_gpt_100.md` — см. ниже) |

### Файлы ручного аудита

`scripts/analyze_manual_review.py` ожидает два markdown-файла с разметкой автора:

- `reports/manual_review_haiku_100.md` — 100 случаев расхождений Haiku-арбитра, размеченных вручную в категории `BAD_LABEL` / `EVASIVE` / `MODEL_MISS`;
- `reports/manual_review_gpt_100.md` — 100 случаев расхождений GPT-арбитра в том же формате.

Эти файлы **не входят в репозиторий**: они представляют собой первичные данные ручного экспертного разбора, выполненного автором ВКР, и хранятся отдельно. Готовая агрегированная сводка по ним (`reports/manual_review_analysis.json`, на которой основана глава 6 ВКР) в репозитории закоммичена. Запросить исходные .md-файлы можно у автора работы.

Подвыборка из 200 расхождений (по 100 на каждого арбитра), которые подлежат разметке, детерминированно выбирается из полной выборки расхождений скриптом `prepare_manual_review_v3.py` (seed=42).

## Дополнительно

| # | Скрипт | Назначение |
|---|--------|------------|
| 15 | `stability_test.py` | Устойчивость LLM-вердиктов при temp=1.0 (3 повтора) |

## Требования

- Python 3.10+, виртуальное окружение `venv`
- `.env` с `ANTHROPIC_API_KEY` и/или `OPENAI_API_KEY` (для скриптов 4-5, 10-11)
- Датасет HaluEval-QA в `data/qa_data.json` — скачивается из [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) (MIT), см. раздел «Получение датасета» в корневом `README.md`. Прямой URL файла: <https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa/qa_data.json>.

Файл `splits/halueval_split.json` (детерминированное разбиение seed=42, group-aware по `knowledge`) уже закоммичен в репозиторий — его пересборка через `prepare_evaluation.py` нужна только для проверки воспроизводимости.
