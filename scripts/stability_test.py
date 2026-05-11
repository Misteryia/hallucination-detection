"""Тест устойчивости LLM-as-judge при температуре по умолчанию (1.0).

Прогоняет 100 случайно выбранных test-примеров (с фиксированным seed=42)
через каждый из двух арбитров (Claude Haiku 4.5 и GPT-5.4-mini)
``N_REPEATS`` раз. По каждому повтору считаются P/R/F1/Q_prod/Accuracy,
плюс % случаев, где все повторы выдали один и тот же вердикт
(fully consistent).

Использует существующие функции из ``run_llm_baseline.py`` и
``run_llm_baseline_openai.py``:

  * ``PROMPT_TEMPLATE_V2`` и ``parse_yn_v2`` — общий промпт и парсер;
  * ``query_anthropic`` — вызов Claude через Anthropic SDK;
  * ``make_openai_client`` + ``query_openai`` — вызов GPT через OpenAI API.

Запуск:

    nohup python -m scripts.stability_test \\
        > reports/stability_test.console.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from halu_detect.data_loader import load_halueval  # noqa: E402

from scripts.run_llm_baseline import (  # noqa: E402
    PROMPT_TEMPLATE_V2,
    parse_yn_v2,
    query_anthropic,
)
from scripts.run_llm_baseline_openai import (  # noqa: E402
    make_openai_client,
    query_openai,
)

QA_DATA = ROOT / "data" / "qa_data.json"
SPLIT_FILE = ROOT / "splits" / "halueval_split.json"
OUTPUT = ROOT / "reports" / "stability_test.json"
LOG = ROOT / "reports" / "stability_test.log"

N_SAMPLES = 100
N_REPEATS = 3
SEED_SAMPLE = 42
SLEEP_BETWEEN = 0.2          # сек, чтобы не упереться в rate-limit
MAX_PARSE_FAIL_PCT = 5.0     # %, выше — аварийная остановка

HAIKU_MODEL = "claude-haiku-4-5"
GPT_MODEL = "gpt-5.4-mini"


# ─────────────────────────────────────────────────────────────────────────
# Загрузка
# ─────────────────────────────────────────────────────────────────────────


def load_test_subset() -> List:
    """Случайные 100 sample_id из test-сплита (тот же seed=42)."""
    samples = load_halueval(QA_DATA)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    test_ids = sorted(set(split["test"]))
    rng = random.Random(SEED_SAMPLE)
    sampled_ids = sorted(rng.sample(test_ids, N_SAMPLES))
    return [(sid, samples[sid]) for sid in sampled_ids]


# ─────────────────────────────────────────────────────────────────────────
# Запуск одного арбитра на N_SAMPLES
# ─────────────────────────────────────────────────────────────────────────


def run_arbiter(
    samples: List,
    model_name: str,
    repeat_idx: int,
    log_f,
    openai_client=None,
) -> List[Dict]:
    """Прогнать арбитр на samples, вернуть список результатов.

    Каждый sample даёт ДВА результата (right_answer и hallucinated_answer),
    итого 200 запросов на повтор.
    """
    is_haiku = "haiku" in model_name.lower() or "claude" in model_name.lower()

    results: List[Dict] = []
    n_done = 0
    n_parse_fail = 0
    t0 = time.time()

    for sid, sample in samples:
        for kind, answer, expected in (
            ("right", sample.right_answer, False),
            ("hallucinated", sample.hallucinated_answer, True),
        ):
            prompt = PROMPT_TEMPLATE_V2.format(
                knowledge=sample.knowledge,
                question=sample.question,
                answer=answer,
            )
            try:
                if is_haiku:
                    raw = query_anthropic(prompt, model_name)
                else:
                    raw = query_openai(openai_client, prompt, model_name)
                predicted = parse_yn_v2(raw)
            except Exception as exc:
                log_f.write(
                    f"  ! API error sid={sid} kind={kind}: {exc}\n"
                )
                log_f.flush()
                raw, predicted = f"ERROR: {exc}", None

            if predicted is None:
                n_parse_fail += 1

            results.append({
                "sample_id": sid,
                "kind": kind,
                "verdict_hallucination": predicted,   # bool | None
                "expected_hallucination": expected,
                "raw": raw[:200] if raw else "",
            })
            n_done += 1
            time.sleep(SLEEP_BETWEEN)

        # Прогресс
        if n_done % 40 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            log_f.write(
                f"  {model_name} repeat {repeat_idx}: "
                f"{n_done}/{len(samples)*2} "
                f"(elapsed {elapsed:.0f}s, {rate:.2f} req/s, "
                f"parse-fail {n_parse_fail})\n"
            )
            log_f.flush()

        # Аварийная остановка если parse-fail > порога
        pct = 100.0 * n_parse_fail / n_done if n_done else 0
        if pct > MAX_PARSE_FAIL_PCT and n_done >= 20:
            log_f.write(
                f"  !!! parse-fail {pct:.1f}% > {MAX_PARSE_FAIL_PCT}%, "
                f"останавливаюсь\n"
            )
            log_f.flush()
            raise SystemExit(
                f"Слишком много parse-fail в {model_name} repeat {repeat_idx}: "
                f"{n_parse_fail}/{n_done} ({pct:.1f}%)"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────


def compute_metrics(results: List[Dict]) -> Dict:
    """P, R, F1, Q_prod, Accuracy. Случаи verdict=None (parse-fail) исключаются."""
    valid = [r for r in results if r["verdict_hallucination"] is not None]
    tp = sum(1 for r in valid
             if r["verdict_hallucination"] and r["expected_hallucination"])
    fp = sum(1 for r in valid
             if r["verdict_hallucination"] and not r["expected_hallucination"])
    fn = sum(1 for r in valid
             if not r["verdict_hallucination"] and r["expected_hallucination"])
    tn = sum(1 for r in valid
             if not r["verdict_hallucination"] and not r["expected_hallucination"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    q_prod = (precision + recall) / 2.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "q_prod": q_prod, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_valid": len(valid), "n_parse_fail": len(results) - len(valid),
    }


def compute_pairwise_agreement(results_list: List[List[Dict]]) -> Optional[Dict]:
    """% случаев, где ВСЕ ``N_REPEATS`` повторов выдали одинаковый вердикт.

    Случаи с хотя бы одним parse-fail классифицируются как ``unparseable`` —
    они не учитываются в ``fully_consistent``.
    """
    if len(results_list) < 2:
        return None

    by_key = defaultdict(list)
    for repeat_results in results_list:
        for r in repeat_results:
            key = (r["sample_id"], r["kind"])
            by_key[key].append(r["verdict_hallucination"])

    fully_consistent = 0
    flipped = 0
    unparseable = 0
    for verdicts in by_key.values():
        if any(v is None for v in verdicts):
            unparseable += 1
            continue
        if len(set(verdicts)) == 1:
            fully_consistent += 1
        else:
            flipped += 1

    parseable = fully_consistent + flipped
    return {
        "total_cases": len(by_key),
        "fully_consistent": fully_consistent,
        "flipped_at_least_once": flipped,
        "unparseable": unparseable,
        "fully_consistent_pct": (
            100.0 * fully_consistent / parseable if parseable else 0.0
        ),
    }


def metric_ranges(per_repeat: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in ("precision", "recall", "f1", "q_prod", "accuracy"):
        vals = [m[k] for m in per_repeat]
        lo, hi = min(vals), max(vals)
        out[k] = {
            "min": lo, "max": hi, "spread": hi - lo,
            "mean": sum(vals) / len(vals),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    samples = load_test_subset()
    print(f"Загружено {len(samples)} примеров для теста устойчивости")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with LOG.open("w", encoding="utf-8") as log_f:
        log_f.write("Тест устойчивости LLM-as-judge (температура по умолчанию = 1.0)\n")
        log_f.write(f"N_SAMPLES={N_SAMPLES}, N_REPEATS={N_REPEATS}\n")
        log_f.write(f"Seed выборки: {SEED_SAMPLE}\n")
        log_f.write(f"Запросов всего: "
                    f"{N_SAMPLES * 2 * N_REPEATS * 2} "
                    f"({N_SAMPLES} × 2 × {N_REPEATS} × 2 моделей)\n")
        log_f.write(f"Старт: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_f.flush()

        all_results: Dict = {}

        # Один openai-клиент на все повторы GPT.
        openai_client = make_openai_client()

        for model_name in (HAIKU_MODEL, GPT_MODEL):
            log_f.write(f"=== {model_name} ===\n")
            log_f.flush()

            repeats: List[List[Dict]] = []
            metrics_per_repeat: List[Dict] = []
            t_model = time.time()

            for r in range(N_REPEATS):
                log_f.write(f"Повтор {r + 1}/{N_REPEATS}\n")
                log_f.flush()
                t0 = time.time()
                results = run_arbiter(
                    samples, model_name, r + 1, log_f,
                    openai_client=openai_client,
                )
                elapsed = time.time() - t0
                metrics = compute_metrics(results)
                metrics_per_repeat.append(metrics)
                repeats.append(results)
                log_f.write(
                    f"  P={metrics['precision']:.4f} "
                    f"R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f} "
                    f"Q={metrics['q_prod']:.4f} "
                    f"Acc={metrics['accuracy']:.4f} "
                    f"(elapsed {elapsed:.0f}s, n_valid={metrics['n_valid']}, "
                    f"parse_fail={metrics['n_parse_fail']})\n"
                )
                log_f.flush()

            agreement = compute_pairwise_agreement(repeats)
            ranges = metric_ranges(metrics_per_repeat)

            all_results[model_name] = {
                "metrics_per_repeat": metrics_per_repeat,
                "agreement": agreement,
                "metric_ranges": ranges,
                "repeats_raw": repeats,
            }

            log_f.write(f"\n{model_name} итог (за {time.time() - t_model:.0f}s):\n")
            log_f.write(
                f"  Полное согласие между повторами: "
                f"{agreement['fully_consistent_pct']:.1f}% "
                f"({agreement['fully_consistent']}/"
                f"{agreement['fully_consistent'] + agreement['flipped_at_least_once']} "
                f"парсимых случаев); "
                f"flipped={agreement['flipped_at_least_once']}, "
                f"unparseable={agreement['unparseable']}\n"
            )
            for k in ("precision", "recall", "q_prod"):
                rng = ranges[k]
                log_f.write(
                    f"  {k:<10s}: range [{rng['min']:.4f}, {rng['max']:.4f}], "
                    f"spread={rng['spread']:.4f}, mean={rng['mean']:.4f}\n"
                )
            log_f.write("\n")
            log_f.flush()

        # Сводка
        log_f.write("=" * 70 + "\n")
        log_f.write("ИТОГОВАЯ ТАБЛИЦА СТАБИЛЬНОСТИ\n")
        log_f.write("=" * 70 + "\n")
        log_f.write(
            f"{'модель':<24s} {'Q range':<22s} {'spread':<8s} "
            f"{'fully consistent':<20s}\n"
        )
        log_f.write("-" * 70 + "\n")
        for model_name in (HAIKU_MODEL, GPT_MODEL):
            r = all_results[model_name]
            qr = r["metric_ranges"]["q_prod"]
            ag = r["agreement"]
            log_f.write(
                f"{model_name:<24s} "
                f"[{qr['min']:.4f}, {qr['max']:.4f}]   "
                f"{qr['spread']:.4f}   "
                f"{ag['fully_consistent_pct']:5.1f}%\n"
            )
        log_f.write("\n")

        OUTPUT.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log_f.write(f"Сохранено: {OUTPUT}\n")
        log_f.write(f"Финиш: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nТест завершён. Лог: {LOG}, JSON: {OUTPUT}")


if __name__ == "__main__":
    main()
