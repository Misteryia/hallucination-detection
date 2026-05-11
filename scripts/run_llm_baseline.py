"""
LLM-as-judge baseline на test-сплите HaluEval.

Берёт те же sample_id, которые выбрал ``scripts/run_test_evaluation`` (та
же rng-логика и тот же seed), и для каждого ``right_answer`` /
``hallucinated_answer`` спрашивает claude-haiku-4-5: «является ли это
галлюцинацией с учётом knowledge?». Полученные YES/NO превращаются в
:class:`EvaluationResult` и пропускаются через :class:`Aggregator` —
метрики считаются той же функцией, что и для ``FactChecker``, чтобы
цифры были напрямую сравнимы.

В отчёт ``reports/llm_baseline.json`` дополнительно подмешивается блок
``comparison_with_classical`` — лучшие конфигурации классических методов
по test (из ``reports/test_evaluation.json``) рядом с LLM-baseline.

Запуск:

    python -m scripts.run_llm_baseline --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from tqdm import tqdm  # noqa: E402

from halu_detect.aggregator import Aggregator, EvaluationResult  # noqa: E402
from halu_detect.data_loader import load_halueval  # noqa: E402
from scripts.tune_thresholds import _hash_ids  # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Промт и парсеры
# ─────────────────────────────────────────────────────────────────────────


PROMPT_TEMPLATE_V2 = """Ты эксперт по проверке фактов.
Твоя задача — определить, содержит ли ответ галлюцинацию с учётом приведённого контекста.

Контекст (knowledge):
{knowledge}

Вопрос:
{question}

Ответ для проверки:
{answer}

Содержит ли этот ответ информацию, которая ПРОТИВОРЕЧИТ контексту или
не подтверждается им? Ответь СТРОГО в формате:

VERDICT: YES (если галлюцинация) или NO (если корректно)
EXPLANATION: одно короткое предложение, почему."""


def parse_yn_v2(text: str) -> Optional[bool]:
    """Извлечь YES/NO из VERDICT-строки. Fallback — первое слово ответа."""
    m = re.search(r"VERDICT\s*[:\-]\s*(YES|NO)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "YES"
    stripped = text.strip()
    if not stripped:
        return None
    first = stripped.split()[0].upper()
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    return None


def extract_explanation(text: str) -> str:
    """Извлечь объяснение после строки EXPLANATION."""
    m = re.search(
        r"EXPLANATION\s*[:\-]\s*(.+?)(?:\n\s*\n|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    # Fallback: всё после первой строки.
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return " ".join(lines[1:]).strip() if len(lines) > 1 else ""


# ─────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────


def query_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────────────────
# Печать
# ─────────────────────────────────────────────────────────────────────────


_METHOD_LABELS = {
    "nli": "NLI",
    "embedding": "Embedding",
    "combined_or": "Combined-OR",
    "combined_and": "Combined-AND",
}


def _format_thresholds(method: str, thr: Dict) -> str:
    nli = thr.get("nli_thr")
    emb = thr.get("emb_thr")
    if method == "nli":
        return f"thr={nli:.2f}"
    if method == "embedding":
        return f"emb={emb:.2f}"
    return f"{nli:.2f}/{emb:.2f}"


def _is_degenerate(test_metrics: Dict) -> bool:
    """Точка считается вырожденной, если TN ≤ 1 (классификатор флажит почти всё)."""
    return test_metrics.get("tn", 0) <= 1


def render_comparison_table(
    classical: Dict[str, Dict],
    llm_metrics: Dict,
    n_test: int,
    elapsed_sec: float,
) -> str:
    out = []
    out.append("=" * 100)
    out.append("=== СРАВНЕНИЕ: КЛАССИЧЕСКИЕ МЕТОДЫ vs LLM-as-JUDGE ===")
    out.append("=" * 100)
    out.append(
        f"{'method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Q_prod':>7}  {'note':<20}"
    )
    out.append("-" * 100)
    for method in ("nli", "embedding", "combined_or", "combined_and"):
        e = classical.get(method)
        if not e:
            continue
        ts = e["test"]
        thr_str = _format_thresholds(method, e["thresholds"])
        label = f"{_METHOD_LABELS[method]} ({thr_str})"
        note = "⚠ TN≤1" if _is_degenerate(ts) else "free"
        out.append(
            f"{label:<35} {ts['precision']:>7.3f} {ts['recall']:>7.3f} "
            f"{ts['f1']:>7.3f} {ts['q_prod']:>7.3f}  {note:<20}"
        )
    out.append(
        f"{'LLM-as-judge (claude-haiku-4-5)':<35} "
        f"{llm_metrics['precision']:>7.3f} {llm_metrics['recall']:>7.3f} "
        f"{llm_metrics['f1']:>7.3f} {llm_metrics['q_prod']:>7.3f}  "
        f"{f'$$, ~{elapsed_sec:.0f}s/{n_test}':<20}"
    )
    out.append("=" * 100)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("reports/llm_baseline.json"))
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--data", type=Path, default=Path("data/qa_data.json"))
    parser.add_argument("--split", type=Path, default=Path("splits/halueval_split.json"))
    parser.add_argument(
        "--test-eval", type=Path, default=Path("reports/test_evaluation.json"),
        help="Файл с метриками классических методов для блока сравнения.",
    )
    parser.add_argument(
        "--test-scores", type=Path, default=Path("reports/test_scores.json"),
        help="Кеш test-скоров; используется для проверки совпадения test_ids_hash.",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Нужно установить ANTHROPIC_API_KEY (через .env в корне проекта)"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else project_root / p

    data_path = _resolve(args.data)
    split_path = _resolve(args.split)
    out_path = _resolve(args.out)
    test_eval_path = _resolve(args.test_eval)
    test_scores_path = _resolve(args.test_scores)

    # ── Сэмплирование test-ID той же rng-логикой, что и run_test_evaluation ──
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = sorted(set(split["test"]))
    if args.max_samples > 0 and args.max_samples < len(test_ids):
        rng = random.Random(args.seed)
        sampled_test_ids = sorted(rng.sample(test_ids, args.max_samples))
    else:
        sampled_test_ids = test_ids

    current_hash = _hash_ids(sampled_test_ids)

    # Сверка с test_scores.json (опциональная, но помогает поймать рассогласование)
    cached_hash = None
    if test_scores_path.exists():
        ts_meta = json.loads(test_scores_path.read_text(encoding="utf-8"))["metadata"]
        cached_hash = ts_meta.get("test_ids_hash")

    if cached_hash and cached_hash == current_hash:
        print(f"✓ test_ids_hash совпадает с {test_scores_path.name}: {current_hash}")
    elif cached_hash:
        print(
            f"⚠ test_ids_hash отличается! "
            f"current={current_hash}, в {test_scores_path.name}={cached_hash}. "
            f"Прогон НЕ будет напрямую сравним с run_test_evaluation."
        )
    else:
        print(f"i test_ids_hash={current_hash} (test_scores.json не найден).")

    # ── Опрос LLM ───────────────────────────────────────────────────────
    samples = load_halueval(data_path)
    print(f"\nLLM-baseline: {len(sampled_test_ids)} примеров → "
          f"{len(sampled_test_ids)*2} запросов через {args.model}\n")

    # ── Чекпоинт-загрузка ────────────────────────────────────────────────
    details: List[Dict] = []
    elapsed_prev = 0.0
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev_meta = prev.get("metadata", {})
            if (prev_meta.get("test_ids_hash") == current_hash
                    and prev_meta.get("model") == args.model):
                details = prev.get("details", [])
                elapsed_prev = float(prev_meta.get("elapsed_sec", 0.0))
                print(
                    f"✓ Чекпоинт найден: {len(details)} запросов уже выполнено "
                    f"(elapsed_prev={elapsed_prev:.0f}s). Продолжаю."
                )
            else:
                print(f"i {out_path.name} есть, но hash/model отличаются — начинаю заново.")
        except Exception as exc:
            logger.warning("Не удалось прочитать чекпоинт %s: %s — начинаю заново.",
                           out_path, exc)
            details = []

    done_pairs = {(d["sample_id"], d["kind"]) for d in details}
    aggregator = Aggregator()
    for d in details:
        if d.get("predicted") is not None:
            aggregator.add_result(EvaluationResult(
                sample_id=d["sample_id"],
                expected_hallucination=d["expected"],
                predicted_hallucination=d["predicted"],
                confidence=1.0,
                method="llm_as_judge",
                claim=d["answer"],
                matched_fact=None,
            ))

    def _save_partial(extra_elapsed: float, completed: bool) -> None:
        """Атомарный дамп текущего состояния details и метрик."""
        m = aggregator.compute_metrics()
        partial_metrics = {
            "tp": m.true_positive, "fp": m.false_positive,
            "tn": m.true_negative, "fn": m.false_negative,
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "accuracy": m.accuracy, "q_prod": m.q_prod,
            "passes_threshold": m.passes_quality_threshold,
            "n_parsed": len(aggregator.results),
            "n_inconsistent": sum(1 for d in details if d.get("inconsistent")),
        }
        partial = {
            "metadata": {
                "model": args.model,
                "n_test_samples": len(sampled_test_ids),
                "n_claims_planned": len(sampled_test_ids) * 2,
                "n_claims_done": len(details),
                "seed": args.seed,
                "test_ids_hash": current_hash,
                "test_ids_hash_matches_run_test_evaluation":
                    bool(cached_hash) and cached_hash == current_hash,
                "elapsed_sec": elapsed_prev + extra_elapsed,
                "completed": completed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "metrics": partial_metrics,
            "details": details,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(partial, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(out_path)

    SAVE_EVERY_EXAMPLES = 100  # = 200 запросов между дампами

    t0 = time.time()
    examples_processed_this_run = 0
    for sid in tqdm(sampled_test_ids, desc="Querying LLM"):
        sample = samples[sid]
        any_new_for_this_sample = False
        for kind, answer, expected in (
            ("right", sample.right_answer, False),
            ("hallucinated", sample.hallucinated_answer, True),
        ):
            if (sid, kind) in done_pairs:
                continue
            any_new_for_this_sample = True

            prompt = PROMPT_TEMPLATE_V2.format(
                knowledge=sample.knowledge,
                question=sample.question,
                answer=answer,
            )
            try:
                raw = query_anthropic(prompt, args.model)
                predicted = parse_yn_v2(raw)
                explanation = extract_explanation(raw)
            except Exception as exc:
                logger.warning("LLM-ошибка (sample %d, %s): %s", sid, kind, exc)
                raw, predicted, explanation = f"ERROR: {exc}", None, ""

            inconsistent = predicted is None
            agreement = (predicted == expected) if predicted is not None else None

            details.append({
                "sample_id": sid,
                "kind": kind,
                "question": sample.question,
                "answer": answer,
                "expected": expected,
                "predicted": predicted,
                "raw_response": raw,
                "explanation": explanation,
                "agreement": agreement,
                "inconsistent": inconsistent,
            })
            done_pairs.add((sid, kind))

            if predicted is not None:
                aggregator.add_result(EvaluationResult(
                    sample_id=sid,
                    expected_hallucination=expected,
                    predicted_hallucination=predicted,
                    confidence=1.0,
                    method="llm_as_judge",
                    claim=answer,
                    matched_fact=None,
                ))

            time.sleep(args.sleep)

        if any_new_for_this_sample:
            examples_processed_this_run += 1

        if examples_processed_this_run > 0 and \
                examples_processed_this_run % SAVE_EVERY_EXAMPLES == 0:
            _save_partial(time.time() - t0, completed=False)
            logger.info(
                "Чекпоинт: %d/%d examples в этом запуске, всего details=%d.",
                examples_processed_this_run, len(sampled_test_ids), len(details),
            )

    elapsed = elapsed_prev + (time.time() - t0)

    metrics = aggregator.compute_metrics()
    metrics_dict = {
        "tp": metrics.true_positive,
        "fp": metrics.false_positive,
        "tn": metrics.true_negative,
        "fn": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "q_prod": metrics.q_prod,
        "passes_threshold": metrics.passes_quality_threshold,
        "n_parsed": len(aggregator.results),
        "n_inconsistent": sum(1 for d in details if d["inconsistent"]),
    }

    # ── Загрузка классических метрик ─────────────────────────────────────
    classical: Dict[str, Dict] = {}
    if test_eval_path.exists():
        test_eval = json.loads(test_eval_path.read_text(encoding="utf-8"))
        classical = test_eval.get("best_by_test_q_prod", {})
    else:
        logger.warning("Файл %s не найден — блок сравнения будет пуст.", test_eval_path)

    # ── Печать сравнительной таблицы ─────────────────────────────────────
    print()
    print(render_comparison_table(
        classical, metrics_dict, len(sampled_test_ids), elapsed,
    ))

    # ── JSON ─────────────────────────────────────────────────────────────
    payload = {
        "metadata": {
            "model": args.model,
            "n_test_samples": len(sampled_test_ids),
            "n_claims": len(details),
            "seed": args.seed,
            "test_ids_hash": current_hash,
            "test_ids_hash_matches_run_test_evaluation":
                bool(cached_hash) and cached_hash == current_hash,
            "elapsed_sec": elapsed,
            "completed": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "metrics": metrics_dict,
        "details": details,
        "comparison_with_classical": {
            **classical,
            "llm_as_judge": {
                "thresholds": None,
                "test": metrics_dict,
                "model": args.model,
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
