"""
Финальный прогон halu_detect на test-сплите HaluEval.

Скрипт сравнивает три стратегии подбора порогов
(``q_prod``, ``f1``, ``q_prod_constrained``), полученные на train
через ``scripts/tune_thresholds.py``. Для каждой комбинации
(метод × стратегия) считаются метрики на test и сравниваются с
train-метриками — это показывает train/test gap (``ΔQ``) и помогает
отличить устойчивые конфигурации от переобученных под train.

Архитектура повторяет двухфазный подход ``tune_thresholds``:

  1. **Scoring (медленно, кешируется в reports/test_scores.json).**
     Для каждого test-claim считаются сырые ``nli_score``/``emb_score``.
  2. **Применение порогов (мгновенно).** 12 конфигураций
     (4 метода × 3 стратегии) применяются к одним и тем же сырым
     скорам.

Запуск:

    python -m scripts.run_test_evaluation --max-samples 200 \\
        2>&1 | tee reports/test_eval.log
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from halu_detect.data_loader import load_halueval
from halu_detect.fact_checker import FactChecker

# Переиспользуем логику скоринга и метрик 1-в-1 — это гарантирует, что
# train- и test-метрики посчитаны одной и той же функцией.
from scripts.tune_thresholds import (
    ScoredClaim,
    _hash_ids,
    compute_train_scores,
    metrics_from_pairs,
    predict,
)

logger = logging.getLogger(__name__)

METHODS = ("nli", "embedding", "combined_or", "combined_and")
STRATEGIES = ("q_prod", "f1", "q_prod_constrained")

TARGET_PRECISION = 0.80
TARGET_RECALL = 0.75
TARGET_Q_PROD = 0.5 * TARGET_PRECISION + 0.5 * TARGET_RECALL  # = 0.775


# ─────────────────────────────────────────────────────────────────────────
# Кеш test-скоров
# ─────────────────────────────────────────────────────────────────────────


def save_test_scores(
    scores: List[ScoredClaim],
    path: Path,
    test_ids: List[int],
    completed: bool = True,
) -> None:
    """Сохранить скоры. ``completed`` помечает, доскорил ли весь target-сет.

    Промежуточные дампы (``completed=False``) пишутся каждые ``save_every``
    примеров — это позволяет продолжить с места обрыва при рестарте.
    Атомарность: пишем во временный файл и переименовываем, чтобы не
    остаться с обрезанным JSON, если процесс убили в момент записи.
    """
    payload = {
        "metadata": {
            "split": "test",
            "n_claims": len(scores),
            "test_ids_hash": _hash_ids(test_ids),
            "n_test_samples": len(test_ids),
            "completed": bool(completed),
        },
        "scores": [s.to_dict() for s in scores],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_test_scores(
    path: Path, test_ids: List[int], allow_partial: bool = False
) -> Optional[List[ScoredClaim]]:
    """Загрузить кеш скоров.

    При ``allow_partial=False`` (по умолчанию) — поведение как раньше:
    возвращаем список только если хеш совпадает И ``completed=True``.
    При ``allow_partial=True`` — возвращаем то, что есть, при совпадении
    хеша; вызывающая сторона сама определит, какие ID ещё надо обсчитать.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cached = payload.get("metadata", {}).get("test_ids_hash")
    expected = _hash_ids(test_ids)
    if cached != expected:
        logger.warning(
            "Кеш test-скоров не соответствует подвыборке (cached=%s, expected=%s); "
            "пересчитываем.", cached, expected,
        )
        return None
    completed = bool(payload.get("metadata", {}).get("completed", True))
    if not completed and not allow_partial:
        return None
    return [ScoredClaim.from_dict(d) for d in payload["scores"]]


def compute_test_scores_checkpointed(
    samples,
    test_ids: List[int],
    fact_checker,
    cache_path: Path,
    save_every: int = 100,
) -> List[ScoredClaim]:
    """Посчитать сырые скоры с периодическим чекпоинтингом.

    На вход — полный список ``test_ids``. Если в ``cache_path`` уже
    лежит частичный (или полный) кеш с тем же хешем, дочислит только
    недостающие. Каждые ``save_every`` обработанных examples делает
    атомарный дамп, чтобы рестарт после kill-а возобновлял работу.
    """
    from halu_detect.utils import split_to_facts

    full_hash = _hash_ids(test_ids)

    out: List[ScoredClaim] = []
    completed_ids: set = set()
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("metadata", {}).get("test_ids_hash") == full_hash:
                out = [ScoredClaim.from_dict(d) for d in payload.get("scores", [])]
                completed_ids = {s.sample_id for s in out}
                logger.info(
                    "Найден частичный кеш: %d/%d sample_id уже обсчитано.",
                    len(completed_ids), len(test_ids),
                )
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s; начинаем с нуля.",
                           cache_path, exc)
            out, completed_ids = [], set()

    remaining = [sid for sid in test_ids if sid not in completed_ids]
    if not remaining:
        save_test_scores(out, cache_path, test_ids, completed=True)
        return out

    for i, sid in enumerate(tqdm(remaining, desc="Scoring test")):
        sample = samples[sid]
        facts = split_to_facts(sample.knowledge)
        for kind, claim, expected in (
            ("right", sample.right_answer, False),
            ("hallucinated", sample.hallucinated_answer, True),
        ):
            if not facts:
                out.append(ScoredClaim(
                    sample_id=sid, kind=kind, expected_hallucination=expected,
                    n_facts=0, nli_score=0.0, emb_score=0.0,
                    nli_matched_fact="", emb_matched_fact="",
                ))
                continue
            nli_s, nli_f = fact_checker.score_nli(claim, facts)
            emb_s, emb_f = fact_checker.score_embedding(claim, facts)
            out.append(ScoredClaim(
                sample_id=sid, kind=kind, expected_hallucination=expected,
                n_facts=len(facts),
                nli_score=float(nli_s), emb_score=float(emb_s),
                nli_matched_fact=nli_f, emb_matched_fact=emb_f,
            ))

        if (i + 1) % save_every == 0:
            save_test_scores(out, cache_path, test_ids, completed=False)
            logger.info(
                "Чекпоинт: %d/%d examples, всего claims в кеше: %d.",
                len(completed_ids) + i + 1, len(test_ids), len(out),
            )

    save_test_scores(out, cache_path, test_ids, completed=True)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Применение конфигов
# ─────────────────────────────────────────────────────────────────────────


def apply_config(
    scores: List[ScoredClaim],
    method_key: str,
    nli_thr: Optional[float],
    emb_thr: Optional[float],
) -> Dict:
    """Применить конфиг (method_key, пороги) к скорам и вернуть метрики."""
    if method_key == "nli":
        pairs = predict(scores, "nli", nli_thr, None)
    elif method_key == "embedding":
        pairs = predict(scores, "embedding", None, emb_thr)
    elif method_key == "combined_or":
        pairs = predict(scores, "combined", nli_thr, emb_thr, combined_mode="or")
    elif method_key == "combined_and":
        pairs = predict(scores, "combined", nli_thr, emb_thr, combined_mode="and")
    else:
        raise ValueError(f"Неизвестный метод: {method_key!r}")
    return metrics_from_pairs(pairs)


# ─────────────────────────────────────────────────────────────────────────
# Рендер таблиц
# ─────────────────────────────────────────────────────────────────────────


def render_method_comparison(method_key: str, rows: Dict[str, Dict]) -> str:
    """Сравнительная таблица train vs test для одного метода."""
    out = []
    out.append("=" * 110)
    out.append(f"=== МЕТОД: {method_key} ===")
    out.append("=" * 110)
    out.append(
        f"{'strategy':<22} {'nli_thr':>8} {'emb_thr':>8} "
        f"{'TR P':>6} {'TR R':>6} {'TR Q':>6} "
        f"{'TS P':>6} {'TS R':>6} {'TS Q':>6} {'ΔQ':>7}"
    )
    out.append("-" * 110)
    for strategy in STRATEGIES:
        e = rows.get(strategy)
        if not e:
            continue
        thr = e["thresholds"]
        tr = e["train"]
        ts = e["test"]
        nli = f"{thr['nli_thr']:.2f}" if thr["nli_thr"] is not None else "—"
        emb = f"{thr['emb_thr']:.2f}" if thr["emb_thr"] is not None else "—"
        delta = e["delta_q"]
        out.append(
            f"{strategy:<22} {nli:>8} {emb:>8} "
            f"{tr['precision']:>6.3f} {tr['recall']:>6.3f} {tr['q_prod']:>6.3f} "
            f"{ts['precision']:>6.3f} {ts['recall']:>6.3f} {ts['q_prod']:>6.3f} "
            f"{delta:>+7.3f}"
        )
    out.append("=" * 110)
    return "\n".join(out)


def render_best_by_test(best_by_test: Dict[str, Dict]) -> str:
    out = []
    out.append("=" * 100)
    out.append("=== ЛУЧШИЕ КОМБИНАЦИИ ПО TEST Q_prod ===")
    out.append("=" * 100)
    out.append(
        f"{'method':<14} {'strategy':<22} "
        f"{'TEST P':>7} {'TEST R':>7} {'TEST Q':>7} {'TEST F1':>8} {'passes?':>8}"
    )
    out.append("-" * 100)
    for method_key in METHODS:
        e = best_by_test.get(method_key)
        if not e:
            continue
        ts = e["test"]
        ok = "✓" if ts["passes_threshold"] else "✗"
        out.append(
            f"{method_key:<14} {e['strategy']:<22} "
            f"{ts['precision']:>7.3f} {ts['recall']:>7.3f} {ts['q_prod']:>7.3f} "
            f"{ts['f1']:>8.3f} {ok:>8}"
        )
    out.append("=" * 100)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/qa_data.json"))
    parser.add_argument("--split", type=Path, default=Path("splits/halueval_split.json"))
    parser.add_argument(
        "--thresholds", type=Path,
        default=Path("reports/thresholds_comparison.json"),
        help="Файл с порогами по трём стратегиям из tune_thresholds.",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/test_evaluation.json"))
    parser.add_argument(
        "--scores-cache", type=Path, default=Path("reports/test_scores.json"),
        help="Где кешировать сырые скоры test-сплита.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Урезать test-сплит для CPU-прогона (0 = весь test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else project_root / p

    data_path = _resolve(args.data)
    split_path = _resolve(args.split)
    thresholds_path = _resolve(args.thresholds)
    out_path = _resolve(args.out)
    scores_cache = _resolve(args.scores_cache)

    # ── Сплит и проверка непересечения ───────────────────────────────────
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_set = set(split["train"])
    test_set = set(split["test"])
    overlap = train_set & test_set
    if overlap:
        raise SystemExit(
            f"Train и test пересекаются: {len(overlap)} общих ID — "
            f"проверь {split_path}"
        )
    logger.info(
        "Сплит ок: train=%d, test=%d, пересечение=%d",
        len(train_set), len(test_set), len(overlap),
    )

    test_ids = sorted(test_set)
    if args.max_samples > 0 and args.max_samples < len(test_ids):
        rng = random.Random(args.seed)
        test_ids = sorted(rng.sample(test_ids, args.max_samples))
        logger.warning(
            "Используется подвыборка test: %d из %d (seed=%d).",
            len(test_ids), len(test_set), args.seed,
        )

    # ── Загрузка порогов ────────────────────────────────────────────────
    thresholds_data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    by_strategy = thresholds_data["by_strategy"]

    # ── Скоринг test ─────────────────────────────────────────────────────
    samples = load_halueval(data_path)
    print(f"Test: {len(test_ids)} примеров → {len(test_ids)*2} claims\n")

    scores = load_test_scores(scores_cache, test_ids)
    if scores is not None:
        logger.info("Кеш test-скоров найден и валиден — пропускаю Phase 1.")
    else:
        logger.info("Phase 1: считаем сырые скоры на test (с чекпоинтом каждые 100)...")
        t0 = time.perf_counter()
        fc = FactChecker(device=args.device)
        scores = compute_test_scores_checkpointed(
            samples, test_ids, fc, scores_cache, save_every=100,
        )
        logger.info(
            "Phase 1 завершена за %.1f с. Кеш: %s",
            time.perf_counter() - t0, scores_cache,
        )

    # ── Применение 12 конфигов ───────────────────────────────────────────
    logger.info("Phase 2: применение 12 конфигов (%d методов × %d стратегий)...",
                len(METHODS), len(STRATEGIES))
    t0 = time.perf_counter()

    by_method: Dict[str, Dict[str, Dict]] = {m: {} for m in METHODS}

    for strategy in STRATEGIES:
        strat_data = by_strategy.get(strategy)
        if strat_data is None:
            logger.warning("В thresholds.json нет стратегии %s — пропуск", strategy)
            continue
        best = strat_data["best"]
        for method_key in METHODS:
            train_metrics = best.get(method_key)
            if train_metrics is None:
                continue
            nli_thr = train_metrics["nli_thr"]
            emb_thr = train_metrics["emb_thr"]
            test_metrics = apply_config(scores, method_key, nli_thr, emb_thr)

            train_view = {
                k: train_metrics[k]
                for k in ("precision", "recall", "f1", "accuracy", "q_prod",
                          "tp", "fp", "tn", "fn", "passes_threshold")
            }
            by_method[method_key][strategy] = {
                "thresholds": {"nli_thr": nli_thr, "emb_thr": emb_thr},
                "train": train_view,
                "test": test_metrics,
                "delta_q": test_metrics["q_prod"] - train_metrics["q_prod"],
            }

    logger.info("Phase 2 завершена за %.3f с.", time.perf_counter() - t0)

    # ── Лучшие по test ───────────────────────────────────────────────────
    best_by_test: Dict[str, Dict] = {}
    for method_key in METHODS:
        candidates = [
            (strategy, e) for strategy, e in by_method[method_key].items()
        ]
        if not candidates:
            continue
        # Сортируем по test Q_prod, тай-брейкер — F1
        strategy, entry = max(
            candidates,
            key=lambda se: (se[1]["test"]["q_prod"], se[1]["test"]["f1"]),
        )
        best_by_test[method_key] = {"strategy": strategy, **entry}

    # ── Сохранение JSON ─────────────────────────────────────────────────
    out_payload = {
        "metadata": {
            "n_test_samples": len(test_ids),
            "n_test_claims": len(scores),
            "seed": args.seed,
            "thresholds_source": str(thresholds_path.relative_to(project_root)),
            "target_precision": TARGET_PRECISION,
            "target_recall": TARGET_RECALL,
            "target_q_prod": TARGET_Q_PROD,
        },
        "by_method": by_method,
        "best_by_test_q_prod": {
            m: {
                "strategy": e["strategy"],
                "thresholds": e["thresholds"],
                "test": e["test"],
                "train": e["train"],
                "delta_q": e["delta_q"],
            }
            for m, e in best_by_test.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Печать таблиц ───────────────────────────────────────────────────
    print()
    for method_key in METHODS:
        print(render_method_comparison(method_key, by_method[method_key]))
        print()

    print(render_best_by_test(best_by_test))
    print(f"\nРезультаты сохранены: {out_path}")
    print(f"Скоры (кеш):          {scores_cache}")


if __name__ == "__main__":
    main()
