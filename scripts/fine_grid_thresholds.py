"""
Fine-grid (шаг 0.01) подбор порогов в окрестности лучших точек coarse-grid.

Логика:

1. Читать ``reports/thresholds_n1000_comparison.json`` — coarse-best
   (шаг 0.05) на n=1000 train-примерах.
2. Для каждой пары (метод × стратегия) построить локальную fine-сетку
   с шагом 0.01 в окрестности ±0.05 от coarse-best. Если coarse-best
   на границе coarse-сетки (NLI ∈ {0.10, 0.90}, EMB ∈ {0.30, 0.85}),
   расширить локальную сетку только в одну сторону (внутрь coarse-области).
3. Использовать существующие сырые скоры из ``reports/train_scores_n1000.json``
   — модели НЕ прогоняем, только пересчёт метрик.
4. Найти fine-best для каждого (метод × стратегия) и сохранить в
   ``reports/thresholds_n1000_finegrid.json``.
5. Применить fine-best к существующим ``reports/test_scores.json`` и
   сохранить в ``reports/test_evaluation_n1000_finegrid.json``.

Запуск:

    nohup python -m scripts.fine_grid_thresholds \\
        > reports/fine_grid.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.tune_thresholds import (
    EMB_GRID,
    NLI_GRID,
    STRATEGIES,
    TARGET_PRECISION,
    TARGET_RECALL,
    ScoredClaim,
    _factchecker_hint,
    _hash_ids,
    best_per_method,
    metrics_from_pairs,
    predict,
)
from scripts.run_test_evaluation import (
    METHODS,
    apply_config,
    render_best_by_test,
    render_method_comparison,
)

logger = logging.getLogger(__name__)


# Границы coarse-сетки берём из tune_thresholds: NLI [0.10, 0.90], EMB [0.30, 0.85].
NLI_LOW, NLI_HIGH = round(min(NLI_GRID), 3), round(max(NLI_GRID), 3)
EMB_LOW, EMB_HIGH = round(min(EMB_GRID), 3), round(max(EMB_GRID), 3)

FINE_STEP = 0.01
FINE_SPAN = 0.05    # ±span вокруг coarse-best
EPS = 1e-6


def _round(x: float) -> float:
    return round(float(x), 3)


def _frange(lo: float, hi: float, step: float) -> List[float]:
    """Сгенерировать [lo, lo+step, ..., hi] с округлением до 3 знаков."""
    out: List[float] = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        out.append(_round(lo + i * step))
    return out


def build_fine_axis(
    best: Optional[float],
    low_bound: float,
    high_bound: float,
    step: float = FINE_STEP,
    span: float = FINE_SPAN,
) -> List[Optional[float]]:
    """Локальная сетка вокруг ``best`` с шагом ``step`` в коридоре ±span.

    Если ``best is None`` — методу этот порог не нужен → ``[None]``.
    Если ``best`` на границе coarse-сетки, вся сетка строится в одну
    сторону (ширина 2·span), чтобы не выходить за coarse-диапазон.
    """
    if best is None:
        return [None]

    best = _round(best)
    width = 2.0 * span

    on_low = abs(best - low_bound) < EPS
    on_high = abs(best - high_bound) < EPS

    if on_low:
        lo, hi = low_bound, _round(low_bound + width)
    elif on_high:
        lo, hi = _round(high_bound - width), high_bound
    else:
        lo, hi = _round(best - span), _round(best + span)

    # Жёсткое отсечение по coarse-границам.
    lo = max(lo, low_bound)
    hi = min(hi, high_bound)

    return [v for v in _frange(lo, hi, step)]


def fine_grid_for_method(
    method: str,
    coarse_best: Dict,
    scores: List[ScoredClaim],
) -> Tuple[List[Dict], List[Optional[float]], List[Optional[float]]]:
    """Перебрать fine-сетку для ОДНОГО метода вокруг coarse-best этого метода.

    Возвращает (плоский список точек, ось NLI, ось EMB).
    Точки помечены ``method``, чтобы можно было выбрать best_per_method.
    """
    nli_axis: List[Optional[float]]
    emb_axis: List[Optional[float]]

    if method == "nli":
        nli_axis = build_fine_axis(coarse_best["nli_thr"], NLI_LOW, NLI_HIGH)
        emb_axis = [None]
    elif method == "embedding":
        nli_axis = [None]
        emb_axis = build_fine_axis(coarse_best["emb_thr"], EMB_LOW, EMB_HIGH)
    elif method in ("combined_or", "combined_and"):
        nli_axis = build_fine_axis(coarse_best["nli_thr"], NLI_LOW, NLI_HIGH)
        emb_axis = build_fine_axis(coarse_best["emb_thr"], EMB_LOW, EMB_HIGH)
    else:
        raise ValueError(f"Неизвестный метод: {method!r}")

    points: List[Dict] = []
    for n in nli_axis:
        for e in emb_axis:
            if method == "nli":
                pairs = predict(scores, "nli", n, None)
            elif method == "embedding":
                pairs = predict(scores, "embedding", None, e)
            elif method == "combined_or":
                pairs = predict(scores, "combined", n, e, combined_mode="or")
            else:  # combined_and
                pairs = predict(scores, "combined", n, e, combined_mode="and")
            m = metrics_from_pairs(pairs)
            points.append({"method": method, "nli_thr": n, "emb_thr": e, **m})

    return points, nli_axis, emb_axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse", type=Path,
        default=Path("reports/thresholds_n1000_comparison.json"),
    )
    parser.add_argument(
        "--train-scores", type=Path,
        default=Path("reports/train_scores_n1000.json"),
    )
    parser.add_argument(
        "--test-scores", type=Path,
        default=Path("reports/test_scores.json"),
    )
    parser.add_argument(
        "--out-thresholds", type=Path,
        default=Path("reports/thresholds_n1000_finegrid.json"),
    )
    parser.add_argument(
        "--out-test-eval", type=Path,
        default=Path("reports/test_evaluation_n1000_finegrid.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else project_root / p

    coarse_path = _resolve(args.coarse)
    train_path = _resolve(args.train_scores)
    test_path = _resolve(args.test_scores)
    out_thr_path = _resolve(args.out_thresholds)
    out_test_path = _resolve(args.out_test_eval)

    # ── 1. Читаем coarse-best ────────────────────────────────────────────
    coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
    by_strategy_coarse = coarse["by_strategy"]

    # ── 2. Читаем train-скоры n=1000 ─────────────────────────────────────
    train_payload = json.loads(train_path.read_text(encoding="utf-8"))
    train_scores = [ScoredClaim.from_dict(d) for d in train_payload["scores"]]
    train_meta = train_payload.get("metadata", {})
    train_ids_hash = train_meta.get("ids_hash")
    n_train_claims = len(train_scores)
    print(f"Train n=1000:  {n_train_claims} claims (ids_hash={train_ids_hash})")

    # ── 3. Fine-grid: для каждого (стратегия × метод) строим локальную сетку ──
    by_strategy_fine: Dict[str, Dict] = {}
    grid_summary: Dict[str, Dict] = {}

    t0 = time.perf_counter()
    for strategy in STRATEGIES:
        coarse_best = by_strategy_coarse[strategy]["best"]

        all_points: List[Dict] = []
        axes: Dict[str, Dict[str, List[Optional[float]]]] = {}
        for method in METHODS:
            cb = coarse_best.get(method)
            if not cb:
                continue
            points, nli_axis, emb_axis = fine_grid_for_method(method, cb, train_scores)
            all_points.extend(points)
            axes[method] = {
                "nli_axis": nli_axis,
                "emb_axis": emb_axis,
                "n_points": len(points),
                "coarse_best": {
                    "nli_thr": cb["nli_thr"], "emb_thr": cb["emb_thr"],
                    "q_prod": cb["q_prod"], "f1": cb["f1"],
                    "precision": cb["precision"], "recall": cb["recall"],
                },
            }

        fine_best = best_per_method(all_points, strategy=strategy)
        by_strategy_fine[strategy] = {
            "best": fine_best,
            "factchecker_init_hint": _factchecker_hint(fine_best),
            "axes": axes,
        }
        grid_summary[strategy] = {
            method: axes[method]["n_points"]
            for method in axes
        }

    elapsed = time.perf_counter() - t0
    logger.info("Fine-grid завершён за %.2f с (%d стратегий).",
                elapsed, len(STRATEGIES))

    # ── 4. Сохраняем fine-grid отчёт ─────────────────────────────────────
    out_thr_path.parent.mkdir(parents=True, exist_ok=True)
    out_thr_path.write_text(json.dumps({
        "metadata": {
            "n_train_claims": n_train_claims,
            "n_train_samples": train_meta.get("n_samples"),
            "ids_hash": train_ids_hash,
            "coarse_source": str(coarse_path.relative_to(project_root)),
            "coarse_step": 0.05,
            "fine_step": FINE_STEP,
            "fine_span": FINE_SPAN,
            "nli_bounds": [NLI_LOW, NLI_HIGH],
            "emb_bounds": [EMB_LOW, EMB_HIGH],
            "target_precision": TARGET_PRECISION,
            "target_recall": TARGET_RECALL,
            "strategies": list(STRATEGIES),
            "n_points_per_strategy_method": grid_summary,
            "elapsed_sec": elapsed,
        },
        "by_strategy": by_strategy_fine,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fine-grid сохранён:  {out_thr_path}")

    # ── 5. Применяем fine-best к test-скорам ─────────────────────────────
    test_payload = json.loads(test_path.read_text(encoding="utf-8"))
    test_scores = [ScoredClaim.from_dict(d) for d in test_payload["scores"]]
    print(f"Test:                {len(test_scores)} claims")

    by_method: Dict[str, Dict[str, Dict]] = {m: {} for m in METHODS}
    for strategy in STRATEGIES:
        fine_best = by_strategy_fine[strategy]["best"]
        for method in METHODS:
            train_metrics = fine_best.get(method)
            if not train_metrics:
                continue
            nli_thr = train_metrics["nli_thr"]
            emb_thr = train_metrics["emb_thr"]
            test_metrics = apply_config(test_scores, method, nli_thr, emb_thr)
            train_view = {k: train_metrics[k] for k in (
                "precision", "recall", "f1", "accuracy", "q_prod",
                "tp", "fp", "tn", "fn", "passes_threshold")}
            by_method[method][strategy] = {
                "thresholds": {"nli_thr": nli_thr, "emb_thr": emb_thr},
                "train": train_view,
                "test": test_metrics,
                "delta_q": test_metrics["q_prod"] - train_metrics["q_prod"],
            }

    best_by_test: Dict[str, Dict] = {}
    for method in METHODS:
        candidates = list(by_method[method].items())
        if not candidates:
            continue
        strategy, entry = max(
            candidates,
            key=lambda se: (se[1]["test"]["q_prod"], se[1]["test"]["f1"]),
        )
        best_by_test[method] = {"strategy": strategy, **entry}

    out_payload = {
        "metadata": {
            "n_test_samples": test_payload["metadata"].get("n_test_samples"),
            "n_test_claims": len(test_scores),
            "n_train_samples_calibration": train_meta.get("n_samples"),
            "thresholds_source": str(out_thr_path.relative_to(project_root)),
            "test_scores_source": str(test_path.relative_to(project_root)),
            "fine_step": FINE_STEP,
            "fine_span": FINE_SPAN,
            "target_precision": TARGET_PRECISION,
            "target_recall": TARGET_RECALL,
            "target_q_prod": 0.5 * TARGET_PRECISION + 0.5 * TARGET_RECALL,
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
    out_test_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"Test eval сохранён:  {out_test_path}")

    # ── 6. Печатаем сравнение по методам ─────────────────────────────────
    print()
    for method in METHODS:
        print(render_method_comparison(method, by_method[method]))
        print()
    print(render_best_by_test(best_by_test))

    # ── 7. Сводка fine vs coarse: насколько ушли пороги и метрики ────────
    print()
    print("=" * 100)
    print("=== ИЗМЕНЕНИЯ FINE vs COARSE (на train n=1000) ===")
    print("=" * 100)
    for strategy in STRATEGIES:
        print(f"\n[стратегия: {strategy}]")
        cb_all = by_strategy_coarse[strategy]["best"]
        fb_all = by_strategy_fine[strategy]["best"]
        print(f"  {'method':<14} {'coarse thr':<14} {'fine thr':<14} "
              f"{'coarse Q':>9} {'fine Q':>9} {'ΔQ':>9}")
        for method in METHODS:
            cb = cb_all.get(method)
            fb = fb_all.get(method)
            if not cb or not fb:
                continue
            ct = (cb["nli_thr"], cb["emb_thr"])
            ft = (fb["nli_thr"], fb["emb_thr"])
            ct_str = "/".join("—" if v is None else f"{v:.2f}" for v in ct)
            ft_str = "/".join("—" if v is None else f"{v:.2f}" for v in ft)
            dq = fb["q_prod"] - cb["q_prod"]
            print(f"  {method:<14} {ct_str:<14} {ft_str:<14} "
                  f"{cb['q_prod']:>9.4f} {fb['q_prod']:>9.4f} {dq:>+9.4f}")
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
