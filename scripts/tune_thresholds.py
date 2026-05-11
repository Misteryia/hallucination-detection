"""
Подбор порогов FactChecker на train-сплите HaluEval.

Скрипт состоит из двух фаз:

1. **Scoring (медленно, кешируется).** Для каждого claim из train-сплита
   считаются СЫРЫЕ скоры:
     * ``nli_score`` — max P(contradiction) по фактам KB,
     * ``emb_score`` — max cosine similarity по фактам KB.
   Один проход через NLI- и embedding-модели сохраняется в файл, чтобы
   при последующих запусках Фаза 1 пропускалась.

2. **Grid search (мгновенно).** Из кеша сырых скоров перебираются пары
   ``(nli_threshold, similarity_threshold)`` для каждого из методов
   (nli, embedding, combined_or, combined_and). Для каждой точки сетки
   считаются метрики (P, R, F1, accuracy, Q_prod) и выбирается лучшая
   по Q_prod. Если хочется поменять метрику оптимизации или сетку —
   достаточно перезапустить только Фазу 2 (см. ``--from-cache``).

Используется FactChecker из ``halu_detect.fact_checker`` — те же модели
и те же ``score_*`` методы, которые будут применяться на тестовом
прогоне. Это гарантирует, что подобранные пороги релевантны.

Запуск:
    # Полный прогон (scoring + grid search):
    python -m scripts.tune_thresholds

    # Только grid search по уже сохранённым скорам:
    python -m scripts.tune_thresholds --from-cache

    # Ускоренный прогон на подвыборке 500 примеров (для пилота):
    python -m scripts.tune_thresholds --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from halu_detect.data_loader import HaluEvalSample, load_halueval
from halu_detect.fact_checker import FactChecker
from halu_detect.utils import split_to_facts

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Сетка порогов
# ─────────────────────────────────────────────────────────────────────────

# NLI: contradiction_prob > порог → галлюцинация. Сетка плотная вокруг 0.5.
NLI_GRID = [round(x, 3) for x in np.arange(0.10, 0.91, 0.05)]
# Embedding: max_similarity < порог → галлюцинация. Сетка от 0.30 до 0.85.
EMB_GRID = [round(x, 3) for x in np.arange(0.30, 0.86, 0.05)]

# Целевые показатели качества (см. Aggregator)
TARGET_PRECISION = 0.80
TARGET_RECALL = 0.75


# ─────────────────────────────────────────────────────────────────────────
# Фаза 1: сырые скоры
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ScoredClaim:
    sample_id: int
    kind: str  # "right" | "hallucinated"
    expected_hallucination: bool
    n_facts: int
    nli_score: float          # max P(contradiction) по фактам, 0.0 если фактов нет
    emb_score: float          # max cosine similarity, 0.0 если фактов нет
    nli_matched_fact: str
    emb_matched_fact: str

    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "kind": self.kind,
            "expected_hallucination": self.expected_hallucination,
            "n_facts": self.n_facts,
            "nli_score": self.nli_score,
            "emb_score": self.emb_score,
            "nli_matched_fact": self.nli_matched_fact,
            "emb_matched_fact": self.emb_matched_fact,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ScoredClaim":
        return cls(**d)


def compute_train_scores(
    samples: List[HaluEvalSample],
    train_ids: List[int],
    fact_checker: FactChecker,
    desc: str = "Scoring train",
) -> List[ScoredClaim]:
    """Прогнать FactChecker по списку sample_id и сохранить сырые скоры.

    Имя сохранено по историческим причинам — функция универсальна и
    используется также для test-сплита (см. ``scripts/run_test_evaluation``).
    """
    out: List[ScoredClaim] = []
    for sid in tqdm(train_ids, desc=desc):
        sample = samples[sid]
        facts = split_to_facts(sample.knowledge)
        for kind, claim, expected in (
            ("right", sample.right_answer, False),
            ("hallucinated", sample.hallucinated_answer, True),
        ):
            if not facts:
                # Воспроизводим фолбэк FactChecker._empty_kb_verdict:
                # без фактов любое утверждение помечается как галлюцинация.
                # В скорах фиксируем это нулевыми значениями + n_facts=0,
                # а в predict-функции применяем то же правило.
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
    return out


def save_scores(scores: List[ScoredClaim], path: Path, train_ids: List[int]) -> None:
    payload = {
        "metadata": {
            "n_claims": len(scores),
            "train_ids_hash": _hash_ids(train_ids),
            "n_train_samples": len(train_ids),
        },
        "scores": [s.to_dict() for s in scores],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scores(path: Path, train_ids: List[int]) -> Optional[List[ScoredClaim]]:
    """Загрузить кеш и проверить, что он соответствует текущему train-сплиту."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cached_hash = payload.get("metadata", {}).get("train_ids_hash")
    expected_hash = _hash_ids(train_ids)
    if cached_hash != expected_hash:
        logger.warning(
            "Кеш скоров не соответствует текущему train-сплиту "
            "(cached=%s, expected=%s); пересчитываем.", cached_hash, expected_hash
        )
        return None
    return [ScoredClaim.from_dict(d) for d in payload["scores"]]


def _hash_ids(ids: List[int]) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(",".join(str(i) for i in sorted(ids)).encode("utf-8"))
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────
# Фаза 2: предсказания и метрики
# ─────────────────────────────────────────────────────────────────────────


def predict(
    scores: List[ScoredClaim],
    method: str,
    nli_thr: Optional[float],
    emb_thr: Optional[float],
    combined_mode: str = "or",
) -> List[Tuple[bool, bool]]:
    """Получить пары (expected, predicted) для заданного метода и порога(ов).

    Воспроизводит логику FactChecker._check_*_internal и _empty_kb_verdict:
      * nli:        max_contradiction > nli_thr;
      * embedding:  max_similarity < emb_thr;
      * combined:   объединение по or / and;
      * пустой KB:  всегда галлюцинация (как в _empty_kb_verdict).
    """
    pairs: List[Tuple[bool, bool]] = []
    for s in scores:
        if s.n_facts == 0:
            pred = True
        elif method == "nli":
            pred = s.nli_score > nli_thr
        elif method == "embedding":
            pred = s.emb_score < emb_thr
        elif method == "combined":
            nli_pred = s.nli_score > nli_thr
            emb_pred = s.emb_score < emb_thr
            pred = (nli_pred or emb_pred) if combined_mode == "or" else (nli_pred and emb_pred)
        else:
            raise ValueError(f"Неизвестный метод: {method!r}")
        pairs.append((s.expected_hallucination, bool(pred)))
    return pairs


def metrics_from_pairs(
    pairs: List[Tuple[bool, bool]],
    target_precision: float = TARGET_PRECISION,
    target_recall: float = TARGET_RECALL,
) -> Dict:
    """Подсчитать метрики из списка (expected, predicted).

    Формулы и определения совпадают с Aggregator.compute_metrics
    (см. тест в конце файла, который это проверяет).
    """
    tp = sum(1 for e, p in pairs if e and p)
    fp = sum(1 for e, p in pairs if not e and p)
    tn = sum(1 for e, p in pairs if not e and not p)
    fn = sum(1 for e, p in pairs if e and not p)
    total = tp + fp + tn + fn

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    q_prod = 0.5 * p + 0.5 * r

    threshold = 0.5 * target_precision + 0.5 * target_recall
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "accuracy": acc, "q_prod": q_prod,
        "passes_threshold": q_prod >= threshold and total > 0,
    }


def grid_search(scores: List[ScoredClaim]) -> List[Dict]:
    """Перебрать все методы × все пороги, вернуть плоский список точек сетки."""
    grid: List[Dict] = []

    # NLI alone
    for n in NLI_GRID:
        m = metrics_from_pairs(predict(scores, "nli", n, None))
        grid.append({"method": "nli", "nli_thr": n, "emb_thr": None, **m})

    # Embedding alone
    for e in EMB_GRID:
        m = metrics_from_pairs(predict(scores, "embedding", None, e))
        grid.append({"method": "embedding", "nli_thr": None, "emb_thr": e, **m})

    # Combined: 2D-сетка для двух режимов
    for n in NLI_GRID:
        for e in EMB_GRID:
            for mode in ("or", "and"):
                m = metrics_from_pairs(predict(scores, "combined", n, e, mode))
                grid.append({
                    "method": f"combined_{mode}",
                    "nli_thr": n, "emb_thr": e, **m,
                })
    return grid


STRATEGIES = ("q_prod", "f1", "q_prod_constrained")


def best_per_method(
    grid: List[Dict],
    strategy: str = "q_prod",
    target_precision: float = TARGET_PRECISION,
) -> Dict[str, Dict]:
    """Найти лучшую точку сетки для каждого метода в заданной стратегии.

    Поддерживаются три стратегии оптимизации:

    * ``"q_prod"`` — максимум ``Q_prod = 0.5·P + 0.5·R``;
      тай-брейкеры: F1, затем Precision.
    * ``"f1"`` — максимум F1; тай-брейкеры: Q_prod, затем Precision.
    * ``"q_prod_constrained"`` — максимум Q_prod **при условии**
      ``Precision ≥ target_precision``. Если в сетке нет ни одной
      такой точки для метода, делаем фолбэк: выбираем точку с
      максимальным Precision (тай-брейкер — F1). Это даёт устойчивый
      результат даже когда ни одна комбинация порогов не достигает
      целевой Precision (типичная ситуация на embedding-only).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Неизвестная стратегия: {strategy!r}; ожидается одна из {STRATEGIES}")

    methods = sorted({r["method"] for r in grid})
    best: Dict[str, Dict] = {}

    for method in methods:
        candidates = [r for r in grid if r["method"] == method]

        if strategy == "q_prod":
            best[method] = max(
                candidates,
                key=lambda r: (r["q_prod"], r["f1"], r["precision"]),
            )
        elif strategy == "f1":
            best[method] = max(
                candidates,
                key=lambda r: (r["f1"], r["q_prod"], r["precision"]),
            )
        elif strategy == "q_prod_constrained":
            qualifying = [r for r in candidates if r["precision"] >= target_precision]
            if qualifying:
                best[method] = max(
                    qualifying,
                    key=lambda r: (r["q_prod"], r["f1"], r["precision"]),
                )
            else:
                # Фолбэк: ни одна точка не достигла target precision —
                # берём ту, у которой precision максимален (с тай-брейкером F1).
                best[method] = max(
                    candidates,
                    key=lambda r: (r["precision"], r["f1"]),
                )

    return best


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def render_summary(
    best: Dict[str, Dict],
    title: str = "ЛУЧШИЕ ПОРОГИ ПО КАЖДОМУ МЕТОДУ",
) -> str:
    out = []
    out.append("=" * 88)
    out.append(title)
    out.append("=" * 88)
    out.append(
        f"{'method':<14} {'nli_thr':>8} {'emb_thr':>8} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'Q_prod':>7} {'pass?':>6}"
    )
    out.append("-" * 88)
    for method in ("nli", "embedding", "combined_or", "combined_and"):
        r = best.get(method)
        if not r:
            continue
        nli = f"{r['nli_thr']:.2f}" if r["nli_thr"] is not None else "—"
        emb = f"{r['emb_thr']:.2f}" if r["emb_thr"] is not None else "—"
        ok = "✓" if r["passes_threshold"] else "✗"
        out.append(
            f"{method:<14} {nli:>8} {emb:>8} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{r['accuracy']:>6.3f} {r['q_prod']:>7.3f} {ok:>6}"
        )
    out.append("=" * 88)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/qa_data.json"))
    parser.add_argument("--split", type=Path, default=Path("splits/halueval_split.json"))
    parser.add_argument(
        "--scores-cache", type=Path, default=Path("reports/train_scores.json"),
        help="Где кешировать сырые скоры (NLI и embedding).",
    )
    parser.add_argument(
        "--grid-out", type=Path, default=Path("reports/threshold_grid.json"),
        help="Куда сохранить полную сетку метрик.",
    )
    parser.add_argument(
        "--best-out", type=Path, default=Path("reports/thresholds.json"),
        help="Куда сохранить лучшие пороги по каждому методу.",
    )
    parser.add_argument(
        "--compare-out", type=Path, default=Path("reports/thresholds_comparison.json"),
        help="Куда сохранить сравнительный отчёт при --compare-strategies.",
    )
    parser.add_argument(
        "--optimize-by", choices=list(STRATEGIES), default="q_prod",
        help="Стратегия выбора лучших порогов в одиночном прогоне "
             "(игнорируется при --compare-strategies).",
    )
    parser.add_argument(
        "--compare-strategies", action="store_true",
        help="Запустить grid search один раз и посчитать best для всех "
             "трёх стратегий (q_prod, f1, q_prod_constrained); сохранить "
             "сводный отчёт в --compare-out.",
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Не запускать FactChecker; считать всё из кеша скоров. "
             "Полезно при изменении сетки или метрики оптимизации.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Ограничить число train-примеров для скоринга (0 = весь train). "
             "Полезно для пилота на CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # Воспроизводимость
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
    scores_cache = _resolve(args.scores_cache)
    grid_out = _resolve(args.grid_out)
    best_out = _resolve(args.best_out)

    # Загрузка split и данных
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = split["train"]
    if args.max_samples > 0 and args.max_samples < len(train_ids):
        rng = random.Random(args.seed)
        train_ids = sorted(rng.sample(train_ids, args.max_samples))
        logger.warning(
            "Используется подвыборка train: %d из %d (seed=%d).",
            len(train_ids), len(split["train"]), args.seed,
        )

    samples = load_halueval(data_path)
    print(f"Train: {len(train_ids)} примеров → {len(train_ids)*2} claims\n")

    # ── Фаза 1: scoring (с кешем) ─────────────────────────────────────────
    scores = load_scores(scores_cache, train_ids)
    if scores is not None and not args.from_cache:
        logger.info("Кеш скоров найден и валиден — пропускаю Фазу 1.")
    if scores is None:
        if args.from_cache:
            raise SystemExit(
                f"--from-cache указан, но в {scores_cache} нет валидного "
                f"кеша для текущего train-сплита."
            )
        logger.info("Фаза 1: считаем сырые скоры FactChecker'ом...")
        t0 = time.perf_counter()
        fc = FactChecker(device=args.device)
        scores = compute_train_scores(samples, train_ids, fc)
        scores_cache.parent.mkdir(parents=True, exist_ok=True)
        save_scores(scores, scores_cache, train_ids)
        logger.info(
            "Фаза 1 завершена за %.1f с. Кеш сохранён: %s",
            time.perf_counter() - t0, scores_cache,
        )

    # ── Фаза 2: grid search ──────────────────────────────────────────────
    logger.info("Фаза 2: grid search по %d × %d точкам сетки...",
                len(NLI_GRID), len(EMB_GRID))
    t0 = time.perf_counter()
    grid = grid_search(scores)
    logger.info(
        "Фаза 2 завершена за %.2f с (%d точек сетки).",
        time.perf_counter() - t0, len(grid),
    )

    # Полная сетка пишется всегда — она не зависит от стратегии выбора best.
    grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid_out.write_text(
        json.dumps({
            "metadata": {
                "n_train_claims": len(scores),
                "nli_grid": NLI_GRID,
                "emb_grid": EMB_GRID,
                "target_precision": TARGET_PRECISION,
                "target_recall": TARGET_RECALL,
            },
            "grid": grid,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.compare_strategies:
        # ── Сравнительный отчёт по трём стратегиям ──────────────────────
        compare_out = _resolve(args.compare_out)
        compare_out.parent.mkdir(parents=True, exist_ok=True)

        by_strategy: Dict[str, Dict] = {}
        for strategy in STRATEGIES:
            best = best_per_method(grid, strategy=strategy)
            by_strategy[strategy] = {
                "best": best,
                "factchecker_init_hint": _factchecker_hint(best),
            }

        compare_out.write_text(
            json.dumps({
                "metadata": {
                    "n_train_claims": len(scores),
                    "nli_grid": NLI_GRID,
                    "emb_grid": EMB_GRID,
                    "target_precision": TARGET_PRECISION,
                    "target_recall": TARGET_RECALL,
                    "strategies": list(STRATEGIES),
                    "tiebreak_q_prod": ["f1", "precision"],
                    "tiebreak_f1": ["q_prod", "precision"],
                    "tiebreak_q_prod_constrained": ["f1", "precision"],
                    "constrained_fallback": "max precision (tiebreak f1)",
                },
                "by_strategy": by_strategy,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print()
        for strategy in STRATEGIES:
            title = f"ЛУЧШИЕ ПОРОГИ — стратегия: {strategy}"
            print(render_summary(by_strategy[strategy]["best"], title=title))
            print()
        print(f"Полная сетка:        {grid_out}")
        print(f"Сравнительный отчёт: {compare_out}")
        print("\nfactchecker_init_hint доступен внутри by_strategy[<strategy>].")
        return

    # ── Одиночная стратегия ───────────────────────────────────────────────
    strategy = args.optimize_by
    best = best_per_method(grid, strategy=strategy)

    if strategy == "q_prod":
        tiebreak = ["f1", "precision"]
    elif strategy == "f1":
        tiebreak = ["q_prod", "precision"]
    else:  # q_prod_constrained
        tiebreak = ["f1", "precision"]

    best_out.write_text(
        json.dumps({
            "metadata": {
                "optimized_by": strategy,
                "tiebreak": tiebreak,
                "n_train_claims": len(scores),
                "target_precision": TARGET_PRECISION,
                "target_recall": TARGET_RECALL,
            },
            "best": best,
            "factchecker_init_hint": _factchecker_hint(best),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(render_summary(
        best,
        title=f"ЛУЧШИЕ ПОРОГИ — стратегия: {strategy}",
    ))
    print(f"\nПолная сетка:  {grid_out}")
    print(f"Лучшие пороги: {best_out}")
    print("\nГотовые параметры для FactChecker(...) при финальной оценке "
          "см. в поле 'factchecker_init_hint' в файле выше.")


def _factchecker_hint(best: Dict[str, Dict]) -> Dict[str, Dict]:
    """Заготовки kwargs для каждого режима, готовые к подстановке в FactChecker."""
    out: Dict[str, Dict] = {}
    if "nli" in best:
        out["nli"] = {"nli_threshold": best["nli"]["nli_thr"]}
    if "embedding" in best:
        out["embedding"] = {"similarity_threshold": best["embedding"]["emb_thr"]}
    for mode in ("or", "and"):
        key = f"combined_{mode}"
        if key in best:
            out[key] = {
                "nli_threshold": best[key]["nli_thr"],
                "similarity_threshold": best[key]["emb_thr"],
                "combined_mode": mode,
            }
    return out


if __name__ == "__main__":
    main()