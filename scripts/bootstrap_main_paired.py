#!/usr/bin/env python3
"""Bootstrap CI с парной структурой HaluEval-QA.

Каждый sample_id даёт ДВЕ связанные проверки (right_answer и
hallucinated_answer). Корректный bootstrap ресэмплирует sample_id,
после чего для каждого сэмплированного id берутся ОБЕ его проверки.
Это учитывает зависимость двух проверок одного примера (общий контекст
и вопрос) и даёт корректные CI.

Решающие правила воспроизводят predict() из tune_thresholds.py
(пустой KB → True; > для NLI, < для embedding).

Запуск:

    python -m scripts.bootstrap_main_paired
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CLASSICAL_METHODS: Dict[str, Dict] = {
    "NLI/F1": {"method": "nli", "nli_thr": 0.10, "emb_thr": None},
    "embedding/q_prod": {"method": "embedding", "nli_thr": None, "emb_thr": 0.85},
    "combined-or/q_prod": {"method": "combined-or", "nli_thr": 0.10, "emb_thr": 0.85},
    "combined-and/q_prod": {"method": "combined-and", "nli_thr": 0.90, "emb_thr": 0.85},
}

LLM_FILES = [
    ("LLM-as-judge (Haiku)", "reports/llm_baseline_n1991.json"),
    ("LLM-as-judge (GPT)",   "reports/llm_baseline_openai_n1991.json"),
]


def predict_classical(item: Dict, method: str, nli_thr, emb_thr) -> bool:
    if item["n_facts"] == 0:
        return True
    nli, emb = item["nli_score"], item["emb_score"]
    if method == "nli":
        return nli > nli_thr
    if method == "embedding":
        return emb < emb_thr
    if method == "combined-or":
        return (nli > nli_thr) or (emb < emb_thr)
    if method == "combined-and":
        return (nli > nli_thr) and (emb < emb_thr)
    raise ValueError(method)


def metrics_from_arrays(preds: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(preds & truth))
    fp = int(np.sum(preds & ~truth))
    fn = int(np.sum(~preds & truth))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "q_prod": (p + r) / 2.0}


def bootstrap_ci_paired(
    preds: np.ndarray, truth: np.ndarray, sample_ids: np.ndarray,
    n_iter: int = 10000, seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    # Группируем индексы по sample_id
    by_sid: Dict[int, List[int]] = defaultdict(list)
    for i, sid in enumerate(sample_ids.tolist()):
        by_sid[sid].append(i)
    unique_sids = np.array(list(by_sid.keys()))
    n_pairs = len(unique_sids)

    # Преобразуем в массив-массивов фиксированной длины (обычно 2)
    # Большинство sample_id даёт 2 проверки. Кладём индексы в плоский массив
    # + индекс начала каждой группы.
    flat_idx = []
    starts = [0]
    for sid in unique_sids:
        flat_idx.extend(by_sid[sid])
        starts.append(len(flat_idx))
    flat_idx = np.array(flat_idx, dtype=np.int64)
    starts = np.array(starts, dtype=np.int64)
    sid_pos = {int(sid): i for i, sid in enumerate(unique_sids)}

    iters = {k: np.empty(n_iter) for k in ("precision", "recall", "f1", "q_prod")}

    for it in range(n_iter):
        sampled_pos = rng.integers(0, n_pairs, size=n_pairs)
        # Собираем индексы для всех проверок ресэмплированных sample_id.
        idx_list = []
        for pos in sampled_pos:
            idx_list.append(flat_idx[starts[pos]:starts[pos + 1]])
        sampled_idx = np.concatenate(idx_list)
        m = metrics_from_arrays(preds[sampled_idx], truth[sampled_idx])
        for k in iters:
            iters[k][it] = m[k]

    return {
        k: {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
        }
        for k, arr in iters.items()
    }


def load_test_scores() -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    payload = json.loads((ROOT / "reports" / "test_scores.json")
                         .read_text(encoding="utf-8"))
    items = payload["scores"]
    truth = np.array([it["expected_hallucination"] for it in items], dtype=bool)
    sids = np.array([it["sample_id"] for it in items], dtype=np.int64)
    return truth, sids, items


def load_llm(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preds, truth, sids = [], [], []
    for d in payload["details"]:
        pred = d.get("predicted")
        if pred is None:
            continue
        preds.append(bool(pred))
        truth.append(bool(d.get("expected", d["kind"] == "hallucinated")))
        sids.append(d["sample_id"])
    return (np.array(preds, dtype=bool), np.array(truth, dtype=bool),
            np.array(sids, dtype=np.int64))


def fmt_ci(name: str, m: Dict[str, float]) -> str:
    return (f"  {name:<10s} = {m['mean']:.4f}  "
            f"[{m['lower']:.4f}; {m['upper']:.4f}]  (std={m['std']:.4f})")


def main() -> None:
    print("=== Paired Bootstrap CI (n_iter=10000, seed=42) ===\n")

    truth_all, sids_all, items = load_test_scores()
    n_unique = len(set(sids_all.tolist()))
    print(f"Загружено {len(items)} проверок, {n_unique} уникальных sample_id\n")

    results: Dict[str, Dict] = {}

    for label, cfg in CLASSICAL_METHODS.items():
        preds = np.array(
            [predict_classical(it, cfg["method"], cfg["nli_thr"], cfg["emb_thr"])
             for it in items], dtype=bool,
        )
        point = metrics_from_arrays(preds, truth_all)
        cis = bootstrap_ci_paired(preds, truth_all, sids_all)

        print(f"--- {label} ---")
        print(f"  Точечно: P={point['precision']:.4f} R={point['recall']:.4f} "
              f"F1={point['f1']:.4f} Q={point['q_prod']:.4f}")
        for k in ("precision", "recall", "f1", "q_prod"):
            print(fmt_ci(k, cis[k]))
        print()
        results[label] = {"point": point, "ci": cis,
                          "thresholds": {"nli_thr": cfg["nli_thr"],
                                         "emb_thr": cfg["emb_thr"]}}

    for label, rel_path in LLM_FILES:
        print(f"--- {label} ---")
        preds_llm, truth_llm, sids_llm = load_llm(ROOT / rel_path)
        point = metrics_from_arrays(preds_llm, truth_llm)
        cis = bootstrap_ci_paired(preds_llm, truth_llm, sids_llm)
        print(f"  Точечно: P={point['precision']:.4f} R={point['recall']:.4f} "
              f"F1={point['f1']:.4f} Q={point['q_prod']:.4f}")
        for k in ("precision", "recall", "f1", "q_prod"):
            print(fmt_ci(k, cis[k]))
        print()
        results[label] = {"point": point, "ci": cis}

    out = ROOT / "reports" / "bootstrap_main_paired.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=" * 78)
    print("СВОДНАЯ ТАБЛИЦА (для подстановки в Таблицу 4)")
    print("=" * 78)
    print(f"{'Метод':<32s} {'Q_prod':>8s}  {'95% CI Q_prod':>22s}")
    print("-" * 78)
    for name, r in results.items():
        q = r["point"]["q_prod"]
        c = r["ci"]["q_prod"]
        print(f"{name:<32s} {q:>8.4f}  [{c['lower']:.4f}; {c['upper']:.4f}]")
    print("=" * 78)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
