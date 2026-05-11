#!/usr/bin/env python3
"""McNemar test для пар бинарных классификаторов.

Сравниваем три пары:
  1. Haiku-арбитр vs GPT-арбитр — значимо ли различие 0,833 vs 0,831?
  2. NLI/F1 (порог 0,10) vs Haiku-арбитр — значимо ли превосходство LLM?
  3. NLI/F1 vs GPT-арбитр — то же.

Все источники сравниваются на ОБЩИХ парах (sample_id, kind), для которых
у обеих моделей есть валидное предсказание (parse_fail=None исключаются).

Запуск:

    python -m scripts.mcnemar_tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent

# Каноническая конфигурация NLI: стратегия F1, порог 0,10. Решающее правило
# воспроизводит predict() из tune_thresholds.py: pred = (n_facts == 0) or
# (nli_score > nli_thr).
NLI_THR = 0.10


def load_llm_predictions(path: Path, label: str) -> Dict[Tuple[int, str], bool]:
    """Извлечь предсказания LLM-арбитра. parse_fail (predicted=None) исключаются."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[int, str], bool] = {}
    n_skipped = 0
    for d in data["details"]:
        sid, kind = d["sample_id"], d["kind"]
        pred = d.get("predicted")
        if pred is None:
            n_skipped += 1
            continue
        out[(sid, kind)] = bool(pred)
    print(f"  {label}: {len(out)} предсказаний "
          f"(skip {n_skipped} parse-fail)")
    return out


def load_nli_predictions(path: Path, nli_thr: float = NLI_THR) -> Dict[Tuple[int, str], bool]:
    """Восстановить вердикты NLI-метода с заданным порогом по сырым скорам."""
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data["scores"]
    out: Dict[Tuple[int, str], bool] = {}
    for s in scores:
        sid, kind = s["sample_id"], s["kind"]
        # Воспроизводим predict() из tune_thresholds.py:
        # пустой KB → галлюцинация всегда; иначе nli_score > thr.
        if s["n_facts"] == 0:
            pred = True
        else:
            pred = s["nli_score"] > nli_thr
        out[(sid, kind)] = pred
    print(f"  NLI/F1 (порог {nli_thr}): {len(out)} предсказаний")
    return out


def load_true_labels(path: Path) -> Dict[Tuple[int, str], bool]:
    """expected_hallucination: kind=='hallucinated' → True, иначе False."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[int, str], bool] = {}
    for d in data["details"]:
        sid, kind = d["sample_id"], d["kind"]
        out[(sid, kind)] = bool(d.get("expected", kind == "hallucinated"))
    return out


def mcnemar_test(
    preds_a: Dict[Tuple[int, str], bool],
    preds_b: Dict[Tuple[int, str], bool],
    true_labels: Dict[Tuple[int, str], bool],
    label_a: str,
    label_b: str,
) -> Dict:
    """McNemar test: вычислить b, c, χ²-статистику и p-value.

    b = A правильно, B неправильно.
    c = A неправильно, B правильно.
    Случаи a (оба правильно) и d (оба неправильно) для теста несущественны.
    """
    common_keys = set(preds_a) & set(preds_b) & set(true_labels)
    a = b = c = d = 0
    for key in common_keys:
        true = true_labels[key]
        a_correct = (preds_a[key] == true)
        b_correct = (preds_b[key] == true)
        if a_correct and b_correct:
            a += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
        else:
            d += 1

    statistic: Optional[float] = None
    p_value: Optional[float] = None
    method: str

    n_disc = b + c
    if n_disc == 0:
        method = "trivial"
        p_value = 1.0
    elif n_disc < 25:
        from scipy.stats import binomtest
        method = "exact-binomial"
        p_value = binomtest(min(b, c), n_disc, 0.5,
                            alternative="two-sided").pvalue
    else:
        method = "chi2-yates"
        statistic = (abs(b - c) - 1) ** 2 / n_disc
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    n_total = a + b + c + d
    acc_a = (a + b) / n_total if n_total else 0
    acc_b = (a + c) / n_total if n_total else 0

    return {
        "label_a": label_a, "label_b": label_b,
        "n_common": n_total,
        "a_both_correct": a, "b_only_a_correct": b,
        "c_only_b_correct": c, "d_both_wrong": d,
        "accuracy_a": acc_a, "accuracy_b": acc_b,
        "method": method,
        "statistic": statistic, "p_value": p_value,
        "significant_at_0.05": (p_value is not None and p_value < 0.05),
    }


def render(r: Dict) -> str:
    out = []
    out.append(f"  n общих   = {r['n_common']}")
    out.append(f"  acc {r['label_a']:<14s} = {r['accuracy_a']:.4f}")
    out.append(f"  acc {r['label_b']:<14s} = {r['accuracy_b']:.4f}")
    out.append(f"  a (оба правильно)         = {r['a_both_correct']}")
    out.append(f"  b ({r['label_a']} прав, {r['label_b']} нет) = {r['b_only_a_correct']}")
    out.append(f"  c ({r['label_a']} нет, {r['label_b']} прав) = {r['c_only_b_correct']}")
    out.append(f"  d (оба неправильно)        = {r['d_both_wrong']}")
    out.append(f"  метод                      = {r['method']}")
    if r["statistic"] is not None:
        out.append(f"  χ² (Yates)                 = {r['statistic']:.4f}")
    out.append(f"  p-value                    = {r['p_value']:.6g}")
    sig = "ЗНАЧИМО" if r["significant_at_0.05"] else "не значимо"
    out.append(f"  α=0.05                     → {sig}")
    return "\n".join(out)


def main() -> None:
    print("Загрузка предсказаний...")
    preds_haiku = load_llm_predictions(
        ROOT / "reports" / "llm_baseline_n1991.json", "Haiku-арбитр")
    preds_gpt = load_llm_predictions(
        ROOT / "reports" / "llm_baseline_openai_n1991.json", "GPT-арбитр")
    preds_nli = load_nli_predictions(
        ROOT / "reports" / "test_scores.json", nli_thr=NLI_THR)

    true_labels = load_true_labels(
        ROOT / "reports" / "llm_baseline_n1991.json")
    print(f"  Истинных меток: {len(true_labels)}")
    print()

    tests = [
        ("Haiku vs GPT", preds_haiku, "Haiku", preds_gpt, "GPT"),
        ("NLI/F1 vs Haiku", preds_nli, "NLI/F1", preds_haiku, "Haiku"),
        ("NLI/F1 vs GPT", preds_nli, "NLI/F1", preds_gpt, "GPT"),
    ]

    results = {}
    for name, pa, la, pb, lb in tests:
        print(f"=== McNemar: {name} ===")
        r = mcnemar_test(pa, pb, true_labels, la, lb)
        print(render(r))
        print()
        results[name] = r

    out_path = ROOT / "reports" / "mcnemar_tests.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
