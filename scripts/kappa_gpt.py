#!/usr/bin/env python3
"""Cohen's kappa между двумя автоклассификаторами расхождений.

Считается отдельно для двух подмножеств:
  * расхождения GPT-арбитра — классификации Haiku vs GPT;
  * расхождения Haiku-арбитра — классификации Haiku vs GPT (для сравнения).

Запуск:

    python -m scripts.kappa_gpt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ("BAD_LABEL", "EVASIVE", "MODEL_MISS")


def to_dict(path: Path) -> Dict[Tuple[int, str], str]:
    """{(sample_id, kind): category} из JSON автоклассификатора."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", data.get("details", []))
    out: Dict[Tuple[int, str], str] = {}
    for r in rows:
        cat = r.get("category")
        if cat in CATEGORIES:
            out[(r["sample_id"], r["kind"])] = cat
    return out


def cohen_kappa(
    rater1: Dict[Tuple[int, str], str],
    rater2: Dict[Tuple[int, str], str],
) -> Optional[Dict]:
    common_keys = set(rater1) & set(rater2)
    if not common_keys:
        return None

    n = len(common_keys)
    agree = sum(1 for k in common_keys if rater1[k] == rater2[k])
    p_o = agree / n

    p_e = 0.0
    for cat in CATEGORIES:
        p1 = sum(1 for k in common_keys if rater1[k] == cat) / n
        p2 = sum(1 for k in common_keys if rater2[k] == cat) / n
        p_e += p1 * p2

    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "n": n, "agree": agree,
        "p_observed": p_o, "p_expected": p_e,
        "kappa": kappa,
    }


def interpret(k: float) -> str:
    if k < 0:
        return "хуже случайного"
    if k < 0.20:
        return "плохое (slight)"
    if k < 0.40:
        return "слабое (fair)"
    if k < 0.60:
        return "умеренное (moderate)"
    if k < 0.80:
        return "существенное (substantial)"
    return "почти полное (almost perfect)"


def main() -> None:
    pairs = [
        ("Расхождения GPT-арбитра",
         "reports/disagreements_gpt_classified_by_haiku.json",
         "reports/disagreements_gpt_classified_by_gpt.json"),
        ("Расхождения Haiku-арбитра",
         "reports/disagreements_auto.json",
         "reports/disagreements_auto_openai.json"),
    ]

    out_data: Dict = {}
    for label, p_haiku, p_gpt in pairs:
        haiku_clf = to_dict(ROOT / p_haiku)
        gpt_clf = to_dict(ROOT / p_gpt)
        result = cohen_kappa(haiku_clf, gpt_clf)
        if result is None:
            print(f"=== {label}: нет общих ключей ===\n")
            continue
        interp = interpret(result["kappa"])
        print(f"=== Cohen's kappa: {label} ===")
        print(f"  Haiku-классификатор: {p_haiku.split('/')[-1]} ({len(haiku_clf)} строк)")
        print(f"  GPT-классификатор:   {p_gpt.split('/')[-1]} ({len(gpt_clf)} строк)")
        print(f"  Общих случаев:       {result['n']}")
        print(f"  Согласие наблюдённое (p_o):  {result['p_observed']:.4f} "
              f"= {result['agree']}/{result['n']}")
        print(f"  Согласие ожидаемое (p_e):    {result['p_expected']:.4f}")
        print(f"  Cohen's κ:                   {result['kappa']:.4f}")
        print(f"  Интерпретация:               {interp}")
        print()
        out_data[label] = {**result, "interpretation": interp,
                            "haiku_clf_source": p_haiku,
                            "gpt_clf_source": p_gpt}

    out_path = ROOT / "reports" / "kappa_results.json"
    out_path.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
