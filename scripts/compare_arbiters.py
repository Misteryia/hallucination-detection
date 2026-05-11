"""
Сравнение двух автоматических арбитров расхождений LLM↔HaluEval —
Claude Haiku 4.5 (Anthropic) и GPT-5.4-mini (OpenAI).

Считает confusion matrix 3×3 их вердиктов и Cohen's kappa, и
вытаскивает наиболее показательные подмножества:

  * Haiku=BAD_LABEL, GPT=MODEL_MISS — кейсы, где Haiku оправдывает
    другого LLM-судью (тоже Claude), а GPT, будучи независимой
    моделью, диагностирует ошибку именно судьи. Это эмпирическая
    подсветка возможной circular-reasoning у Haiku.
  * обе на BAD_LABEL — кейсы, где обе модели уверены, что
    проблема в разметке HaluEval; самая твёрдая иллюстрация
    устаревания датасета.

Запуск:

    python -m scripts.compare_arbiters > reports/arbiter_comparison.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Порядок категорий используем во всех таблицах:
# a = BAD_LABEL, b = EVASIVE, c = MODEL_MISS.
CATEGORIES = ("BAD_LABEL", "EVASIVE", "MODEL_MISS")
CAT_LABEL = {"BAD_LABEL": "a", "EVASIVE": "b", "MODEL_MISS": "c"}


def _load_rows(path: Path) -> Dict[str, Dict]:
    """Прочитать JSON автоклассификатора и вернуть rows по ключу sid/kind."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return {f"{r['sample_id']}/{r['kind']}": r for r in rows}


def cohen_kappa(pairs: List[Tuple[str, str]]) -> float:
    """Cohen's κ для двух разметчиков с фиксированной номенклатурой.

    Используем все случаи, где обе модели выдали валидную категорию.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    # Наблюдаемое согласие
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    # Ожидаемое согласие по случайному распределению
    cnt_a = Counter(a for a, _ in pairs)
    cnt_b = Counter(b for _, b in pairs)
    p_e = sum((cnt_a[c] / n) * (cnt_b[c] / n) for c in CATEGORIES)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def kappa_band(k: float) -> str:
    """Вербальная шкала Landis & Koch (1977)."""
    if k < 0.0:
        return "хуже случайного"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def render_confusion_matrix(matrix: Dict[Tuple[str, str], int]) -> str:
    """Вывести 3×3 таблицу с маргинальными суммами."""
    lines = []
    lines.append("=== CONFUSION MATRIX (Haiku × GPT) ===")
    lines.append(
        f"  {'':<14} | "
        + " | ".join(f"GPT: {CAT_LABEL[c]} ({c[:10]})".ljust(20) for c in CATEGORIES)
        + " |   итого"
    )
    lines.append("  " + "-" * 92)
    for h in CATEGORIES:
        row = []
        row_total = 0
        for g in CATEGORIES:
            v = matrix.get((h, g), 0)
            row.append(f"{v:>20}")
            row_total += v
        lines.append(
            f"  Haiku: {CAT_LABEL[h]} ({h[:8]}) | "
            + " | ".join(row)
            + f" | {row_total:>6}"
        )
    lines.append("  " + "-" * 92)
    col_totals = [
        sum(matrix.get((h, g), 0) for h in CATEGORIES) for g in CATEGORIES
    ]
    lines.append(
        f"  {'итого':<14} | "
        + " | ".join(f"{t:>20}" for t in col_totals)
        + f" | {sum(col_totals):>6}"
    )
    return "\n".join(lines)


def render_cases(rows: List[Tuple[Dict, Dict]], header: str, n: int = 3) -> str:
    """Распечатать топ-N кейсов в человекочитаемом виде."""
    out = [f"=== {header} ===", ""]
    if not rows:
        out.append("(нет таких случаев)")
        return "\n".join(out)
    for i, (h, g) in enumerate(rows[:n], 1):
        out.append(
            f"Случай #{i}: sample_id={h['sample_id']} (kind={h['kind']})"
        )
        out.append(f"  Вопрос:           {h['question']}")
        out.append(f"  Ответ:            {h['answer']}")
        expected_label = "галлюцинация" if h["expected"] else "корректный"
        predicted_label = (
            "галлюцинация" if h.get("predicted") else "корректный"
        ) if h.get("predicted") is not None else "(не распознан)"
        out.append(
            f"  Метки:            HaluEval={expected_label}; "
            f"LLM-судья={predicted_label}"
        )
        explanation = h.get("explanation") or "(нет)"
        out.append(f"  LLM-обоснование:  {explanation}")
        out.append(
            f"  Haiku-вердикт:    {h.get('category')} — "
            f"{h.get('justification') or '(нет обоснования)'}"
        )
        out.append(
            f"  GPT-вердикт:      {g.get('category')} — "
            f"{g.get('justification') or '(нет обоснования)'}"
        )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--haiku", type=Path,
        default=Path("reports/disagreements_auto.json"),
    )
    parser.add_argument(
        "--gpt", type=Path,
        default=Path("reports/disagreements_auto_openai.json"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else project_root / p

    haiku_path = _resolve(args.haiku)
    gpt_path = _resolve(args.gpt)

    haiku_rows = _load_rows(haiku_path)
    gpt_rows = _load_rows(gpt_path)
    common_keys = sorted(set(haiku_rows) & set(gpt_rows))

    print(f"Сравниваем:")
    print(f"  Haiku:  {haiku_path}  ({len(haiku_rows)} строк)")
    print(f"  GPT:    {gpt_path}  ({len(gpt_rows)} строк)")
    print(f"  Общих ключей (sample_id/kind): {len(common_keys)}")
    print()

    # ── Confusion matrix ────────────────────────────────────────────────
    matrix: Dict[Tuple[str, str], int] = {}
    pairs: List[Tuple[str, str]] = []
    only_haiku_unparsed = 0
    only_gpt_unparsed = 0
    both_unparsed = 0

    for k in common_keys:
        h_cat = haiku_rows[k].get("category")
        g_cat = gpt_rows[k].get("category")
        if h_cat is None and g_cat is None:
            both_unparsed += 1
            continue
        if h_cat is None:
            only_haiku_unparsed += 1
            continue
        if g_cat is None:
            only_gpt_unparsed += 1
            continue
        matrix[(h_cat, g_cat)] = matrix.get((h_cat, g_cat), 0) + 1
        pairs.append((h_cat, g_cat))

    print(render_confusion_matrix(matrix))
    print()
    print(f"  Не распарсено только у Haiku: {only_haiku_unparsed}")
    print(f"  Не распарсено только у GPT:   {only_gpt_unparsed}")
    print(f"  Не распарсено у обеих:        {both_unparsed}")
    print()

    # ── Cohen's kappa ───────────────────────────────────────────────────
    k = cohen_kappa(pairs)
    n = len(pairs)
    p_o = (sum(matrix.get((c, c), 0) for c in CATEGORIES) / n) if n else 0.0
    print("=== Cohen's kappa ===")
    print(f"  N сравнимых пар: {n}")
    print(f"  Наблюдаемое согласие p_o = {p_o:.3f}")
    print(f"  κ = {k:.3f}  ({kappa_band(k)})")
    print()

    # ── Топ-3 случая Haiku=BAD_LABEL, GPT=MODEL_MISS ──────────────────────
    haiku_a_gpt_c = [
        (haiku_rows[k], gpt_rows[k])
        for k in common_keys
        if haiku_rows[k].get("category") == "BAD_LABEL"
        and gpt_rows[k].get("category") == "MODEL_MISS"
    ]
    print(
        f"=== Топ-3 случая Haiku=a (BAD_LABEL), GPT=c (MODEL_MISS) — "
        f"всего {len(haiku_a_gpt_c)} ==="
    )
    print(
        "(GPT диагностирует ошибку именно LLM-судьи там, где Haiku оправдывает разметку HaluEval)"
    )
    print()
    print(render_cases(haiku_a_gpt_c, "Возможные кейсы circular reasoning Haiku", n=3))
    print()

    # ── Топ-3 случая Haiku=BAD_LABEL, GPT=EVASIVE ─────────────────────────
    haiku_a_gpt_b = [
        (haiku_rows[k], gpt_rows[k])
        for k in common_keys
        if haiku_rows[k].get("category") == "BAD_LABEL"
        and gpt_rows[k].get("category") == "EVASIVE"
    ]
    print(
        f"=== Топ-3 случая Haiku=a (BAD_LABEL), GPT=b (EVASIVE) — "
        f"всего {len(haiku_a_gpt_b)} ==="
    )
    print(
        "(Haiku считает разметку ошибочной, GPT — что ответ просто уклончивый)"
    )
    print()
    print(render_cases(haiku_a_gpt_b, "Haiku агрессивно защищает решения LLM-судьи", n=3))
    print()

    # ── Топ-3 случая, где обе модели сошлись на BAD_LABEL ─────────────────
    both_bad = [
        (haiku_rows[k], gpt_rows[k])
        for k in common_keys
        if haiku_rows[k].get("category") == "BAD_LABEL"
        and gpt_rows[k].get("category") == "BAD_LABEL"
    ]
    print(
        f"=== Топ-3 случая, где обе модели согласны на BAD_LABEL — "
        f"всего {len(both_bad)} ==="
    )
    print(
        "(Самые твёрдые кандидаты на «реально ошибочная разметка HaluEval»)"
    )
    print()
    print(render_cases(both_bad, "Согласие обеих моделей: разметка HaluEval плоха", n=3))


if __name__ == "__main__":
    main()
