#!/usr/bin/env python3
"""Точный тест Фишера для независимости расхождений двух LLM-арбитров.

Проверяет гипотезу H0: множества расхождений Claude Haiku 4.5 и
GPT-5.4-mini с эталонной разметкой HaluEval статистически независимы.

Строит таблицу сопряжённости 2×2 на основе reports/disagreements_intersection.json:

                  GPT расходится  GPT согласен
Haiku расходится     intersection   h_only
Haiku согласен       g_only         (N − union)

где N = 3982 — общее число проверок на test-сплите (1991 пример × 2 проверки).

Воспроизводит результаты Утверждения 3 (подраздел 6.2 ВКР).

Запуск:

    python -m scripts.fisher_test
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2_contingency, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "disagreements_intersection.json"
OUT = ROOT / "reports" / "fisher_test.json"

N_TOTAL = 3982  # 1991 пример × 2 проверки на test-сплите


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))

    intersection = int(src["intersection"])
    h_only = int(src["h_only"])
    g_only = int(src["g_only"])
    both_agree = N_TOTAL - intersection - h_only - g_only

    if both_agree <= 0:
        raise ValueError(
            f"Некорректные данные: N − union = {both_agree}; "
            f"проверь {SRC.relative_to(ROOT)}"
        )

    table = [[intersection, h_only], [g_only, both_agree]]

    h_disagree = intersection + h_only
    g_disagree = intersection + g_only
    exp_intersection = h_disagree * g_disagree / N_TOTAL

    odds_ratio, p_fisher = fisher_exact(table, alternative="two-sided")
    chi2_stat, p_chi2, dof, exp_table = chi2_contingency(table)

    # ── Печать ──────────────────────────────────────────────────────────
    print("=== Точный тест Фишера для независимости расхождений арбитров ===\n")
    print(f"Источник:  {SRC.relative_to(ROOT)}")
    print(f"Всего проверок (N): {N_TOTAL}\n")

    print("Таблица сопряжённости 2×2:")
    print(f"                  GPT расходится  GPT согласен")
    print(f"  Haiku расходится     {intersection:>6d}        {h_only:>6d}")
    print(f"  Haiku согласен       {g_only:>6d}        {both_agree:>6d}")
    print(f"  (всего: {sum(sum(r) for r in table)})\n")

    print("Под гипотезой независимости:")
    print(f"  ожидаемое пересечение: {exp_intersection:.1f}")
    print(f"  наблюдаемое:           {intersection}")
    print(f"  ratio observed/expected: {intersection / exp_intersection:.2f}x\n")

    print("Fisher exact (two-sided):")
    print(f"  odds ratio = {odds_ratio:.4f}")
    if p_fisher > 0:
        print(f"  p-value    = {p_fisher:.3e}")
        print(f"  log10(p)   = {math.log10(p_fisher):.2f}")
    else:
        print("  p-value    = 0 (underflow ниже точности double)")

    print("\nChi² для перекрёстной проверки:")
    print(f"  chi2 = {chi2_stat:.2f},  dof = {dof}")
    if p_chi2 > 0:
        print(f"  p    = {p_chi2:.3e}")
    else:
        print("  p    = 0 (underflow)")

    print("\nВывод: гипотеза независимости отвергается с очень высокой "
          "доверительной вероятностью; расхождения двух арбитров"
          " статистически зависимы (Утверждение 3, подраздел 6.2 ВКР).\n")

    out = {
        "source": str(SRC.relative_to(ROOT)),
        "n_total": N_TOTAL,
        "contingency_table": {
            "haiku_disagrees_gpt_disagrees": intersection,
            "haiku_disagrees_gpt_agrees": h_only,
            "haiku_agrees_gpt_disagrees": g_only,
            "haiku_agrees_gpt_agrees": both_agree,
        },
        "marginals": {
            "haiku_disagree_total": h_disagree,
            "gpt_disagree_total": g_disagree,
            "expected_intersection_under_h0": round(exp_intersection, 2),
        },
        "fisher_exact": {
            "odds_ratio": odds_ratio,
            "p_value": p_fisher,
            "log10_p": math.log10(p_fisher) if p_fisher > 0 else None,
            "alternative": "two-sided",
        },
        "chi2_contingency": {
            "chi2": chi2_stat,
            "p_value": p_chi2,
            "log10_p": math.log10(p_chi2) if p_chi2 > 0 else None,
            "dof": dof,
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
