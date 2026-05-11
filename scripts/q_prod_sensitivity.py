#!/usr/bin/env python3
"""Анализ устойчивости порядка методов к выбору весов в Q_λ.

Q_λ(P, R) = λ·P + (1−λ)·R

Чем устойчивее порядок методов при разных λ ∈ {0,3; 0,4; 0,5; 0,6; 0,7},
тем менее произвольным является выбор λ = 0,5 в работе.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Точечные P, R из bootstrap_main_metrics.json (тестовая выборка n=3982).
# Грузим из файла, чтобы цифры всегда были согласованы с bootstrap-отчётом.
SOURCE = ROOT / "reports" / "bootstrap_main_metrics.json"

LAMBDAS: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_methods() -> List[Tuple[str, float, float]]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    out: List[Tuple[str, float, float]] = []
    for name, r in payload.items():
        p = r["point"]["precision"]
        rec = r["point"]["recall"]
        out.append((name, p, rec))
    return out


def main() -> None:
    methods = load_methods()

    print("=" * 86)
    print("Устойчивость порядка методов к выбору весов в Q_λ(P, R) = λ·P + (1−λ)·R")
    print("=" * 86)
    print()
    print(f"Источник P, R: {SOURCE.relative_to(ROOT)} (тест n=3982)")
    print()

    # Таблица значений Q_λ
    header = f"{'Метод':<26s} {'P':>7s} {'R':>7s}  " + \
             "  ".join(f"λ={l:.1f}" for l in LAMBDAS)
    print(header)
    print("-" * len(header))

    rows = {}
    for name, p, r in methods:
        q_vals = {l: round(l * p + (1 - l) * r, 4) for l in LAMBDAS}
        rows[name] = {"P": p, "R": r, "q_lambda": q_vals}
        print(f"{name:<26s} {p:>7.4f} {r:>7.4f}  " +
              "  ".join(f"{q_vals[l]:.4f}" for l in LAMBDAS))

    print()
    print("=" * 86)
    print("ПОРЯДОК МЕТОДОВ ПО УБЫВАНИЮ Q_λ ДЛЯ КАЖДОГО λ")
    print("=" * 86)

    orders_by_lambda = {}
    for l in LAMBDAS:
        sorted_methods = sorted(methods,
                                key=lambda x: -(l * x[1] + (1 - l) * x[2]))
        order = [name for name, _, _ in sorted_methods]
        orders_by_lambda[l] = order
        print(f"\nλ = {l:.1f}:")
        for rank, name in enumerate(order, 1):
            p, r = next((x[1], x[2]) for x in methods if x[0] == name)
            q = l * p + (1 - l) * r
            print(f"  {rank}. {name:<26s} Q_λ = {q:.4f}")

    print()
    print("=" * 86)
    print("СРАВНЕНИЕ ПОРЯДКОВ ПО ПАРАМ ЗНАЧЕНИЙ λ")
    print("=" * 86)
    base = orders_by_lambda[0.5]
    print(f"\nЭталон (λ=0.5): {' > '.join(n[:18] for n in base)}\n")

    same = True
    for l in LAMBDAS:
        if l == 0.5:
            continue
        cur = orders_by_lambda[l]
        if cur == base:
            print(f"  λ = {l:.1f}: ПОРЯДОК ИДЕНТИЧЕН λ=0.5")
        else:
            same = False
            # Найдём конкретные перестановки относительно λ=0.5
            differences = []
            for i, name in enumerate(cur):
                base_pos = base.index(name)
                if base_pos != i:
                    differences.append(f"{name} (был #{base_pos+1}, стал #{i+1})")
            print(f"  λ = {l:.1f}: ПОРЯДОК ИЗМЕНИЛСЯ. Перестановки:")
            for d in differences:
                print(f"      • {d}")

    print()
    if same:
        print("ВЫВОД: порядок методов УСТОЙЧИВ во всём диапазоне λ ∈ [0.3; 0.7].")
        print("       Выбор λ = 0.5 не влияет на ранжирование методов.")
    else:
        print("ВЫВОД: порядок методов ЗАВИСИТ от λ.")
        print("       Выбор λ существенно влияет на ранжирование некоторых пар методов.")
    print("=" * 86)

    out_path = ROOT / "reports" / "q_prod_sensitivity.json"
    out_path.write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "lambdas": LAMBDAS,
        "methods": [{"name": n, "P": p, "R": r} for n, p, r in methods],
        "values": rows,
        "orders_by_lambda": {str(l): orders_by_lambda[l] for l in LAMBDAS},
        "order_stable_in_range": same,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
