#!/usr/bin/env python3
"""Полный анализ ручной разметки 100 + 100 случаев.

Извлекает категории из manual_review_haiku_100.md и
manual_review_gpt_100.md, считает статистики, Wilson CI,
сопоставляет с автоклассификациями, находит парные sample_id.
"""

import json
import re
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
HAIKU_MD = ROOT / 'reports' / 'manual_review_haiku_100.md'
GPT_MD = ROOT / 'reports' / 'manual_review_gpt_100.md'


def parse_review_md(path):
    """Извлечь список случаев с их sample_id, kind и проставленной категорией."""
    text = path.read_text(encoding='utf-8')

    # Разбить на блоки по случаям
    blocks = re.split(r'\n## Случай №\d+', text)
    cases = []

    for block in blocks[1:]:  # первый блок — инструкции
        m_id = re.search(r'sample_id=(\d+)', block)
        m_kind = re.search(r'kind=(\w+)', block)
        # Категория: либо просто "a"/"b"/"c", либо "a (BAD_LABEL)" и т.п.
        m_cat = re.search(r'\*\*Моя категория:\*\*\s*([abc])\b', block)
        m_obs = re.search(
            r'\*\*Моё обоснование[^:]*:\*\*\s*\n*(.+?)(?:\n---|\n##|\Z)',
            block, re.DOTALL,
        )

        if not (m_id and m_kind):
            continue

        sample_id = int(m_id.group(1))
        kind = m_kind.group(1)
        category = m_cat.group(1).lower() if m_cat else None
        reasoning = m_obs.group(1).strip() if m_obs else ''

        cases.append({
            'sample_id': sample_id,
            'kind': kind,
            'category': category,
            'reasoning': reasoning,
        })

    return cases


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI для пропорции."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def summarize(cases, label):
    """Подсчитать a/b/c с Wilson CI."""
    n = len(cases)
    n_marked = sum(1 for c in cases if c['category'] is not None)
    cats = Counter(c['category'] for c in cases if c['category'])

    print(f"=== {label} ===")
    print(f"Всего случаев: {n}")
    print(f"Размечено: {n_marked} ({100 * n_marked / n:.1f}%)")
    if n_marked < n:
        print(f"⚠ Не размечено: {n - n_marked} случаев")
    print()

    for cat, name in [('a', 'BAD_LABEL'), ('b', 'EVASIVE'), ('c', 'MODEL_MISS')]:
        k = cats.get(cat, 0)
        if n_marked > 0:
            p = 100 * k / n_marked
            ci_lo, ci_hi = wilson_ci(k, n_marked)
            print(
                f"  {cat} ({name:11s}): {k:3d} = {p:5.1f}%  "
                f"95% Wilson CI: [{100 * ci_lo:.1f}%, {100 * ci_hi:.1f}%]"
            )
    print()
    return cats, n_marked


def find_paired_cases(haiku_cases, gpt_cases):
    """Найти случаи где один и тот же (sample_id, kind) есть в обоих файлах."""
    haiku_by_key = {(c['sample_id'], c['kind']): c for c in haiku_cases}
    gpt_by_key = {(c['sample_id'], c['kind']): c for c in gpt_cases}

    common_keys = set(haiku_by_key) & set(gpt_by_key)

    pairs = []
    for key in common_keys:
        h = haiku_by_key[key]
        g = gpt_by_key[key]
        pairs.append({
            'sample_id': key[0],
            'kind': key[1],
            'haiku_category': h['category'],
            'gpt_category': g['category'],
        })

    return pairs


def find_internal_pairs(cases):
    """Пары случаев в одном файле, где оба kind для одного sample_id."""
    by_sid = {}
    for c in cases:
        by_sid.setdefault(c['sample_id'], []).append(c)

    pairs = [v for v in by_sid.values() if len(v) == 2]
    return pairs


# Маппинг категорий из автоклассификаций (BAD_LABEL/EVASIVE/MODEL_MISS) → a/b/c
AUTO_CAT_MAP = {
    'BAD_LABEL': 'a',
    'EVASIVE': 'b',
    'MODEL_MISS': 'c',
}


def compare_with_auto(cases, auto_data, label):
    """Сравнить ручную разметку с автоклассификацией."""
    rows = auto_data.get('rows') or auto_data.get('details') or auto_data
    auto_by_key = {}
    for d in rows:
        key = (d['sample_id'], d.get('kind', ''))
        cat = d.get('category')
        if isinstance(cat, str):
            cat_short = AUTO_CAT_MAP.get(cat.upper(), cat.lower()[:1] if cat else None)
        else:
            cat_short = None
        auto_by_key[key] = cat_short

    matches = 0
    mismatches = 0
    matrix = Counter()  # (manual, auto) -> count
    not_found = 0

    for c in cases:
        if c['category'] is None:
            continue
        key = (c['sample_id'], c['kind'])
        auto_cat = auto_by_key.get(key)
        if auto_cat is None:
            not_found += 1
            continue
        manual_cat = c['category']
        matrix[(manual_cat, auto_cat)] += 1
        if manual_cat == auto_cat:
            matches += 1
        else:
            mismatches += 1

    total = matches + mismatches
    if total == 0:
        print(f"=== {label} ===")
        print(f"⚠ Нет пересечений (not_found={not_found})")
        print()
        return None

    agreement = matches / total
    print(f"=== Согласие ручной разметки с {label} ===")
    print(f"Полное согласие: {matches}/{total} ({100 * agreement:.1f}%)")
    print(f"Расхождений: {mismatches} ({100 * mismatches / total:.1f}%)")
    if not_found > 0:
        print(f"⚠ Не найдено в авто: {not_found}")
    print()
    print("Confusion matrix (ручной × авто):")
    print(f"            auto:a   auto:b   auto:c")
    for manual_cat in ['a', 'b', 'c']:
        row = []
        for auto_cat in ['a', 'b', 'c']:
            row.append(f"{matrix.get((manual_cat, auto_cat), 0):4d}")
        print(f"  manual:{manual_cat}  {'   '.join(row)}")
    print()

    return {
        'matches': matches,
        'total': total,
        'agreement': agreement,
        'matrix': {f"{m}_vs_{a}": v for (m, a), v in matrix.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# ОСНОВНОЕ ВЫПОЛНЕНИЕ
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("АНАЛИЗ РУЧНОЙ РАЗМЕТКИ 100+100 СЛУЧАЕВ")
print("=" * 70)
print()

haiku_cases = parse_review_md(HAIKU_MD)
gpt_cases = parse_review_md(GPT_MD)

# 1. Базовые статистики
haiku_cats, haiku_n = summarize(
    haiku_cases, "РУЧНАЯ РАЗМЕТКА: расхождения Haiku-арбитра"
)
gpt_cats, gpt_n = summarize(
    gpt_cases, "РУЧНАЯ РАЗМЕТКА: расхождения GPT-арбитра"
)

# 2. Скорректированная точность
print("=" * 70)
print("СКОРРЕКТИРОВАННАЯ ТОЧНОСТЬ (на основе ручной разметки)")
print("=" * 70)
print()

haiku_baseline = json.load(open(ROOT / 'reports' / 'llm_baseline_n1991.json'))
gpt_baseline = json.load(open(ROOT / 'reports' / 'llm_baseline_openai_n1991.json'))


def count_baseline(b):
    details = b.get('details') if isinstance(b, dict) else b
    n_total = len(details)
    n_disagree = sum(1 for d in details if d.get('agreement') is False)
    n_agree = n_total - n_disagree
    return n_total, n_agree, n_disagree


n_total_haiku, n_agree_haiku, n_disagree_haiku = count_baseline(haiku_baseline)
n_total_gpt, n_agree_gpt, n_disagree_gpt = count_baseline(gpt_baseline)

print(
    f"Haiku-арбитр: {n_total_haiku} проверок, {n_agree_haiku} совпадений, "
    f"{n_disagree_haiku} расхождений (raw acc = {n_agree_haiku / n_total_haiku:.4f})"
)
print(
    f"GPT-арбитр:   {n_total_gpt} проверок, {n_agree_gpt} совпадений, "
    f"{n_disagree_gpt} расхождений (raw acc = {n_agree_gpt / n_total_gpt:.4f})"
)
print()

corrected_results = {}
for arbiter, cats, n_disagree, n_total, n_agree in [
    ("Haiku-арбитр", haiku_cats, n_disagree_haiku, n_total_haiku, n_agree_haiku),
    ("GPT-арбитр", gpt_cats, n_disagree_gpt, n_total_gpt, n_agree_gpt),
]:
    n_marked = sum(cats.values())
    if n_marked == 0:
        continue

    # Доли категорий среди расхождений
    p_bad = cats.get('a', 0) / n_marked
    p_eva = cats.get('b', 0) / n_marked
    p_miss = cats.get('c', 0) / n_marked

    # Скорректированная точность: считаем только MODEL_MISS реальными ошибками
    corrected_errors = p_miss * n_disagree
    corrected_correct = n_total - corrected_errors
    corrected_acc = corrected_correct / n_total

    # Wilson CI для p_miss
    miss_count = cats.get('c', 0)
    ci_lo_miss, ci_hi_miss = wilson_ci(miss_count, n_marked)

    # CI для скорректированной точности (через CI для p_miss)
    corrected_acc_lo = (n_total - ci_hi_miss * n_disagree) / n_total
    corrected_acc_hi = (n_total - ci_lo_miss * n_disagree) / n_total

    # Также: точность если EVASIVE считать ошибками (компромиссная)
    p_real_errors = p_eva + p_miss
    real_errors = p_real_errors * n_disagree
    compromise_acc = (n_total - real_errors) / n_total

    print(f"{arbiter}:")
    print(f"  Raw accuracy:                          {n_agree / n_total:.4f}")
    print(f"  Скорр. точность (только MODEL_MISS):  {corrected_acc:.4f}")
    print(
        f"  95% Wilson CI:                         "
        f"[{corrected_acc_lo:.4f}, {corrected_acc_hi:.4f}]"
    )
    print(f"  Скорр. точность (b+c как ошибки):     {compromise_acc:.4f}")
    print(
        f"  Доля BAD_LABEL: {p_bad * 100:.1f}% от расхождений = "
        f"{p_bad * n_disagree / n_total * 100:.2f}% от всей выборки "
        f"({int(round(p_bad * n_disagree))} случаев)"
    )
    print(
        f"  Доля EVASIVE:   {p_eva * 100:.1f}% от расхождений = "
        f"{p_eva * n_disagree / n_total * 100:.2f}% от всей выборки "
        f"({int(round(p_eva * n_disagree))} случаев)"
    )
    print(
        f"  Доля MODEL_MISS: {p_miss * 100:.1f}% от расхождений = "
        f"{p_miss * n_disagree / n_total * 100:.2f}% от всей выборки "
        f"({int(round(p_miss * n_disagree))} случаев)"
    )
    print()

    corrected_results[arbiter] = {
        'raw_accuracy': n_agree / n_total,
        'corrected_accuracy_strict': corrected_acc,
        'corrected_acc_ci': (corrected_acc_lo, corrected_acc_hi),
        'corrected_accuracy_compromise': compromise_acc,
        'p_bad_label': p_bad,
        'p_evasive': p_eva,
        'p_model_miss': p_miss,
        'n_disagree_total': n_disagree,
        'n_total': n_total,
    }

# 3. Парные случаи внутри файлов
print("=" * 70)
print("ВНУТРЕННИЕ ПАРЫ (один sample_id, оба kind в одном файле)")
print("=" * 70)
print()

haiku_internal_pairs = find_internal_pairs(haiku_cases)
gpt_internal_pairs = find_internal_pairs(gpt_cases)

for pairs, label in [
    (haiku_internal_pairs, "Haiku"),
    (gpt_internal_pairs, "GPT"),
]:
    print(f"{label}-файл: {len(pairs)} пар")
    for pair in pairs[:5]:
        sid = pair[0]['sample_id']
        cats_str = ', '.join(f"{p['kind']}={p['category']}" for p in pair)
        print(f"  sample_id={sid}: {cats_str}")
    if len(pairs) > 5:
        print(f"  (всего {len(pairs)}, показаны первые 5)")
    print()

# 4. Перекрёстные пары
print("=" * 70)
print("ПЕРЕКРЁСТНЫЕ ПАРЫ (один (sample_id, kind) в обоих файлах)")
print("=" * 70)
print()

cross_pairs = find_paired_cases(haiku_cases, gpt_cases)
print(f"Найдено перекрёстных (sample_id, kind): {len(cross_pairs)}")
for pair in cross_pairs[:10]:
    sid = pair['sample_id']
    print(
        f"  sample_id={sid}, kind={pair['kind']}: "
        f"haiku={pair['haiku_category']}, gpt={pair['gpt_category']}"
    )
if len(cross_pairs) > 10:
    print(f"  (всего {len(cross_pairs)}, показаны первые 10)")
print()

# 5. Сравнение с автоклассификациями
print("=" * 70)
print("СРАВНЕНИЕ С АВТОКЛАССИФИКАТОРАМИ")
print("=" * 70)
print()

agreements = {}

agreements['haiku_manual_vs_haiku_classifier'] = compare_with_auto(
    haiku_cases,
    json.load(open(ROOT / 'reports' / 'disagreements_auto.json')),
    "Haiku-классификатором (ручная разметка vs Haiku про Haiku-арбитра)",
)

agreements['haiku_manual_vs_gpt_classifier'] = compare_with_auto(
    haiku_cases,
    json.load(open(ROOT / 'reports' / 'disagreements_auto_openai.json')),
    "GPT-классификатором (ручная разметка vs GPT про Haiku-арбитра)",
)

agreements['gpt_manual_vs_haiku_classifier'] = compare_with_auto(
    gpt_cases,
    json.load(
        open(ROOT / 'reports' / 'disagreements_gpt_classified_by_haiku.json')
    ),
    "Haiku-классификатором (ручная разметка vs Haiku про GPT-арбитра)",
)

agreements['gpt_manual_vs_gpt_classifier'] = compare_with_auto(
    gpt_cases,
    json.load(
        open(ROOT / 'reports' / 'disagreements_gpt_classified_by_gpt.json')
    ),
    "GPT-классификатором (ручная разметка vs GPT про GPT-арбитра)",
)

# 6. Топ-3 интересных случая каждой категории
print("=" * 70)
print("ТОП-3 ИНТЕРЕСНЫХ СЛУЧАЯ С ОБОСНОВАНИЯМИ (для главы 6.5)")
print("=" * 70)
print()

for cases, label in [(haiku_cases, "Haiku-арбитр"), (gpt_cases, "GPT-арбитр")]:
    print(f"--- {label} ---")
    for cat, name in [('a', 'BAD_LABEL'), ('b', 'EVASIVE'), ('c', 'MODEL_MISS')]:
        examples = [
            c for c in cases
            if c['category'] == cat and len(c['reasoning']) > 50
        ]
        examples = examples[:3]
        print(f"\n  Категория {cat} ({name}) — топ-3 случая:")
        for ex in examples:
            print(f"    sample_id={ex['sample_id']} (kind={ex['kind']})")
            obs = ex['reasoning'][:200].replace('\n', ' ')
            print(f"    Обоснование: {obs}")
    print()

# JSON-сводка
print("=" * 70)
print("СОХРАНЕНИЕ JSON-СВОДКИ")
print("=" * 70)

results = {
    'haiku_arbiter': {
        'n_marked': haiku_n,
        'distribution': dict(haiku_cats),
        'percentages': {
            cat: 100 * haiku_cats.get(cat, 0) / haiku_n
            for cat in ['a', 'b', 'c']
        } if haiku_n else {},
        'wilson_ci': {
            cat: list(wilson_ci(haiku_cats.get(cat, 0), haiku_n))
            for cat in ['a', 'b', 'c']
        } if haiku_n else {},
    },
    'gpt_arbiter': {
        'n_marked': gpt_n,
        'distribution': dict(gpt_cats),
        'percentages': {
            cat: 100 * gpt_cats.get(cat, 0) / gpt_n
            for cat in ['a', 'b', 'c']
        } if gpt_n else {},
        'wilson_ci': {
            cat: list(wilson_ci(gpt_cats.get(cat, 0), gpt_n))
            for cat in ['a', 'b', 'c']
        } if gpt_n else {},
    },
    'cross_pairs_count': len(cross_pairs),
    'haiku_internal_pairs': len(haiku_internal_pairs),
    'gpt_internal_pairs': len(gpt_internal_pairs),
    'corrected_accuracy': {
        k: {
            **v,
            'corrected_acc_ci': list(v['corrected_acc_ci']),
        }
        for k, v in corrected_results.items()
    },
    'agreements_with_auto_classifiers': agreements,
}

out_json = ROOT / 'reports' / 'manual_review_analysis.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"JSON сохранён: {out_json}")
print()
print("=" * 70)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 70)
