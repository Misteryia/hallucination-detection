"""
Подготовка датасета HaluEval к финальной оценке.

Делает две вещи:

1. **Sanity-check** загруженного датасета:
   - распределение количества фактов на пример (после ``split_to_facts``);
   - примеры с подозрительной разбивкой (Dr., Mt., U.S., Inc. и т. п.);
   - дубликаты по ``knowledge`` и по ``(question, right_answer)``;
   - распределение длин полей.

2. **Детерминированный split** на pilot / train / test:
   - ``pilot`` — первые N примеров (по умолчанию 50). Эти примеры
     уже видела grid-search-эвристика для подбора порогов; чтобы
     избежать data leakage, в финальные метрики они не входят.
   - ``train`` и ``test`` берутся из остатка с фиксированным seed.
     ``train`` — для перепроверки/донастройки порогов; ``test`` —
     для финальных метрик защиты.

Результаты:
   * ``reports/data_sanity.json`` — машиночитаемый отчёт sanity-check;
   * ``reports/data_sanity.txt`` — то же текстом для глаз;
   * ``splits/halueval_split.json`` — ID примеров для каждого слоя.
     Этот файл коммитится в репозиторий — split воспроизводим.

Запуск:
    python -m scripts.prepare_evaluation
    python -m scripts.prepare_evaluation --test-ratio 0.2 --seed 42
    python -m scripts.prepare_evaluation --pilot-size 50
    python -m scripts.prepare_evaluation --check-only   # без создания split
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from halu_detect.data_loader import HaluEvalSample, load_halueval
from halu_detect.utils import split_to_facts

logger = logging.getLogger(__name__)


# Префиксы, которые при наивном split-е по точкам с большой вероятностью
# означают «здесь сегментатор ошибся, начало фрагмента — это хвост
# сокращения предыдущего предложения».
SUSPICIOUS_PREFIXES = [
    # титулы и имена
    "Smith ", "Jones ", "Brown ", "Wilson ", "Johnson ",
    # географические сокращения
    "S. ", "K. ", "A. ", "Everest", "Fuji", "Kilimanjaro",
    # юридические формы
    "in 2", "in 1", "in 19", "in 20", "Inc ", "Ltd ", "Co ",
]

# Регекс, ловящий: "X." на конце предыдущей строки и заглавную букву в начале новой.
# Используется для эвристического подсчёта подозрительных пар.
ABBREV_RE = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|St|Sr|Jr|Mt|Mts|U\.S|U\.K|N\.Y|D\.C|"
    r"Ph\.D|M\.D|Inc|Ltd|Co|Corp|No|vol|pp|p|ed|eds|et al|cf|etc|i\.e|e\.g)\.\s+[A-Z]"
)


# ─────────────────────────────────────────────────────────────────────────
# Sanity-check
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class SanityReport:
    n_total: int
    n_empty_knowledge: int
    n_zero_facts: int
    n_one_fact: int
    facts_per_sample: List[int] = field(default_factory=list)
    knowledge_lengths: List[int] = field(default_factory=list)
    answer_lengths: List[int] = field(default_factory=list)
    suspicious_split_ids: List[int] = field(default_factory=list)
    duplicate_knowledge_groups: List[List[int]] = field(default_factory=list)
    duplicate_qa_groups: List[List[int]] = field(default_factory=list)

    def summary_dict(self) -> Dict:
        fps = self.facts_per_sample or [0]
        kls = self.knowledge_lengths or [0]
        return {
            "n_total": self.n_total,
            "n_empty_knowledge": self.n_empty_knowledge,
            "n_zero_facts": self.n_zero_facts,
            "n_one_fact": self.n_one_fact,
            "n_suspicious_split": len(self.suspicious_split_ids),
            "n_duplicate_knowledge_groups": len(self.duplicate_knowledge_groups),
            "n_duplicate_qa_groups": len(self.duplicate_qa_groups),
            "facts_per_sample": {
                "min": min(fps),
                "max": max(fps),
                "mean": round(statistics.mean(fps), 2),
                "median": statistics.median(fps),
                "p10": _percentile(fps, 10),
                "p90": _percentile(fps, 90),
            },
            "knowledge_chars": {
                "min": min(kls),
                "max": max(kls),
                "mean": round(statistics.mean(kls), 1),
                "median": statistics.median(kls),
            },
        }


def _percentile(values: List[int], p: int) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def _is_suspicious_split(facts: List[str], original_text: str) -> bool:
    """Эвристика: видны ли явные следы поломанного сегментатора.

    Признаки:
      1. В исходном тексте найдена пара «сокращение + заглавная»
         (например, ``"Dr. Smith"``), которую наивный regex почти
         наверняка разрезал.
      2. Среди полученных фактов есть подозрительно короткие фрагменты
         (5–15 символов), начинающиеся с заглавной — типичный артефакт
         обрыва на сокращении.
    """
    if ABBREV_RE.search(original_text):
        return True
    for f in facts:
        if 5 <= len(f) <= 15 and f[0].isupper() and " " not in f.rstrip("."):
            return True
    return False


def run_sanity_check(samples: List[HaluEvalSample]) -> SanityReport:
    rep = SanityReport(
        n_total=len(samples),
        n_empty_knowledge=0,
        n_zero_facts=0,
        n_one_fact=0,
    )

    knowledge_to_ids: Dict[str, List[int]] = {}
    qa_to_ids: Dict[Tuple[str, str], List[int]] = {}

    for i, s in enumerate(samples):
        rep.knowledge_lengths.append(len(s.knowledge))
        rep.answer_lengths.append(len(s.right_answer))
        rep.answer_lengths.append(len(s.hallucinated_answer))

        if not s.knowledge.strip():
            rep.n_empty_knowledge += 1

        facts = split_to_facts(s.knowledge)
        rep.facts_per_sample.append(len(facts))
        if len(facts) == 0:
            rep.n_zero_facts += 1
        elif len(facts) == 1:
            rep.n_one_fact += 1

        if _is_suspicious_split(facts, s.knowledge):
            rep.suspicious_split_ids.append(i)

        knowledge_to_ids.setdefault(s.knowledge, []).append(i)
        qa_to_ids.setdefault((s.question, s.right_answer), []).append(i)

    rep.duplicate_knowledge_groups = [
        ids for ids in knowledge_to_ids.values() if len(ids) > 1
    ]
    rep.duplicate_qa_groups = [
        ids for ids in qa_to_ids.values() if len(ids) > 1
    ]

    return rep


def render_sanity_text(
    rep: SanityReport, samples: List[HaluEvalSample], n_examples: int = 5
) -> str:
    out = []
    s = rep.summary_dict()

    out.append("=" * 72)
    out.append("SANITY CHECK — HaluEval QA dataset")
    out.append("=" * 72)
    out.append(f"Всего примеров: {rep.n_total}")
    out.append("")
    out.append("Факты после split_to_facts (knowledge → предложения):")
    fps = s["facts_per_sample"]
    out.append(
        f"  min={fps['min']}  p10={fps['p10']}  median={fps['median']}  "
        f"mean={fps['mean']}  p90={fps['p90']}  max={fps['max']}"
    )
    out.append(f"  Примеров с пустым knowledge:    {rep.n_empty_knowledge}")
    out.append(f"  Примеров с 0 фактов:            {rep.n_zero_facts}")
    out.append(f"  Примеров с 1 фактом:            {rep.n_one_fact}")
    if rep.n_zero_facts:
        out.append(
            "  ⚠ Примеры с 0 фактами получают пустую базу знаний → "
            "_empty_kb_verdict помечает их галлюцинациями с confidence=1.0."
        )
    out.append("")
    out.append(
        f"Длина knowledge (символов): "
        f"min={s['knowledge_chars']['min']}, median={s['knowledge_chars']['median']}, "
        f"mean={s['knowledge_chars']['mean']}, max={s['knowledge_chars']['max']}"
    )
    out.append("")

    out.append("-" * 72)
    out.append(
        f"Подозрительная разбивка предложений: {len(rep.suspicious_split_ids)} "
        f"примеров ({100*len(rep.suspicious_split_ids)/max(rep.n_total,1):.1f}%)"
    )
    out.append("-" * 72)
    if rep.suspicious_split_ids:
        out.append("Признаки: 'Dr.', 'U.S.', 'Mr.', 'Inc.' и т. п. ломают наивный regex.")
        out.append("Примеры:")
        for sid in rep.suspicious_split_ids[:n_examples]:
            sample = samples[sid]
            facts = split_to_facts(sample.knowledge)
            out.append(f"  [id={sid}] knowledge: {sample.knowledge}")
            out.append(f"           разбивка:  {facts}")
        if len(rep.suspicious_split_ids) > n_examples:
            out.append(
                f"  ... и ещё {len(rep.suspicious_split_ids) - n_examples} "
                f"(полный список в JSON-отчёте)"
            )
    out.append("")

    out.append("-" * 72)
    out.append(
        f"Дубликаты по knowledge: {len(rep.duplicate_knowledge_groups)} групп"
    )
    out.append("-" * 72)
    for grp in rep.duplicate_knowledge_groups[:n_examples]:
        out.append(f"  ids={grp}: {samples[grp[0]].knowledge[:80]}...")
    out.append("")

    out.append("-" * 72)
    out.append(
        f"Дубликаты по (question, right_answer): "
        f"{len(rep.duplicate_qa_groups)} групп"
    )
    out.append("-" * 72)
    for grp in rep.duplicate_qa_groups[:n_examples]:
        out.append(f"  ids={grp}: {samples[grp[0]].question[:80]}")
    out.append("")

    out.append("=" * 72)
    out.append("РЕКОМЕНДАЦИИ")
    out.append("=" * 72)
    if len(rep.suspicious_split_ids) / max(rep.n_total, 1) > 0.05:
        out.append(
            "✗ Подозрительная разбивка > 5%. Рекомендую заменить split_to_facts на "
            "nltk.tokenize.sent_tokenize или spacy — наивный regex даёт шум."
        )
    else:
        out.append("✓ Разбивка предложений в норме.")
    if rep.n_zero_facts:
        out.append(
            f"✗ {rep.n_zero_facts} примеров с 0 фактов будут автоматически "
            "помечены как галлюцинации (фолбэк _empty_kb_verdict). Это завысит "
            "TP/FP. Решение: исключить их из выборки или специально обработать."
        )
    if rep.duplicate_knowledge_groups:
        out.append(
            f"⚠ {len(rep.duplicate_knowledge_groups)} групп дубликатов knowledge. "
            "При случайном split-е связанные дубликаты могут разбежаться по "
            "train и test, давая утечку информации. Обычно мало; если много — "
            "стоит сплитить по уникальному knowledge, а не по индексу."
        )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
# Сплит
# ─────────────────────────────────────────────────────────────────────────


def _build_knowledge_groups(samples: List[HaluEvalSample]) -> List[List[int]]:
    """Сгруппировать sample_id по одинаковому ``knowledge``.

    Каждая группа — список ID примеров с дословно совпадающим knowledge.
    Уникальные примеры образуют группу размера 1. Возвращаемые группы
    отсортированы по минимальному ID для детерминированности.
    """
    by_knowledge: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_knowledge.setdefault(s.knowledge, []).append(i)
    groups = list(by_knowledge.values())
    groups.sort(key=lambda g: min(g))
    return groups


def make_split(
    samples: List[HaluEvalSample],
    pilot_size: int = 50,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Детерминированный group-aware split.

    Логика:
      - первые ``pilot_size`` ID -> ``pilot`` (исключаются из финального теста);
      - оставшиеся ID группируются по совпадению ``knowledge``: дубликаты
        попадают **в один и тот же слой**, чтобы не было утечки между
        train и test через общий контекст;
      - группы перетасовываются ``random.Random(seed)`` и жадно
        распределяются: пока сумма размеров test-групп < target —
        очередная группа уходит в test, остальные в train.

    Сплит делается на уровне ``sample_id``, но единицей разбиения служит
    ГРУППА (knowledge) — чтобы повторяющийся контекст не оказался в
    обоих слоях. Test-доля приблизительна (зависит от размеров групп).
    """
    n_total = len(samples)
    if pilot_size > n_total:
        raise ValueError(
            f"pilot_size={pilot_size} > n_total={n_total}: нечего делить"
        )

    pilot = list(range(min(pilot_size, n_total)))
    pilot_set = set(pilot)

    # Все группы, целиком лежащие вне pilot. Если группа частично в pilot,
    # её non-pilot часть отбрасывается из train/test (чтобы пилотные
    # "дубликаты" не попали обратно в финальный тест через бэкдор).
    all_groups = _build_knowledge_groups(samples)
    eligible_groups: List[List[int]] = []
    discarded_due_to_pilot: List[int] = []
    for g in all_groups:
        if any(sid in pilot_set for sid in g):
            discarded_due_to_pilot.extend(sid for sid in g if sid not in pilot_set)
            continue
        eligible_groups.append(sorted(g))

    rng = random.Random(seed)
    rng.shuffle(eligible_groups)

    n_eligible = sum(len(g) for g in eligible_groups)
    target_test = int(round(n_eligible * test_ratio))

    test_ids: List[int] = []
    train_ids: List[int] = []
    for g in eligible_groups:
        if len(test_ids) < target_test:
            test_ids.extend(g)
        else:
            train_ids.extend(g)

    test_ids.sort()
    train_ids.sort()

    return {
        "metadata": {
            "seed": seed,
            "pilot_size": pilot_size,
            "test_ratio": test_ratio,
            "n_total": n_total,
            "n_pilot": len(pilot),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_discarded_due_to_pilot_overlap": len(discarded_due_to_pilot),
            "n_unique_groups": len(all_groups),
            "n_groups_with_dupes": sum(1 for g in all_groups if len(g) > 1),
            "split_strategy": "group_aware_by_knowledge",
        },
        "pilot": pilot,
        "train": train_ids,
        "test": test_ids,
        "discarded_due_to_pilot_overlap": sorted(discarded_due_to_pilot),
    }


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/qa_data.json"),
        help="Путь к qa_data.json (по умолчанию: data/qa_data.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-size", type=int, default=50)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число загруженных примеров (для отладки)",
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Только sanity-check, без создания split")
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports"),
        help="Куда складывать data_sanity.{json,txt}",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("splits"),
        help="Куда складывать halueval_split.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    data_path = args.data if args.data.is_absolute() else project_root / args.data
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else project_root / args.reports_dir
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else project_root / args.splits_dir

    samples = load_halueval(data_path, limit=args.limit)
    print(f"Загружено: {len(samples)} примеров из {data_path}\n")

    # 1. Sanity check
    rep = run_sanity_check(samples)
    text = render_sanity_text(rep, samples)
    print(text)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_sanity.txt").write_text(text, encoding="utf-8")

    sanity_json = {
        "summary": rep.summary_dict(),
        "suspicious_split_ids": rep.suspicious_split_ids,
        "duplicate_knowledge_groups": rep.duplicate_knowledge_groups,
        "duplicate_qa_groups": rep.duplicate_qa_groups,
    }
    (reports_dir / "data_sanity.json").write_text(
        json.dumps(sanity_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nОтчёт сохранён: {reports_dir / 'data_sanity.txt'}")
    print(f"               {reports_dir / 'data_sanity.json'}")

    # 2. Split
    if args.check_only:
        return

    split = make_split(
        samples=samples,
        pilot_size=args.pilot_size,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / "halueval_split.json"
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = split["metadata"]
    print()
    print("=" * 72)
    print("SPLIT (group-aware: дубликаты knowledge не разбегаются)")
    print("=" * 72)
    print(f"  pilot:     {md['n_pilot']:>5d} примеров (id 0..{md['pilot_size']-1})")
    print(f"  train:     {md['n_train']:>5d} примеров (для подбора порогов)")
    print(f"  test:      {md['n_test']:>5d} примеров (для финальных метрик)")
    print(f"  отброшено: {md['n_discarded_due_to_pilot_overlap']:>5d} "
          f"(дубликаты пилотных knowledge)")
    print(f"  seed: {md['seed']}, test_ratio: {md['test_ratio']}")
    print(f"  групп всего: {md['n_unique_groups']}, "
          f"с дубликатами: {md['n_groups_with_dupes']}")
    print(f"\nСплит сохранён: {split_path}")
    print(
        "\nВажно: закоммить splits/halueval_split.json в репозиторий — "
        "это гарантирует воспроизводимость финальных метрик."
    )


if __name__ == "__main__":
    main()