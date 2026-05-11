"""
Smoke-тест модуля FactChecker.

Берёт первые 3 примера из HaluEval, разбивает ``knowledge`` на
предложения-факты и для каждого ``right_answer`` / ``hallucinated_answer``
прогоняет три варианта верификации (NLI, embedding, combined-or).
Печатает результаты в виде таблицы для глазной проверки.

Запуск из корня проекта::

    python -m tests.test_fact_checker

На некоторых GPU (например, GTX 10xx, sm_61) текущая сборка PyTorch
заявляет ``cuda.is_available() == True``, но реальный forward падает.
Авто-выбор устройства в FactChecker откатится на CPU автоматически.
"""

from __future__ import annotations

import logging
from pathlib import Path

from halu_detect.data_loader import load_halueval
from halu_detect.fact_checker import FactChecker, Verdict
from halu_detect.utils import split_to_facts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    samples = load_halueval(project_root / "data" / "qa_data.json", limit=3)

    checker = FactChecker()

    header = (
        f"{'#':<2} {'kind':<12} {'method':<10} "
        f"{'verdict':<8} {'conf':>6}  claim"
    )
    print("\n" + header)
    print("-" * len(header))

    for i, sample in enumerate(samples, start=1):
        facts = split_to_facts(sample.knowledge)
        print(f"\n[{i}] facts: {len(facts)} | question: {sample.question}")

        for kind, answer in (
            ("right", sample.right_answer),
            ("hallucinated", sample.hallucinated_answer),
        ):
            v_nli = checker.check_nli(answer, facts)
            v_emb = checker.check_embedding(answer, facts)
            v_or = checker.check_combined(answer, facts, mode="or")

            for method, v in (("nli", v_nli), ("embedding", v_emb), ("comb-or", v_or)):
                verdict = "HALLU" if v.is_hallucination else "OK"
                print(
                    f"{i:<2} {kind:<12} {method:<10} "
                    f"{verdict:<8} {v.confidence:>6.3f}  {answer[:80]}"
                )

    # Дополнительно: пакетный режим на тех же claims, чтобы убедиться,
    # что check_batch проходит без ошибок и согласован с одиночным
    # вызовом combined.
    print("\n--- check_batch (combined / OR) ---")
    all_claims = []
    for sample in samples:
        all_claims.append(sample.right_answer)
        all_claims.append(sample.hallucinated_answer)
    facts_all = split_to_facts(samples[0].knowledge)  # KB первого примера
    batch_verdicts = checker.check_batch(all_claims, facts_all, method="combined", mode="or")
    for v in batch_verdicts:
        verdict = "HALLU" if v.is_hallucination else "OK"
        print(f"  {verdict:<6} conf={v.confidence:.3f}  {v.claim[:80]}")

    # Базовые инварианты структуры ответа — без assert на исход.
    for v in batch_verdicts:
        assert isinstance(v, Verdict)
        assert v.method == "combined"
        assert 0.0 <= v.confidence <= 1.0

    print("\nOK: smoke-тест FactChecker пройден.")


if __name__ == "__main__":
    main()
