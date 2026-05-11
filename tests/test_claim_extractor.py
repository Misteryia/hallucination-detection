"""
Smoke-тест модуля ClaimExtractor.

Берёт ``hallucinated_answer`` первого примера датасета HaluEval,
прогоняет через экстрактор и выводит извлечённые claims с указанием
позиций. Запуск из корня проекта::

    python -m tests.test_claim_extractor
"""

from __future__ import annotations

import logging
from pathlib import Path

from halu_detect.claim_extractor import ClaimExtractor
from halu_detect.data_loader import load_halueval


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "qa_data.json"

    samples = load_halueval(dataset_path, limit=1)
    assert samples, "Не удалось загрузить ни одного примера из HaluEval"
    sample = samples[0]

    print("\n--- Hallucinated answer ---")
    print(sample.hallucinated_answer)

    extractor = ClaimExtractor()
    claims = extractor.extract(sample.hallucinated_answer)

    print(f"\nИзвлечено claims: {len(claims)}\n")
    for c in claims:
        print(f"  [pos={c.position}] {c.text}")

    for c in claims:
        assert c.text and not c.text.endswith("?"), "Вопросы должны быть отфильтрованы"
        assert isinstance(c.position, int) and c.position >= 0, "position — неотрицательное целое"

    print("\nOK: smoke-тест ClaimExtractor пройден.")


if __name__ == "__main__":
    main()
