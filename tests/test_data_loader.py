"""
Простейший smoke-тест модуля DataLoader.

Читает реальный датасет HaluEval из ``data/qa_data.json`` и выводит
первые три примера. Запускать из корня проекта:

    python -m tests.test_data_loader

Этот тест намеренно не использует pytest — он демонстрирует работу
модуля при отладке и пригоден как начальная проверка для главы
«Программа испытаний» в пояснительной записке.
"""

from __future__ import annotations

import logging
from pathlib import Path

from halu_detect.data_loader import load_halueval


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "qa_data.json"

    samples = load_halueval(dataset_path, limit=3)

    print(f"\nЗагружено примеров: {len(samples)}\n")
    for i, sample in enumerate(samples, start=1):
        print(f"=== Пример #{i} ===")
        print(f"  knowledge          : {sample.knowledge[:120]}{'...' if len(sample.knowledge) > 120 else ''}")
        print(f"  question           : {sample.question}")
        print(f"  right_answer       : {sample.right_answer}")
        print(f"  hallucinated_answer: {sample.hallucinated_answer}")
        print()

    assert len(samples) == 3, "Ожидалось ровно 3 примера"
    for s in samples:
        assert s.knowledge and s.question, "Поля knowledge/question не должны быть пустыми"
        assert s.right_answer and s.hallucinated_answer, "Ответы не должны быть пустыми"
        assert "  " not in s.knowledge, "В нормализованном тексте не должно быть двойных пробелов"

    print("OK: smoke-тест DataLoader пройден.")


if __name__ == "__main__":
    main()
