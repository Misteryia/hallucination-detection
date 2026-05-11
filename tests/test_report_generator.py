"""
Smoke-тест модуля ReportGenerator.

Проверяет, что:
  1. ``generate_evaluation_report`` создаёт валидный JSON и непустой HTML
     с ожидаемой структурой ключей и упоминанием Precision/Recall/Q_prod.
  2. ``generate_check_report`` создаёт JSON с исходным текстом и список
     вердиктов; HTML содержит подсвеченный фрагмент.

Запуск из корня проекта::

    python -m tests.test_report_generator
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import List

from halu_detect.aggregator import Aggregator, EvaluationResult
from halu_detect.fact_checker import Verdict
from halu_detect.report_generator import ReportGenerator


def _make_results() -> List[EvaluationResult]:
    """10 искусственных результатов: TP=4, FN=1, TN=4, FP=1 (как в test_aggregator)."""
    results: List[EvaluationResult] = []
    for i in range(4):
        results.append(EvaluationResult(
            sample_id=i, expected_hallucination=True, predicted_hallucination=True,
            confidence=0.9, method="nli", claim=f"hallu_tp_{i}",
            matched_fact=f"fact_for_hallu_{i}",
        ))
    results.append(EvaluationResult(
        sample_id=4, expected_hallucination=True, predicted_hallucination=False,
        confidence=0.6, method="nli", claim="missed hallucination claim",
        matched_fact="fact_for_missed",
    ))
    for i in range(4):
        results.append(EvaluationResult(
            sample_id=5 + i, expected_hallucination=False, predicted_hallucination=False,
            confidence=0.85, method="nli", claim=f"right_tn_{i}",
            matched_fact=f"fact_for_right_{i}",
        ))
    results.append(EvaluationResult(
        sample_id=9, expected_hallucination=False, predicted_hallucination=True,
        confidence=0.72, method="nli", claim="false positive claim",
        matched_fact="fact_for_fp",
    ))
    return results


def test_evaluation_report(tmp_dir: Path) -> None:
    agg = Aggregator()
    agg.add_results(_make_results())

    rg = ReportGenerator(output_dir=tmp_dir)
    json_path, html_path = rg.generate_evaluation_report(
        aggregator=agg,
        method="nli",
        dataset_name="HaluEval",
        n_examples=10,
        output_name="eval_test",
    )

    assert json_path.is_file(), "JSON должен быть создан"
    assert html_path.is_file(), "HTML должен быть создан"
    assert json_path.stat().st_size > 0
    assert html_path.stat().st_size > 0

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    for key in ("metadata", "metrics", "errors"):
        assert key in data, f"в JSON нет ключа {key!r}"
    assert data["metadata"]["method"] == "nli"
    assert data["metadata"]["dataset"] == "HaluEval"
    assert data["metadata"]["n_examples"] == 10
    assert data["metrics"]["precision"] == 0.8
    assert data["metrics"]["recall"] == 0.8
    assert data["metrics"]["confusion_matrix"]["tp"] == 4
    assert len(data["errors"]["false_positives"]) == 1
    assert data["errors"]["false_positives"][0]["claim"] == "false positive claim"
    assert len(data["errors"]["false_negatives"]) == 1

    html = html_path.read_text(encoding="utf-8")
    for keyword in ("Precision", "Recall", "Q", "TP", "FP", "FN", "TN"):
        assert keyword in html, f"в HTML нет ключевого слова {keyword!r}"
    assert "false positive claim" in html
    assert "missed hallucination claim" in html
    assert "halu_detect v" in html  # footer с версией

    print(f"  evaluation_report: {json_path}, {html_path}")


def test_check_report(tmp_dir: Path) -> None:
    text = (
        "Arthur's Magazine was started in 1844. "
        "First for Women was started first."
    )
    verdicts = [
        Verdict(
            claim="Arthur's Magazine was started in 1844.",
            method="nli",
            is_hallucination=False,
            confidence=0.95,
            matched_fact="Arthur's Magazine (1844–1846) was an American literary periodical.",
            details={"max_contradiction_prob": 0.05},
        ),
        Verdict(
            claim="First for Women was started first.",
            method="nli",
            is_hallucination=True,
            confidence=0.88,
            matched_fact="Arthur's Magazine (1844–1846) was published before First for Women.",
            details={"max_contradiction_prob": 0.88},
        ),
    ]

    rg = ReportGenerator(output_dir=tmp_dir)
    json_path, html_path = rg.generate_check_report(
        text=text,
        verdicts=verdicts,
        method="nli",
        output_name="check_test",
    )

    assert json_path.is_file()
    assert html_path.is_file()

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["text"] == text
    assert data["metadata"]["method"] == "nli"
    assert data["metadata"]["n_verdicts"] == 2
    assert data["summary"] == {"total": 2, "hallucinations": 1, "ok": 1}
    assert len(data["verdicts"]) == 2
    assert data["verdicts"][0]["claim"] == verdicts[0].claim
    assert data["verdicts"][1]["is_hallucination"] is True

    html = html_path.read_text(encoding="utf-8")
    assert "Arthur&#39;s Magazine was started in 1844." in html or \
           "Arthur's Magazine was started in 1844." in html
    assert "First for Women was started first." in html
    # Подсветочные классы
    assert 'class="hallu"' in html
    assert 'class="ok"' in html
    # Tooltip с matched_fact
    assert "Matched fact:" in html
    # Метод и версия
    assert "<code>nli</code>" in html
    assert "halu_detect v" in html

    print(f"  check_report: {json_path}, {html_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with tempfile.TemporaryDirectory(prefix="halu_test_") as tmp:
        tmp_dir = Path(tmp)
        print("\nTemp dir:", tmp_dir)

        test_evaluation_report(tmp_dir)
        test_check_report(tmp_dir)

    print("\nOK: smoke-тесты ReportGenerator пройдены.")


if __name__ == "__main__":
    main()
