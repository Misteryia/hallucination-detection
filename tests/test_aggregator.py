"""
Smoke-тест модуля Aggregator.

Использует исключительно искусственные данные — реальные ML-модели
не подгружаются. Цель — проверить корректность арифметики (TP/FP/TN/FN,
Precision/Recall/F1, accuracy, Q_prod), классификацию ошибок и
устойчивость к делению на ноль.

Запуск из корня проекта::

    python -m tests.test_aggregator
"""

from __future__ import annotations

import logging
from typing import List

from halu_detect.aggregator import Aggregator, EvaluationResult, Metrics


def _make_results() -> List[EvaluationResult]:
    """Сконструировать 10 искусственных результатов с известным исходом.

    Конфигурация: 5 hallucinated (TP=4, FN=1) + 5 right (TN=4, FP=1).
    Ожидаемые метрики: P=R=F1=Acc=Q_prod=0.8.
    """
    results: List[EvaluationResult] = []

    # 5 hallucinated → ожидаем галлюцинацию.
    # 4 предсказаны правильно (TP).
    for i in range(4):
        results.append(EvaluationResult(
            sample_id=i,
            expected_hallucination=True,
            predicted_hallucination=True,
            confidence=0.9,
            method="nli",
            claim=f"hallu_tp_{i}",
            matched_fact=f"fact_for_hallu_{i}",
        ))
    # 1 пропущена (FN).
    results.append(EvaluationResult(
        sample_id=4,
        expected_hallucination=True,
        predicted_hallucination=False,
        confidence=0.55,
        method="nli",
        claim="hallu_fn",
        matched_fact="fact_for_fn",
    ))

    # 5 right → ожидаем НЕ галлюцинацию.
    # 4 предсказаны правильно (TN).
    for i in range(4):
        results.append(EvaluationResult(
            sample_id=5 + i,
            expected_hallucination=False,
            predicted_hallucination=False,
            confidence=0.85,
            method="nli",
            claim=f"right_tn_{i}",
            matched_fact=f"fact_for_right_{i}",
        ))
    # 1 ложное срабатывание (FP).
    results.append(EvaluationResult(
        sample_id=9,
        expected_hallucination=False,
        predicted_hallucination=True,
        confidence=0.7,
        method="nli",
        claim="right_fp",
        matched_fact="fact_for_fp",
    ))

    return results


def _print_metrics(label: str, m: Metrics) -> None:
    print(
        f"{label}: TP={m.true_positive} FP={m.false_positive} "
        f"TN={m.true_negative} FN={m.false_negative} | "
        f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
        f"Acc={m.accuracy:.3f} Q_prod={m.q_prod:.3f} | "
        f"passes={m.passes_quality_threshold}"
    )


def test_basic_metrics() -> None:
    agg = Aggregator()
    agg.add_results(_make_results())
    m = agg.compute_metrics()

    _print_metrics("[all]", m)

    assert m.n_samples == 10
    assert m.true_positive == 4
    assert m.false_positive == 1
    assert m.true_negative == 4
    assert m.false_negative == 1

    # P = TP / (TP+FP) = 4/5 = 0.8
    assert abs(m.precision - 0.8) < 1e-9
    # R = TP / (TP+FN) = 4/5 = 0.8
    assert abs(m.recall - 0.8) < 1e-9
    # F1 = 2PR / (P+R) = 2·0.8·0.8 / 1.6 = 0.8
    assert abs(m.f1 - 0.8) < 1e-9
    # Acc = (TP+TN) / total = 8/10 = 0.8
    assert abs(m.accuracy - 0.8) < 1e-9
    # Q_prod = 0.5·P + 0.5·R = 0.8
    assert abs(m.q_prod - 0.8) < 1e-9

    # 0.8 > 0.775 → проходит порог
    assert m.passes_quality_threshold is True


def test_per_method() -> None:
    agg = Aggregator()
    agg.add_results(_make_results())
    per_method = agg.compute_metrics_per_method()

    assert set(per_method) == {"nli"}
    assert abs(per_method["nli"].q_prod - 0.8) < 1e-9


def test_get_errors() -> None:
    agg = Aggregator()
    agg.add_results(_make_results())
    errors = agg.get_errors()

    assert len(errors["false_positives"]) == 1
    assert errors["false_positives"][0].claim == "right_fp"

    assert len(errors["false_negatives"]) == 1
    assert errors["false_negatives"][0].claim == "hallu_fn"


def test_empty_aggregator() -> None:
    """Пустой накопитель не должен падать на делении на ноль."""
    agg = Aggregator()
    m = agg.compute_metrics()

    _print_metrics("[empty]", m)

    assert m.n_samples == 0
    assert m.true_positive == m.false_positive == 0
    assert m.true_negative == m.false_negative == 0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.accuracy == 0.0
    assert m.q_prod == 0.0
    assert m.passes_quality_threshold is False


def test_reset() -> None:
    agg = Aggregator()
    agg.add_results(_make_results())
    assert agg.compute_metrics().n_samples == 10
    agg.reset()
    assert agg.compute_metrics().n_samples == 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_basic_metrics()
    test_per_method()
    test_get_errors()
    test_empty_aggregator()
    test_reset()

    print("\nOK: все smoke-тесты Aggregator пройдены.")


if __name__ == "__main__":
    main()
