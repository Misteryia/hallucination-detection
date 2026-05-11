"""
Модуль evaluation.

Связывает :class:`FactChecker` и :class:`Aggregator` в единый сценарий
оценки на размеченном датасете HaluEval. Помещён в отдельный файл,
чтобы :mod:`halu_detect.aggregator` оставался чисто математическим и
тестировался без подгрузки тяжёлых ML-зависимостей.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from tqdm import tqdm

from .aggregator import Aggregator, EvaluationResult
from .data_loader import HaluEvalSample
from .fact_checker import FactChecker, Verdict
from .utils import split_to_facts

logger = logging.getLogger(__name__)


def _check(
    fact_checker: FactChecker,
    claim: str,
    facts: List[str],
    method: str,
    mode: str,
) -> Verdict:
    """Дёрнуть нужный метод верификации по строковому имени."""
    if method == "nli":
        return fact_checker.check_nli(claim, facts)
    if method == "embedding":
        return fact_checker.check_embedding(claim, facts)
    if method == "combined":
        return fact_checker.check_combined(claim, facts, mode=mode)
    raise ValueError(f"Неизвестный метод: {method!r}")


def evaluate_on_halueval(
    samples: List[HaluEvalSample],
    fact_checker: FactChecker,
    method: str = "combined",
    mode: str = "or",
    limit: Optional[int] = None,
) -> Tuple[Aggregator, List[EvaluationResult]]:
    """Прогнать :class:`FactChecker` по датасету HaluEval с авторазметкой.

    Для каждого примера ``HaluEvalSample`` поле ``knowledge``
    разбивается на список фактов-предложений; затем оба ответа
    (``right_answer`` и ``hallucinated_answer``) проверяются выбранным
    методом и записываются в накопитель. Эталонная метка ставится
    автоматически: ``right_answer`` → ``expected_hallucination=False``,
    ``hallucinated_answer`` → ``expected_hallucination=True``.

    Параметры:
        samples: список загруженных :class:`HaluEvalSample`.
        fact_checker: уже сконструированный :class:`FactChecker`.
        method: ``"nli"``, ``"embedding"`` или ``"combined"``.
        mode: режим объединения для ``method="combined"``: ``"or"`` —
            recall-ориентир, ``"and"`` — precision-ориентир.
        limit: ограничение по числу примеров (``None`` — все).

    Возвращает:
        Кортеж ``(aggregator, results)``: заполненный
        :class:`Aggregator` и плоский список всех
        :class:`EvaluationResult` (по два на пример).
    """
    if method not in ("nli", "embedding", "combined"):
        raise ValueError(f"Неизвестный метод: {method!r}")

    if limit is not None:
        samples = samples[:limit]

    logger.info(
        "Оценка на HaluEval: %d примеров, метод=%s, режим=%s",
        len(samples), method, mode if method == "combined" else "—",
    )

    aggregator = Aggregator()
    results: List[EvaluationResult] = []

    for sample_id, sample in enumerate(tqdm(samples, desc=f"Eval[{method}]")):
        facts = split_to_facts(sample.knowledge)

        for expected, claim in (
            (False, sample.right_answer),
            (True, sample.hallucinated_answer),
        ):
            verdict = _check(fact_checker, claim, facts, method, mode)
            results.append(
                EvaluationResult(
                    sample_id=sample_id,
                    expected_hallucination=expected,
                    predicted_hallucination=verdict.is_hallucination,
                    confidence=verdict.confidence,
                    method=method,
                    claim=claim,
                    matched_fact=verdict.matched_fact,
                )
            )

    aggregator.add_results(results)
    return aggregator, results
