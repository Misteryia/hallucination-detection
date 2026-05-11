"""
Модуль Aggregator.

Накопитель и агрегатор результатов верификации. Принимает на вход
объекты :class:`EvaluationResult` (по одному на проверенный claim),
сохраняет их и по запросу подсчитывает сводные метрики качества
детектирования галлюцинаций: confusion matrix (TP/FP/TN/FN), Precision,
Recall, F1, Accuracy и интегральный показатель ``Q_prod``.

Формулы (защита от деления на ноль — возвращается ``0.0``):

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2 · P · R / (P + R)
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Q_prod    = 0.5 · precision + 0.5 · recall

Целевые показатели на датасете HaluEval: Precision ≥ 0.80,
Recall ≥ 0.75, ``Q_prod`` ≥ 0.775. Признак ``passes_quality_threshold``
выставляется в ``True``, если фактический ``Q_prod`` достиг порога,
вычисляемого из переданных целевых значений
(``0.5·target_precision + 0.5·target_recall``).

Реализация намеренно не использует scikit-learn: подсчёт метрик руками
прозрачен и легко цитируется в пояснительной записке.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Результат верификации одного утверждения с эталонной разметкой.

    Атрибуты:
        sample_id: порядковый номер исходного примера в датасете.
            Один пример HaluEval порождает обычно два результата
            (``right_answer`` и ``hallucinated_answer``), поэтому
            ``sample_id`` сам по себе не уникален в пределах списка.
        expected_hallucination: эталонная метка из датасета
            (``True`` для ``hallucinated_answer``, ``False`` для
            ``right_answer``).
        predicted_hallucination: предсказание системы.
        confidence: уверенность системы в принятом решении ``[0, 1]``.
        method: имя метода (``"nli"``, ``"embedding"``, ``"combined"``).
        claim: текст проверенного утверждения.
        matched_fact: факт из KB, на который сослалась система при
            принятии решения (см. :class:`Verdict`).
    """

    sample_id: int
    expected_hallucination: bool
    predicted_hallucination: bool
    confidence: float
    method: str
    claim: str
    matched_fact: Optional[str] = None


@dataclass
class Metrics:
    """Сводные метрики качества детектирования.

    Поля ``true_positive``, ``false_positive``, ``true_negative``,
    ``false_negative`` образуют confusion matrix; «положительным»
    классом считается «галлюцинация».
    """

    method: str
    n_samples: int

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    precision: float
    recall: float
    f1: float
    accuracy: float
    q_prod: float

    target_precision: float = 0.80
    target_recall: float = 0.75
    passes_quality_threshold: bool = False


class Aggregator:
    """Накопитель и агрегатор :class:`EvaluationResult`.

    Хранит результаты в памяти и по требованию подсчитывает метрики
    либо по всему накопленному набору, либо по выбранному методу.
    Не делает внешних вызовов и не зависит от ML-библиотек, что
    позволяет тестировать математику отдельно от тяжёлых моделей.

    Параметры конструктора:
        target_precision: целевой Precision (по умолчанию 0.80).
        target_recall: целевой Recall (по умолчанию 0.75).
            Порог по ``Q_prod`` вычисляется как
            ``0.5·target_precision + 0.5·target_recall``.
    """

    def __init__(
        self,
        target_precision: float = 0.80,
        target_recall: float = 0.75,
    ) -> None:
        self.target_precision = float(target_precision)
        self.target_recall = float(target_recall)
        self._results: List[EvaluationResult] = []

    # ------------------------------------------------------------------
    # Накопление
    # ------------------------------------------------------------------

    def add_result(self, result: EvaluationResult) -> None:
        """Добавить один результат в накопитель."""
        self._results.append(result)

    def add_results(self, results: List[EvaluationResult]) -> None:
        """Добавить список результатов в накопитель."""
        self._results.extend(results)

    def reset(self) -> None:
        """Очистить накопленные результаты."""
        self._results.clear()

    @property
    def results(self) -> List[EvaluationResult]:
        """Прочитать все накопленные результаты (копия списка)."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Подсчёт метрик
    # ------------------------------------------------------------------

    def compute_metrics(self, method: Optional[str] = None) -> Metrics:
        """Подсчитать сводные метрики по накопленным результатам.

        Параметры:
            method: если задано, в подсчёт включаются только результаты
                с соответствующим значением ``method``. Если ``None`` —
                учитываются все результаты, а в поле ``Metrics.method``
                записывается ``"all"``.
        """
        if method is None:
            subset = self._results
            method_label = "all"
        else:
            subset = [r for r in self._results if r.method == method]
            method_label = method

        tp = sum(1 for r in subset if r.expected_hallucination and r.predicted_hallucination)
        fp = sum(1 for r in subset if not r.expected_hallucination and r.predicted_hallucination)
        tn = sum(1 for r in subset if not r.expected_hallucination and not r.predicted_hallucination)
        fn = sum(1 for r in subset if r.expected_hallucination and not r.predicted_hallucination)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        q_prod = 0.5 * precision + 0.5 * recall

        threshold = 0.5 * self.target_precision + 0.5 * self.target_recall
        passes = q_prod >= threshold and total > 0

        metrics = Metrics(
            method=method_label,
            n_samples=total,
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            q_prod=q_prod,
            target_precision=self.target_precision,
            target_recall=self.target_recall,
            passes_quality_threshold=passes,
        )

        logger.info(
            "Метрики [%s]: TP=%d FP=%d TN=%d FN=%d | "
            "P=%.3f R=%.3f F1=%.3f Acc=%.3f Q_prod=%.3f (порог %.3f, %s)",
            method_label, tp, fp, tn, fn,
            precision, recall, f1, accuracy, q_prod,
            threshold, "OK" if passes else "ниже порога",
        )

        return metrics

    def compute_metrics_per_method(self) -> Dict[str, Metrics]:
        """Подсчитать метрики отдельно для каждого встретившегося метода.

        Возвращает словарь ``{method: Metrics}``. Полезно для построения
        сравнительной таблицы NLI vs embedding vs combined в записке.
        """
        methods = sorted({r.method for r in self._results})
        return {m: self.compute_metrics(method=m) for m in methods}

    # ------------------------------------------------------------------
    # Анализ ошибок
    # ------------------------------------------------------------------

    def get_errors(
        self, method: Optional[str] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """Сгруппировать ошибочные классификации.

        Возвращает словарь с двумя ключами:

          * ``"false_positives"`` — корректные ответы, ошибочно
            помеченные как галлюцинации;
          * ``"false_negatives"`` — пропущенные галлюцинации.

        Если задан ``method``, фильтруются результаты только по нему.
        """
        if method is None:
            subset = self._results
        else:
            subset = [r for r in self._results if r.method == method]

        return {
            "false_positives": [
                r for r in subset
                if not r.expected_hallucination and r.predicted_hallucination
            ],
            "false_negatives": [
                r for r in subset
                if r.expected_hallucination and not r.predicted_hallucination
            ],
        }
