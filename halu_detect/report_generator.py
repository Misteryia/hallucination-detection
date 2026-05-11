"""
Модуль ReportGenerator.

Формирует отчёты по результатам работы утилиты в двух форматах
одновременно — JSON (машиночитаемый, для дальнейшего анализа и
включения в архивные записи) и HTML (человекочитаемый, для
демонстрации научному руководителю и для скриншотов в пояснительной
записке).

Поддерживает два сценария:

  * **Проверка одного текста** (CLI ``check``). Вход — исходный текст,
    список :class:`Verdict` для извлечённых из него claim-ов и имя
    использованного метода. На выходе — JSON с полным набором вердиктов
    и HTML с подсветкой галлюцинаций по тексту и таблицей.

  * **Оценка на размеченном датасете** (CLI ``evaluate``). Вход —
    заполненный :class:`Aggregator` с накопленными
    :class:`EvaluationResult` и метаданные эксперимента. На выходе —
    JSON с метриками и примерами FP/FN; HTML с таблицей метрик,
    confusion matrix и топом ошибок.

HTML-шаблоны (Jinja2) лежат в подпакете ``halu_detect/templates`` и
загружаются через :class:`PackageLoader`. Все CSS — inline, без внешних
ресурсов: отчёт должен открываться без интернета (на ноутбуке для
демонстрации перед комиссией).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from . import __version__
from .aggregator import Aggregator, EvaluationResult, Metrics
from .fact_checker import Verdict

logger = logging.getLogger(__name__)


_TOP_ERRORS = 10  # сколько FP/FN включать в JSON и HTML отчёта


class ReportGenerator:
    """Генератор JSON- и HTML-отчётов.

    Параметры:
        output_dir: каталог, в который пишутся файлы отчётов
            (создаётся, если не существует).
    """

    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(
            loader=PackageLoader("halu_detect", "templates"),
            autoescape=select_autoescape(default=True, default_for_string=True),
            trim_blocks=False,
            lstrip_blocks=False,
        )

    # ------------------------------------------------------------------
    # Сценарий A: один текст
    # ------------------------------------------------------------------

    def generate_check_report(
        self,
        text: str,
        verdicts: List[Verdict],
        method: str,
        output_name: str = "check_report",
    ) -> Tuple[Path, Path]:
        """Сгенерировать отчёт по проверке одного текста.

        Возвращает кортеж ``(json_path, html_path)``.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")

        n_hallu = sum(1 for v in verdicts if v.is_hallucination)
        n_ok = len(verdicts) - n_hallu

        json_data = {
            "metadata": {
                "method": method,
                "n_verdicts": len(verdicts),
                "timestamp": timestamp,
            },
            "text": text,
            "summary": {
                "total": len(verdicts),
                "hallucinations": n_hallu,
                "ok": n_ok,
            },
            "verdicts": [_verdict_to_dict(v) for v in verdicts],
        }

        json_path = self._write_json(json_data, output_name)

        # Подсветка claim-ов в исходном тексте: ищем точные вхождения
        # текстов claim в text и оборачиваем их span-ами.
        highlighted = _highlight_claims_in_text(text, verdicts)

        html_path = self._render_html(
            template_name="check_report.html.j2",
            output_name=output_name,
            context={
                "metadata": json_data["metadata"],
                "summary": json_data["summary"],
                "verdicts": verdicts,
                "highlighted_text": Markup(highlighted),
                "version": __version__,
            },
        )

        logger.info("Check-отчёт записан: %s, %s", json_path, html_path)
        return json_path, html_path

    # ------------------------------------------------------------------
    # Сценарий B: датасет
    # ------------------------------------------------------------------

    def generate_evaluation_report(
        self,
        aggregator: Aggregator,
        method: str,
        dataset_name: str = "HaluEval",
        n_examples: int = 0,
        output_name: str = "evaluation_report",
    ) -> Tuple[Path, Path]:
        """Сгенерировать отчёт по оценке на размеченном датасете.

        Метрики берутся через ``aggregator.compute_metrics(method=method)``
        (если ``method`` не равен ``"all"``), либо общие
        ``compute_metrics()`` иначе. Для разделов FP/FN отбираются
        ``_TOP_ERRORS`` записей с наибольшим ``confidence``.

        Возвращает кортеж ``(json_path, html_path)``.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")

        if method in ("nli", "embedding", "combined"):
            metrics = aggregator.compute_metrics(method=method)
            errors = aggregator.get_errors(method=method)
        else:
            metrics = aggregator.compute_metrics()
            errors = aggregator.get_errors()

        fps_top = sorted(errors["false_positives"], key=lambda r: r.confidence, reverse=True)[:_TOP_ERRORS]
        fns_top = sorted(errors["false_negatives"], key=lambda r: r.confidence, reverse=True)[:_TOP_ERRORS]

        target_q_prod = 0.5 * metrics.target_precision + 0.5 * metrics.target_recall

        json_data = {
            "metadata": {
                "dataset": dataset_name,
                "method": method,
                "n_examples": n_examples,
                "timestamp": timestamp,
            },
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "accuracy": metrics.accuracy,
                "q_prod": metrics.q_prod,
                "passes_quality_threshold": metrics.passes_quality_threshold,
                "target_precision": metrics.target_precision,
                "target_recall": metrics.target_recall,
                "target_q_prod": target_q_prod,
                "confusion_matrix": {
                    "tp": metrics.true_positive,
                    "fp": metrics.false_positive,
                    "tn": metrics.true_negative,
                    "fn": metrics.false_negative,
                },
            },
            "errors": {
                "false_positives": [_eval_result_to_dict(r) for r in fps_top],
                "false_negatives": [_eval_result_to_dict(r) for r in fns_top],
            },
        }

        json_path = self._write_json(json_data, output_name)

        html_path = self._render_html(
            template_name="evaluation_report.html.j2",
            output_name=output_name,
            context={
                "metadata": json_data["metadata"],
                "metrics": metrics,
                "target_q_prod": target_q_prod,
                "errors": {
                    "false_positives": fps_top,
                    "false_negatives": fns_top,
                },
                "version": __version__,
            },
        )

        logger.info("Evaluation-отчёт записан: %s, %s", json_path, html_path)
        return json_path, html_path

    # ------------------------------------------------------------------
    # Внутренние утилиты
    # ------------------------------------------------------------------

    def _write_json(self, data: Dict[str, Any], output_name: str) -> Path:
        path = self.output_dir / f"{output_name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _render_html(
        self,
        template_name: str,
        output_name: str,
        context: Dict[str, Any],
    ) -> Path:
        template = self._env.get_template(template_name)
        rendered = template.render(**context)
        path = self.output_dir / f"{output_name}.html"
        path.write_text(rendered, encoding="utf-8")
        return path


# ----------------------------------------------------------------------
# Сериализация и подсветка
# ----------------------------------------------------------------------

def _verdict_to_dict(v: Verdict) -> Dict[str, Any]:
    return {
        "claim": v.claim,
        "method": v.method,
        "is_hallucination": v.is_hallucination,
        "confidence": v.confidence,
        "matched_fact": v.matched_fact,
        "details": v.details,
    }


def _eval_result_to_dict(r: EvaluationResult) -> Dict[str, Any]:
    return {
        "sample_id": r.sample_id,
        "claim": r.claim,
        "method": r.method,
        "expected_hallucination": r.expected_hallucination,
        "predicted_hallucination": r.predicted_hallucination,
        "confidence": r.confidence,
        "matched_fact": r.matched_fact,
    }


def _highlight_claims_in_text(text: str, verdicts: List[Verdict]) -> str:
    """Вставить в текст HTML-спаны вокруг каждого найденного claim.

    Алгоритм: для каждого вердикта пытаемся найти точное вхождение
    ``claim`` в исходном тексте. Найденные позиции упорядочиваем и
    сшиваем итоговый HTML, экранируя «обычные» промежутки между
    подсвеченными спанами. Не найденные дословно claim-ы добавляются
    отдельным блоком в конце — это редкий случай, возникающий только
    при существенном расхождении сегментации (например, если claim
    был получен из другой версии текста).
    """
    spans: List[Tuple[int, int, Verdict]] = []
    used_positions: set = set()

    for v in verdicts:
        if not v.claim:
            continue
        idx = text.find(v.claim)
        # Если первое вхождение уже занято другим claim — попробуем
        # следующее. Это даёт предсказуемое поведение при повторах.
        while idx != -1 and idx in used_positions:
            idx = text.find(v.claim, idx + 1)
        if idx == -1:
            continue
        spans.append((idx, idx + len(v.claim), v))
        used_positions.add(idx)

    spans.sort(key=lambda s: s[0])

    parts: List[str] = []
    cursor = 0
    for start, end, v in spans:
        if start < cursor:  # перекрытие — пропускаем
            continue
        if start > cursor:
            parts.append(escape(text[cursor:start]))
        cls = "hallu" if v.is_hallucination else "ok"
        title = (
            f"{'Галлюцинация' if v.is_hallucination else 'Подтверждено'} "
            f"(conf={v.confidence:.2f}). "
            f"Matched fact: {v.matched_fact or '—'}"
        )
        parts.append(
            f'<span class="{cls}" title="{escape(title)}">{escape(text[start:end])}</span>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(escape(text[cursor:]))

    found_claims = {(s, e) for s, e, _ in spans}
    if len(found_claims) < len([v for v in verdicts if v.claim]):
        # Дописать ненайденные claim-ы отдельным блоком, чтобы
        # пользователь видел все вердикты, даже если span-разметка
        # текста с ними не сошлась.
        unmatched = [
            v for v in verdicts
            if v.claim and not _claim_in_spans(v.claim, text, spans)
        ]
        if unmatched:
            parts.append('\n\n<em>Утверждения, не найденные дословно в тексте:</em>\n')
            for v in unmatched:
                cls = "hallu" if v.is_hallucination else "ok"
                title = (
                    f"{'Галлюцинация' if v.is_hallucination else 'Подтверждено'} "
                    f"(conf={v.confidence:.2f}). "
                    f"Matched fact: {v.matched_fact or '—'}"
                )
                parts.append(
                    f'<span class="{cls}" title="{escape(title)}">{escape(v.claim)}</span> '
                )

    return "".join(parts)


def _claim_in_spans(claim: str, text: str, spans: List[Tuple[int, int, Verdict]]) -> bool:
    """Проверить, что claim соответствует одному из подсвеченных спанов."""
    return any(text[s:e] == claim for s, e, _ in spans)
