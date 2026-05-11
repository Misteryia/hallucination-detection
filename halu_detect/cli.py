"""
CLI-интерфейс утилиты halu_detect.

Запуск (после установки пакета):
    haludetect check --input text.json --kb kb.json --output reports/check
    haludetect evaluate --dataset data/qa_data.json --limit 200 --output reports/eval

Запуск без установки (из корня репозитория):
    python -m halu_detect.cli check ...
    python -m halu_detect.cli evaluate ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .claim_extractor import ClaimExtractor
from .data_loader import load_custom, load_halueval
from .evaluation import evaluate_on_halueval
from .fact_checker import FactChecker
from .report_generator import ReportGenerator


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group()
@click.version_option(__version__, prog_name="haludetect")
def cli() -> None:
    """halu_detect — детектирование галлюцинаций в текстах LLM."""


# ----------------------------------------------------------------------
# CHECK
# ----------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="JSON-файл с проверяемым текстом ({\"text\": \"...\"})")
@click.option("--kb", "kb_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="JSON-файл с базой знаний ({\"facts\": [...]} или [...])")
@click.option("--output", "output_dir", default="reports",
              type=click.Path(path_type=Path),
              help="Каталог для отчётов (по умолчанию: reports)")
@click.option("--method", default="nli",
              type=click.Choice(["nli", "embedding", "combined"]),
              help="Метод верификации (по умолчанию: nli — оптимальный по grid search)")
@click.option("--mode", default="or", type=click.Choice(["or", "and"]),
              help="Режим объединения для combined")
@click.option("--nli-threshold", default=0.5, type=float, show_default=True)
@click.option("--similarity-threshold", default=0.6, type=float, show_default=True)
@click.option("--name", default="check_report",
              help="Базовое имя файлов отчёта")
@click.option("--verbose", "-v", is_flag=True, help="Подробный лог (DEBUG)")
def check(
    input_path: Path,
    kb_path: Path,
    output_dir: Path,
    method: str,
    mode: str,
    nli_threshold: float,
    similarity_threshold: float,
    name: str,
    verbose: bool,
) -> None:
    """Проверить пользовательский текст на галлюцинации.

    Читает текст и базу знаний, разбивает текст на claims, проверяет
    каждый claim указанным методом и формирует JSON+HTML отчёт.
    """
    _setup_logging(verbose)
    logger = logging.getLogger("haludetect.check")

    logger.info("Загрузка входных данных...")
    custom = load_custom(input_path, kb_path)
    logger.info("Текст: %d символов, фактов в KB: %d",
                len(custom.text), len(custom.facts))

    logger.info("Извлечение claims...")
    extractor = ClaimExtractor()
    claims = extractor.extract(custom.text)
    if not claims:
        click.echo("В тексте не найдено ни одного нетривиального утверждения.", err=True)
        sys.exit(1)
    logger.info("Извлечено claims: %d", len(claims))

    logger.info("Инициализация FactChecker (method=%s)...", method)
    checker = FactChecker(
        nli_threshold=nli_threshold,
        similarity_threshold=similarity_threshold,
        default_method=method,
    )

    logger.info("Верификация...")
    if method == "combined":
        verdicts = [checker.check_combined(c.text, custom.facts, mode=mode) for c in claims]
    elif method == "nli":
        verdicts = [checker.check_nli(c.text, custom.facts) for c in claims]
    else:
        verdicts = [checker.check_embedding(c.text, custom.facts) for c in claims]

    n_hallu = sum(1 for v in verdicts if v.is_hallucination)
    click.echo(f"\nПроверено claims: {len(verdicts)}")
    click.echo(f"  галлюцинаций:     {n_hallu}")
    click.echo(f"  подтверждённых:   {len(verdicts) - n_hallu}\n")

    rg = ReportGenerator(output_dir=output_dir)
    json_path, html_path = rg.generate_check_report(
        text=custom.text,
        verdicts=verdicts,
        method=method,
        output_name=name,
    )
    click.echo(f"Отчёт JSON: {json_path}")
    click.echo(f"Отчёт HTML: {html_path}")


# ----------------------------------------------------------------------
# EVALUATE
# ----------------------------------------------------------------------

@cli.command()
@click.option("--dataset", "dataset_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Путь к qa_data.json (HaluEval)")
@click.option("--limit", default=100, type=int, show_default=True,
              help="Сколько примеров взять из датасета")
@click.option("--method", default="nli",
              type=click.Choice(["nli", "embedding", "combined"]),
              help="Метод верификации")
@click.option("--mode", default="or", type=click.Choice(["or", "and"]))
@click.option("--nli-threshold", default=0.5, type=float, show_default=True)
@click.option("--similarity-threshold", default=0.6, type=float, show_default=True)
@click.option("--output", "output_dir", default="reports",
              type=click.Path(path_type=Path))
@click.option("--name", default="evaluation_report")
@click.option("--verbose", "-v", is_flag=True)
def evaluate(
    dataset_path: Path,
    limit: int,
    method: str,
    mode: str,
    nli_threshold: float,
    similarity_threshold: float,
    output_dir: Path,
    name: str,
    verbose: bool,
) -> None:
    """Оценить качество детектирования на датасете HaluEval.

    Прогоняет указанный метод по подмножеству датасета и формирует
    отчёт с метриками Precision/Recall/F1/Q_prod и анализом ошибок.
    """
    _setup_logging(verbose)
    logger = logging.getLogger("haludetect.evaluate")

    samples = load_halueval(dataset_path, limit=limit)
    logger.info("Загружено примеров: %d", len(samples))

    checker = FactChecker(
        nli_threshold=nli_threshold,
        similarity_threshold=similarity_threshold,
        default_method=method,
    )

    aggregator, _ = evaluate_on_halueval(
        samples=samples,
        fact_checker=checker,
        method=method,
        mode=mode,
    )

    metrics = aggregator.compute_metrics(method=method)
    click.echo(f"\n=== Результаты ({method}) ===")
    click.echo(f"  Precision: {metrics.precision:.3f}  (target {metrics.target_precision:.2f})")
    click.echo(f"  Recall:    {metrics.recall:.3f}  (target {metrics.target_recall:.2f})")
    click.echo(f"  F1:        {metrics.f1:.3f}")
    click.echo(f"  Accuracy:  {metrics.accuracy:.3f}")
    click.echo(f"  Q_prod:    {metrics.q_prod:.3f}")
    click.echo(f"  Confusion: TP={metrics.true_positive} FP={metrics.false_positive} "
               f"TN={metrics.true_negative} FN={metrics.false_negative}")
    status = "[OK] прошёл порог" if metrics.passes_quality_threshold else "[--] ниже порога"
    click.echo(f"  Status:    {status}\n")

    rg = ReportGenerator(output_dir=output_dir)
    json_path, html_path = rg.generate_evaluation_report(
        aggregator=aggregator,
        method=method,
        dataset_name="HaluEval",
        n_examples=len(samples),
        output_name=name,
    )
    click.echo(f"Отчёт JSON: {json_path}")
    click.echo(f"Отчёт HTML: {html_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
