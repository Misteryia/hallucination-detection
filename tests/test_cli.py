"""
Smoke-тест CLI.

Проверяет только структуру команд через :class:`click.testing.CliRunner` —
без запуска ML-моделей: реальные прогоны идут через сами команды
``haludetect check`` / ``haludetect evaluate`` и занимают минуты.

Запуск из корня проекта::

    python -m tests.test_cli
"""

from __future__ import annotations

from click.testing import CliRunner

from halu_detect import __version__
from halu_detect.cli import cli


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_root_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "check" in result.output
    assert "evaluate" in result.output


def test_check_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0, result.output
    for opt in ("--input", "--kb", "--method", "--mode",
                "--nli-threshold", "--similarity-threshold"):
        assert opt in result.output, f"в check --help нет {opt!r}"


def test_evaluate_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0, result.output
    for opt in ("--dataset", "--limit", "--method", "--mode",
                "--nli-threshold", "--similarity-threshold"):
        assert opt in result.output, f"в evaluate --help нет {opt!r}"


def test_unknown_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["does-not-exist"])
    assert result.exit_code != 0


def main() -> None:
    test_version()
    test_root_help()
    test_check_help()
    test_evaluate_help()
    test_unknown_command()
    print("OK: smoke-тесты CLI пройдены.")


if __name__ == "__main__":
    main()
