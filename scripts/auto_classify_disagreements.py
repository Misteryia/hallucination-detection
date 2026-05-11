"""
Автоматическая классификация LLM↔HaluEval расхождений через haiku.

Принимает на вход результат ``run_llm_baseline`` (например,
``reports/llm_baseline_n1991.json``), отбирает все случаи, где LLM не
согласилась с разметкой HaluEval (``agreement == False``), и для каждого
запрашивает у claude-haiku одну из трёх категорий:

  (a) BAD_LABEL  — разметка HaluEval ошибочна или спорна; LLM права.
  (b) EVASIVE    — ответ уклончивый/общий, фактически не ложный.
  (c) MODEL_MISS — реальная ошибка LLM-судьи; разметка корректна.

Результат — JSON со счётчиками и детальной разбивкой, плюс короткий
markdown-отчёт. Поддерживает чекпоинтинг: можно безопасно прерывать и
продолжать с того места, где остановилось.

Запуск:

    python -m scripts.auto_classify_disagreements \\
        --input reports/llm_baseline_n1991.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from tqdm import tqdm  # noqa: E402

from halu_detect.data_loader import load_halueval  # noqa: E402

logger = logging.getLogger(__name__)

CATEGORIES = ("BAD_LABEL", "EVASIVE", "MODEL_MISS")


PROMPT_TEMPLATE = """Ты модератор разметки датасета HaluEval. Дано:

КОНТЕКСТ (knowledge):
{knowledge}

ВОПРОС:
{question}

ОТВЕТ (для проверки):
{answer}

ЭТАЛОННАЯ РАЗМЕТКА HaluEval (2023): ответ {expected_label}
ВЕРДИКТ LLM-СУДЬИ: ответ {predicted_label}
ОБЪЯСНЕНИЕ LLM-СУДЬИ: {llm_explanation}

LLM-судья НЕ согласился с эталонной разметкой. Классифицируй причину расхождения СТРОГО в одну из трёх категорий:

(a) BAD_LABEL — разметка HaluEval ошибочна или спорна; LLM-судья права. Например, в "галлюцинации" по факту нет противоречия с контекстом.
(b) EVASIVE — ответ уклончивый, общий или неполный; формально не противоречит контексту, но и не отвечает по существу. Здесь спор между "галлюцинация ли это или просто плохой ответ".
(c) MODEL_MISS — реальная ошибка LLM-судьи; эталонная разметка корректна, а LLM упустила противоречие или, наоборот, выдумала его.

Ответь СТРОГО в формате:

CATEGORY: <BAD_LABEL|EVASIVE|MODEL_MISS>
JUSTIFICATION: одно короткое предложение, почему именно эта категория."""


def _label(b: bool) -> str:
    return "является галлюцинацией" if b else "корректен"


def parse_category(text: str) -> Optional[str]:
    m = re.search(r"CATEGORY\s*[:\-]\s*([A-Z_]+)", text, re.IGNORECASE)
    if not m:
        return None
    cat = m.group(1).upper()
    if cat in CATEGORIES:
        return cat
    # Допускаем сокращённые формы.
    for c in CATEGORIES:
        if cat.startswith(c[:5]):
            return c
    return None


def parse_justification(text: str) -> str:
    m = re.search(
        r"JUSTIFICATION\s*[:\-]\s*(.+?)(?:\n\s*\n|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def query_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def render_summary(counter: Counter, total: int, model: str) -> str:
    out = []
    out.append("=" * 78)
    out.append("=== АВТОКЛАССИФИКАЦИЯ РАСХОЖДЕНИЙ LLM↔HaluEval ===")
    out.append("=" * 78)
    out.append(f"Модератор:           {model}")
    out.append(f"Расхождений всего:   {total}")
    out.append("-" * 78)
    for cat in CATEGORIES:
        n = counter.get(cat, 0)
        pct = 100.0 * n / total if total else 0.0
        out.append(f"  {cat:<14} {n:>5}  ({pct:.1f} %)")
    n_unparsed = counter.get(None, 0)
    if n_unparsed:
        out.append(f"  не распознан       {n_unparsed:>5}  ({100.0*n_unparsed/total:.1f} %)")
    out.append("=" * 78)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("reports/llm_baseline_n1991.json"),
        help="JSON от run_llm_baseline с полями details/agreement.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/disagreements_auto.json"),
    )
    parser.add_argument(
        "--md-out", type=Path,
        default=Path("reports/disagreements_auto.md"),
    )
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Чекпоинт каждые N классификаций.",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/qa_data.json"),
        help="HaluEval — нужен, чтобы достать knowledge по sample_id.",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Нужно установить ANTHROPIC_API_KEY (через .env в корне проекта)"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else project_root / p

    in_path = _resolve(args.input)
    out_path = _resolve(args.out)
    md_path = _resolve(args.md_out)
    data_path = _resolve(args.data)

    samples = load_halueval(data_path)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    details = payload.get("details", [])
    disagreements = [d for d in details if d.get("agreement") is False]
    print(f"\nИсходник:           {in_path}")
    print(f"Всего claims:       {len(details)}")
    print(f"Расхождений (agree=False): {len(disagreements)}")
    print()

    # ── Чекпоинт: загрузить уже классифицированные ──────────────────────
    classified: Dict[str, Dict] = {}  # ключ — f"{sample_id}/{kind}"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("rows", []):
                key = f"{r['sample_id']}/{r['kind']}"
                classified[key] = r
            print(f"✓ Чекпоинт: {len(classified)} уже классифицировано.")
        except Exception as exc:
            logger.warning("Не удалось прочитать чекпоинт %s: %s", out_path, exc)
            classified = {}

    def _save_partial(completed: bool) -> None:
        rows = list(classified.values())
        counter = Counter(r["category"] for r in rows)
        out = {
            "metadata": {
                "input": str(in_path.relative_to(project_root)) if in_path.is_relative_to(project_root) else str(in_path),
                "model": args.model,
                "n_disagreements": len(disagreements),
                "n_classified": len(rows),
                "completed": completed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "counts": dict(counter),
            "rows": rows,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)

    pending = [d for d in disagreements
               if f"{d['sample_id']}/{d['kind']}" not in classified]
    if not pending:
        print("Все расхождения уже классифицированы. Печатаю сводку и выхожу.")
    else:
        print(f"К классификации:    {len(pending)} (через {args.model})\n")

    for i, d in enumerate(tqdm(pending, desc="Classifying")):
        sample = samples[d["sample_id"]]
        prompt = PROMPT_TEMPLATE.format(
            knowledge=sample.knowledge,
            question=d["question"],
            answer=d["answer"],
            expected_label=_label(d["expected"]),
            predicted_label=_label(d["predicted"]) if d.get("predicted") is not None else "(не распознан)",
            llm_explanation=d.get("explanation", "") or "(не было объяснения)",
        )
        try:
            raw = query_anthropic(prompt, args.model)
            cat = parse_category(raw)
            just = parse_justification(raw)
        except Exception as exc:
            logger.warning("API-ошибка (sample %s, %s): %s",
                           d["sample_id"], d["kind"], exc)
            raw, cat, just = f"ERROR: {exc}", None, ""

        key = f"{d['sample_id']}/{d['kind']}"
        classified[key] = {
            "sample_id": d["sample_id"],
            "kind": d["kind"],
            "question": d["question"],
            "answer": d["answer"],
            "expected": d["expected"],
            "predicted": d.get("predicted"),
            "explanation": d.get("explanation", ""),
            "category": cat,
            "justification": just,
            "raw_response": raw,
        }

        if (i + 1) % args.save_every == 0:
            _save_partial(completed=False)
            logger.info("Чекпоинт: %d/%d расхождений классифицировано.",
                        len(classified), len(disagreements))

        time.sleep(args.sleep)

    _save_partial(completed=True)

    # ── Сводка ───────────────────────────────────────────────────────────
    rows = list(classified.values())
    counter = Counter(r["category"] for r in rows)
    summary = render_summary(counter, len(rows), args.model)
    print()
    print(summary)
    print(f"\nJSON сохранён: {out_path}")

    # ── Markdown с примерами ─────────────────────────────────────────────
    md_lines = ["# Автоклассификация LLM↔HaluEval расхождений", "", summary, ""]
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        md_lines.append(f"## {cat} — {len(cat_rows)} случаев")
        md_lines.append("")
        for r in cat_rows[:8]:
            md_lines.append(f"### sample_id={r['sample_id']} ({r['kind']})")
            md_lines.append(f"**Q:** {r['question']}")
            md_lines.append("")
            md_lines.append(f"**A:** {r['answer']}")
            md_lines.append("")
            md_lines.append(f"**HaluEval:** {_label(r['expected'])}; "
                            f"**LLM:** {_label(r['predicted']) if r.get('predicted') is not None else '?'}")
            md_lines.append("")
            md_lines.append(f"**LLM explanation:** {r['explanation']}")
            md_lines.append("")
            md_lines.append(f"**Justification:** {r['justification']}")
            md_lines.append("")
        md_lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown сохранён: {md_path}")


if __name__ == "__main__":
    main()
