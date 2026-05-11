#!/usr/bin/env python3
"""Подготовить два независимых файла для нового ручного разбора:

  * ``reports/manual_review_haiku_100.md`` — 100 случаев из расхождений
    Claude Haiku 4.5 (seed=100);
  * ``reports/manual_review_gpt_100.md``   — 100 случаев из расхождений
    GPT-5.4-mini (seed=200).

Старая выборка из 50 случаев (``manual_review_50_filled.md``) НЕ
используется и НЕ перезаписывается. Сиды 100 и 200 выбраны так, чтобы
не совпасть со старым seed=42 — это гарантирует независимость новой
разметки от прошлой.

Поля JSON-баз обоих арбитров идентичны (sample_id, kind, question,
answer, expected, predicted, explanation, agreement, ...), но
``knowledge`` в них отсутствует — он подгружается из
``data/qa_data.json`` через :func:`halu_detect.data_loader.load_halueval`.

Запуск:

    python -m scripts.prepare_manual_review_v3
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
QA_DATA = ROOT / "data" / "qa_data.json"

HAIKU_BASELINE = ROOT / "reports" / "llm_baseline_n1991.json"
GPT_BASELINE = ROOT / "reports" / "llm_baseline_openai_n1991.json"

OUT_HAIKU = ROOT / "reports" / "manual_review_haiku_100.md"
OUT_GPT = ROOT / "reports" / "manual_review_gpt_100.md"

SEED_HAIKU = 100
SEED_GPT = 200
N_CASES = 100


INSTRUCTIONS = """# Ручной разбор расхождений LLM vs HaluEval

## Главное правило: closed-book оценка по контексту

HaluEval — это closed-book benchmark. Галлюцинация определяется относительно ПРЕДОСТАВЛЕННОГО контекста (поле knowledge), а не относительно реального мира. Ты разбираешь по тем же правилам, что и LLM-арбитр: только контекст, без интернета, без своих знаний.

**Исключение:** интернет можно использовать ТОЛЬКО для проверки подозрений на BAD_LABEL — то есть когда хочешь убедиться, что разметка датасета фактически неверна по реальному миру. В остальных случаях — строго контекст.

## Алгоритм разбора одного случая

1. Прочитай контекст и вопрос
2. Прочитай проверяемый ответ
3. Зафиксируй конфликт: HaluEval сказал одно, LLM-арбитр другое
4. Сам ответь на вопрос «галлюцинация или нет?» — ОПИРАЯСЬ ТОЛЬКО НА КОНТЕКСТ (не на разметку HaluEval, не на вердикт LLM, не на свою память)
5. Соотнеси с разметкой HaluEval и вердиктом LLM:

| Твоё решение | HaluEval | LLM | Категория |
|---|---|---|---|
| Ответ галлюцинирует | согласен | не согласен | c) MODEL_MISS |
| Ответ корректен | не согласен | согласен | a) BAD_LABEL |
| Ответ корректен по фактам, но не отвечает на вопрос конкретно | формально согласен | формально не согласен | b) EVASIVE |

## Категории

**a) BAD_LABEL — разметка HaluEval ошибочна**

- Правильный ответ датасета фактически неверен (например, перепутан genus и family)
- Галлюцинированный ответ датасета на самом деле тоже корректен по контексту
- Ответ — синонимическая переформулировка факта из контекста (например, «The Post-American» = «Sojourners Magazine», если контекст это явно утверждает)
- Вопрос настолько неоднозначен, что оба ответа допустимы

**b) EVASIVE — уклончивый, но не ложный ответ**

- Ответ слишком общий, не отвечает на вопрос конкретно (например, «an influential figure» вместо имени)
- Не содержит фактической ошибки и не противоречит контексту
- Формально не отвечает на вопрос, но и не лжёт

**c) MODEL_MISS — реальная ошибка LLM как арбитра**

- Утверждение действительно содержит галлюцинацию (контекст говорит одно, ответ — другое), но LLM не распознала
- ИЛИ ответ корректен, но LLM ошибочно пометила как галлюцинацию
- LLM использует внешнее знание против контекста (типичный case: Premier League не существовала в 1902, но контекст говорит что Liverpool is a Premier League team — ответ «не Premier League» есть галлюцинация по правилам HaluEval, и LLM ошиблась если её приняла)

## Подход: строго к LLM, мягко к разметке

Это соответствует closed-book methodology HaluEval.

**Ставь `c) MODEL_MISS`, если:**

- LLM использует внешнее знание против контекста
- LLM явно игнорирует факт из контекста
- LLM путает синонимичные сущности с разными
- LLM не видит явное противоречие

**Ставь `a) BAD_LABEL`, если:**

- Правильный ответ HaluEval сам по себе фактически неверен
- Контекст HaluEval сам содержит ошибку или противоречие
- Вопрос двусмыслен, и ответ — одна из допустимых интерпретаций
- Ответ — семантическая эквивалентность по контексту

**Ставь `b) EVASIVE`, если:**

- Ответ не отвечает на конкретный вопрос (общими словами)
- Но не содержит фактической ошибки
- И не противоречит контексту

## Практические советы

- 2-3 минуты на случай максимум. Не зависай.
- Записывай обоснование в одну строку — пригодится на защите.
- Если совсем 50/50 — выбирай по причине расхождения с разметкой HaluEval.
- Не калибруйся под распределения. Ставь честно как видишь.
- Внешние знания используй только для подтверждения BAD_LABEL.

---

"""


def _label_halueval(kind: str) -> str:
    return "галлюцинация" if kind == "hallucinated" else "корректный ответ"


def _label_llm(predicted: object) -> str:
    if predicted is None:
        return "не распознан"
    return "галлюцинация" if predicted else "корректный ответ"


def make_review_file(
    disagreements: List[Dict],
    knowledge_by_sid: Dict[int, str],
    output_path: Path,
    seed: int,
    n_cases: int,
    source_header: str,
) -> None:
    """Создать markdown-файл для разбора ``n_cases`` случаев."""
    rng = random.Random(seed)
    cases = rng.sample(disagreements, n_cases)

    parts: List[str] = [INSTRUCTIONS, source_header, "", ""]

    for i, case in enumerate(cases, start=1):
        sid = case["sample_id"]
        kind = case["kind"]
        knowledge = knowledge_by_sid.get(sid, "")
        question = case["question"]
        answer = case["answer"]
        explanation = case.get("explanation") or "(нет объяснения)"

        parts.append(f"## Случай №{i} (sample_id={sid}, kind={kind})\n")
        parts.append("**Контекст (knowledge):**\n")
        parts.append(f"> {knowledge}\n")
        parts.append("**Вопрос:**\n")
        parts.append(f"{question}\n")
        parts.append("**Ответ для проверки:**\n")
        parts.append(f"{answer}\n")
        parts.append(f"**Разметка HaluEval:** {_label_halueval(kind)}\n")
        parts.append(f"**Вердикт LLM:** {_label_llm(case.get('predicted'))}\n")
        parts.append("**Обоснование LLM:**\n")
        parts.append(f"{explanation}\n")
        parts.append("**Моя категория:** _\n")
        parts.append("**Моё обоснование (по желанию):**\n")
        parts.append("---\n")

    parts.append(f"\n## Итоговый подсчёт ({n_cases} случаев)\n")
    parts.append("| Категория | Количество | Процент |")
    parts.append("|---|---|---|")
    parts.append("| a (BAD_LABEL)   | __ | __% |")
    parts.append("| b (EVASIVE)     | __ | __% |")
    parts.append("| c (MODEL_MISS)  | __ | __% |")
    parts.append(f"| ВСЕГО           | {n_cases} | 100% |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _load_disagreements(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [d for d in payload["details"] if d.get("agreement") is False]


def main() -> None:
    from halu_detect.data_loader import load_halueval

    samples = load_halueval(QA_DATA)
    knowledge_by_sid = {i: s.knowledge for i, s in enumerate(samples)}

    haiku_disagreements = _load_disagreements(HAIKU_BASELINE)
    gpt_disagreements = _load_disagreements(GPT_BASELINE)

    print(f"Haiku расхождений: {len(haiku_disagreements)}")
    print(f"GPT расхождений:   {len(gpt_disagreements)}")
    print()

    make_review_file(
        haiku_disagreements,
        knowledge_by_sid,
        OUT_HAIKU,
        seed=SEED_HAIKU,
        n_cases=N_CASES,
        source_header=(
            f"**Источник:** расхождения арбитра **Claude Haiku 4.5** "
            f"с разметкой HaluEval-QA на полной тестовой выборке. "
            f"Случайная выборка {N_CASES} случаев из "
            f"{len(haiku_disagreements)} расхождений (seed={SEED_HAIKU})."
        ),
    )

    make_review_file(
        gpt_disagreements,
        knowledge_by_sid,
        OUT_GPT,
        seed=SEED_GPT,
        n_cases=N_CASES,
        source_header=(
            f"**Источник:** расхождения арбитра **GPT-5.4-mini** "
            f"с разметкой HaluEval-QA на полной тестовой выборке. "
            f"Случайная выборка {N_CASES} случаев из "
            f"{len(gpt_disagreements)} расхождений (seed={SEED_GPT})."
        ),
    )

    print("Созданы файлы:")
    for p in (OUT_HAIKU, OUT_GPT):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(ROOT)}   ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
