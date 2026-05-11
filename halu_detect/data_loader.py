"""
Модуль DataLoader.

Отвечает за загрузку и предварительную обработку входных данных для
системы детектирования галлюцинаций. Поддерживает два сценария:

  1. Загрузка эталонного датасета HaluEval (qa_data.json) — используется
     командой ``haludetect evaluate`` для оценки качества двух методов
     верификации (NLI и embedding similarity) на размеченных примерах.

  2. Загрузка пользовательской пары "текст LLM + база знаний" из двух
     JSON-файлов — используется командой ``haludetect check`` для
     проверки одного произвольного текста.

Нормализация входных строк ограничивается удалением ведущих и хвостовых
пробельных символов и схлопыванием последовательностей пробелов в один
пробел.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Тип, обозначающий путь — путь может быть как строкой, так и Path.
PathLike = Union[str, Path]


@dataclass
class HaluEvalSample:
    """Один размеченный пример из датасета HaluEval (подкатегория QA).

    Атрибуты:
        knowledge: текст-источник (например, отрывок из Википедии),
            рассматриваемый как база знаний (premise).
        question: вопрос, относящийся к ``knowledge``.
        right_answer: эталонный (не-галлюцинированный) ответ.
        hallucinated_answer: ответ, содержащий галлюцинацию.
    """

    knowledge: str
    question: str
    right_answer: str
    hallucinated_answer: str


@dataclass
class CustomInput:
    """Пользовательский ввод для команды ``check``.

    Атрибуты:
        text: проверяемый текст, сгенерированный LLM.
        facts: список фактов из базы знаний; каждый факт — отдельная
            строка, выступающая в роли premise при NLI-верификации.
    """

    text: str
    facts: List[str]


def normalize_text(text: str) -> str:
    """Привести текст к каноническому виду.

    Удаляет ведущие и хвостовые пробельные символы и схлопывает любые
    последовательности пробелов (включая переводы строк и табуляции) в
    один пробел.

    Регистр символов сохраняется намеренно: используемые далее модели
    (NLI ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` и
    эмбеддер ``sentence-transformers/all-mpnet-base-v2``) обучались на
    текстах с естественным регистром, и приведение к нижнему регистру
    заметно ухудшает качество предсказаний.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_halueval(
    path: PathLike, limit: Optional[int] = None
) -> List[HaluEvalSample]:
    """Загрузить датасет HaluEval из JSON Lines-файла.

    Файл ``qa_data.json`` содержит по одному JSON-объекту на строку с
    полями ``knowledge``, ``question``, ``right_answer`` и
    ``hallucinated_answer``. Несмотря на расширение ``.json``, формат
    именно построчный (JSONL), поэтому файл читается потоково.

    Параметры:
        path: путь к файлу датасета.
        limit: ограничение на количество загружаемых примеров; ``None``
            означает «загрузить все».

    Возвращает:
        Список объектов :class:`HaluEvalSample` с нормализованными
        строковыми полями.

    Исключения:
        FileNotFoundError: если файл не существует.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    logger.info("Загрузка HaluEval из %s (limit=%s)", path, limit)

    samples: List[HaluEvalSample] = []
    required_fields = ("knowledge", "question", "right_answer", "hallucinated_answer")

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Строка %d: некорректный JSON, пропуск (%s)", line_no, exc)
                continue

            missing = [k for k in required_fields if k not in obj]
            if missing:
                logger.warning("Строка %d: отсутствуют поля %s, пропуск", line_no, missing)
                continue

            samples.append(
                HaluEvalSample(
                    knowledge=normalize_text(obj["knowledge"]),
                    question=normalize_text(obj["question"]),
                    right_answer=normalize_text(obj["right_answer"]),
                    hallucinated_answer=normalize_text(obj["hallucinated_answer"]),
                )
            )

            if limit is not None and len(samples) >= limit:
                break

    logger.info("Загружено %d примеров HaluEval", len(samples))
    return samples


def load_custom(input_path: PathLike, kb_path: PathLike) -> CustomInput:
    """Загрузить пару «текст LLM + база знаний» из двух JSON-файлов.

    Ожидаемый формат файла ``input_path``::

        {"text": "проверяемый текст ..."}

    Ожидаемый формат файла ``kb_path`` (поддерживаются оба варианта)::

        {"facts": ["факт 1", "факт 2", ...]}
        ["факт 1", "факт 2", ...]

    Параметры:
        input_path: путь к JSON-файлу с проверяемым текстом.
        kb_path: путь к JSON-файлу с базой знаний.

    Возвращает:
        Объект :class:`CustomInput` с нормализованным текстом и
        непустыми нормализованными фактами.

    Исключения:
        FileNotFoundError: если какой-либо из файлов не существует.
        KeyError: если структура входных файлов не соответствует
            ожидаемой.
        ValueError: если KB имеет неподдерживаемый формат.
    """
    input_path = Path(input_path)
    kb_path = Path(kb_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Файл входного текста не найден: {input_path}")
    if not kb_path.is_file():
        raise FileNotFoundError(f"Файл базы знаний не найден: {kb_path}")

    logger.info("Загрузка пользовательского текста из %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        input_obj = json.load(f)

    if not isinstance(input_obj, dict) or "text" not in input_obj:
        raise KeyError(f"В файле {input_path} ожидается объект с полем 'text'")

    logger.info("Загрузка базы знаний из %s", kb_path)
    with kb_path.open("r", encoding="utf-8") as f:
        kb_obj = json.load(f)

    if isinstance(kb_obj, dict):
        if "facts" not in kb_obj:
            raise KeyError(f"В файле {kb_path} ожидается объект с полем 'facts'")
        raw_facts = kb_obj["facts"]
    elif isinstance(kb_obj, list):
        raw_facts = kb_obj
    else:
        raise ValueError(
            f"Неподдерживаемый формат базы знаний в {kb_path}: "
            f"ожидался объект или массив, получено {type(kb_obj).__name__}"
        )

    facts = [normalize_text(str(item)) for item in raw_facts]
    facts = [f for f in facts if f]  # отфильтровать пустые строки

    logger.info("Загружено фактов: %d", len(facts))
    return CustomInput(text=normalize_text(input_obj["text"]), facts=facts)
