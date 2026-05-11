"""
Модуль ClaimExtractor.

Отвечает за разбиение исходного текста (как правило — ответа LLM) на
атомарные утверждения (claims). Для MVP за атомарное утверждение
принимается одно предложение: такая гранулярность достаточна, поскольку
последующая верификация (NLI и embedding similarity) выполняется именно
на уровне предложений и более тонкое разбиение не повышает качества
детектирования на датасете HaluEval.

Сегментация выполняется средствами spaCy (модель ``en_core_web_sm``),
поскольку правила-разделители на основе регулярных выражений не
справляются с такими случаями, как сокращения («Mr.», «Dr.»),
многоточия и десятичные дроби.

Тривиальные предложения отфильтровываются:

  * вопросительные (заканчиваются знаком «?») — по определению не
    являются утверждениями;
  * слишком короткие (менее ``min_words`` слов) — не несут проверяемой
    информации;
  * не содержащие существительных или имён собственных — состоят из
    одних местоимений и служебных слов и не могут быть верифицированы
    относительно базы знаний.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import spacy
from spacy.language import Language
from spacy.tokens import Span

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Атомарное утверждение, извлечённое из текста.

    Атрибуты:
        text: текст предложения после удаления хвостовых пробелов.
        position: порядковый индекс предложения в исходном тексте,
            начиная с нуля. Индекс соответствует положению предложения
            *до* фильтрации, то есть позволяет восстановить, в каком
            месте исходного текста находилось утверждение.
    """

    text: str
    position: int


class ClaimExtractor:
    """Разбивает текст на атомарные утверждения с фильтрацией тривиальных.

    spaCy-модель загружается один раз при создании экземпляра класса —
    повторная загрузка модели на каждый вызов недопустима, поскольку
    инициализация ``en_core_web_sm`` занимает несколько сотен миллисекунд
    и существенно сказывается на времени прогона по датасету в 10000
    примеров.

    Пример использования::

        extractor = ClaimExtractor()
        claims = extractor.extract("First for Women was started first.")
        for c in claims:
            print(c.position, c.text)
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        min_words: int = 3,
    ) -> None:
        """Инициализировать экстрактор и загрузить spaCy-модель.

        Параметры:
            model_name: имя spaCy-модели для сегментации.
            min_words: минимально допустимое количество слов в claim;
                более короткие предложения считаются тривиальными.
        """
        logger.info("Загрузка spaCy-модели: %s", model_name)
        self.nlp: Language = spacy.load(model_name)
        self.min_words = min_words

    def extract(self, text: str) -> List[Claim]:
        """Извлечь список атомарных утверждений из текста.

        Параметры:
            text: входной текст (например, ответ LLM).

        Возвращает:
            Список объектов :class:`Claim`. Поле ``position`` отражает
            индекс предложения в исходном тексте; пропущенные индексы
            соответствуют отфильтрованным тривиальным предложениям.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        sentences = list(doc.sents)

        claims: List[Claim] = []
        for idx, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            if self._is_trivial(sent):
                logger.debug("Пропуск тривиального предложения [%d]: %r", idx, sent_text)
                continue
            claims.append(Claim(text=sent_text, position=idx))

        logger.info(
            "Извлечено claims: %d из %d предложений", len(claims), len(sentences)
        )
        return claims

    def _is_trivial(self, sent: Span) -> bool:
        """Определить, является ли предложение тривиальным.

        Возвращает ``True``, если выполнено хотя бы одно из условий:

          * предложение оканчивается на «?» (вопрос);
          * число содержательных токенов меньше ``min_words``;
          * среди токенов нет ни одного существительного (POS = NOUN
            или PROPN), то есть предложение не упоминает никакой
            сущности, относительно которой можно проводить верификацию.
        """
        text = sent.text.strip()

        if text.endswith("?"):
            return True

        # Считаем «словами» токены, не являющиеся пробелами или знаками препинания.
        word_tokens = [t for t in sent if not t.is_space and not t.is_punct]
        if len(word_tokens) < self.min_words:
            return True

        if not any(t.pos_ in ("NOUN", "PROPN") for t in sent):
            return True

        return False
