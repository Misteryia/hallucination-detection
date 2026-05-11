"""
Модуль FactChecker.

Ядро верификации утверждений (claims) относительно базы знаний (KB).
Реализованы два независимых метода и их комбинация:

  * **NLI (Natural Language Inference).** Для каждой пары
    «факт из KB → утверждение» предобученная модель оценивает
    вероятности трёх классов: contradiction, neutral, entailment.
    Утверждение помечается как галлюцинация, если максимальная по KB
    вероятность contradiction превышает порог ``nli_threshold``.

  * **Embedding similarity.** Для утверждения и каждого факта
    вычисляются векторные представления (sentence embeddings) и
    косинусное сходство. Если максимальное сходство ниже порога
    ``similarity_threshold``, утверждение считается не подтверждённым
    базой знаний и помечается как галлюцинация.

  * **Combined.** Объединение вердиктов двух методов:
    режим ``"or"`` — рекоориентир (галлюцинация, если хотя бы один
    метод её обнаружил); режим ``"and"`` — точность-ориентир
    (галлюцинация только при согласии обоих методов).

Тяжёлые модели (``transformers`` для NLI и ``sentence-transformers`` для
эмбеддингов) загружаются **лениво**: первая загрузка происходит при
первом вызове соответствующего публичного метода. Это позволяет
использовать только нужный метод без расхода памяти и времени на
загрузку второй модели.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .claim_extractor import Claim

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    """Вердикт по одному утверждению.

    Атрибуты:
        claim: исходный текст утверждения.
        method: имя метода — ``"nli"``, ``"embedding"`` либо
            ``"combined"``.
        is_hallucination: итоговое решение (True — обнаружена
            галлюцинация).
        confidence: уверенность в принятом решении в диапазоне [0, 1].
        matched_fact: факт из KB, наиболее сильно подтверждающий или
            опровергающий утверждение (для NLI — с максимальной
            contradiction; для embedding — с максимальным cosine
            similarity).
        details: словарь с подробной диагностикой (вероятности классов
            NLI, значения similarity, использованные пороги и т. п.).
            Используется в отчётах и при отладке.
    """

    claim: str
    method: str
    is_hallucination: bool
    confidence: float
    matched_fact: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def _select_device(requested: str) -> str:
    """Выбрать устройство выполнения с учётом фактической поддержки.

    При ``requested == "auto"`` возвращается ``"cuda"``, если CUDA
    доступна **и** compute capability обнаруженного GPU входит в список
    архитектур, поддерживаемых текущей сборкой PyTorch. В противном
    случае возвращается ``"cpu"``. Эта проверка нужна потому, что
    ``torch.cuda.is_available()`` возвращает ``True`` и для устройств,
    которые конкретный билд PyTorch на самом деле не поддерживает
    (например, GTX 10xx серии для современных бинарников); реальный
    forward в такой ситуации падает в рантайме.
    """
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        cc_tag = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()
        if cc_tag in supported:
            return "cuda"
        logger.warning(
            "GPU %s имеет compute capability %s, не поддерживаемую сборкой "
            "PyTorch (%s). Откат на CPU.",
            torch.cuda.get_device_name(0), cc_tag, supported,
        )
        return "cpu"
    except Exception as exc:  # на случай нестандартных драйверов
        logger.warning("Ошибка определения GPU (%s); откат на CPU.", exc)
        return "cpu"


class FactChecker:
    """Верификатор утверждений по двум методам (NLI и embedding similarity).

    В качестве NLI-модели по умолчанию используется
    ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``. Выбор
    обоснован следующим:

      * **Публичность.** Модель свободно доступна на HuggingFace Hub и
        не требует авторизации (в отличие от gated-репозитория
        ``microsoft/deberta-v3-large-mnli``).
      * **Релевантность задаче.** Помимо MNLI, модель дообучена на
        FEVER, ANLI, LingNLI и WANLI. FEVER особенно релевантен —
        этот датасет состоит из утверждений, проверяемых по
        Википедии, что напрямую совпадает с задачей детектирования
        галлюцинаций относительно базы знаний на основе Википедии
        (HaluEval QA).
      * **Совместимый размер.** Архитектура и число параметров
        совпадают с оригинальной DeBERTa-v3-large, поэтому
        характеристики по скорости и памяти идентичны.

    Параметры конструктора:
        nli_model: имя или путь HuggingFace-модели для NLI.
        embedding_model: имя sentence-transformers-модели.
        nli_threshold: порог вероятности contradiction, выше которого
            утверждение считается галлюцинацией.
        similarity_threshold: порог максимального cosine similarity,
            ниже которого утверждение считается неподтверждённым.
        device: ``"cpu"``, ``"cuda"`` или ``"auto"`` (выбор по
            доступности и совместимости).
        default_method: метод верификации, используемый методом
            :meth:`check` и считающийся «дефолтным» для этой утилиты.
            По результатам пилотного grid search на 50 примерах
            HaluEval одиночный NLI даёт лучший Q_prod (~0.69) и
            опережает гибридные режимы OR/AND, поэтому дефолтом
            принят ``"nli"``. Дефолтные пороги
            (``nli_threshold=0.5``, ``similarity_threshold=0.6``)
            подобраны как разумные нейтральные значения для
            интерактивных запусков ``haludetect check``;
            эмпирически оптимальные пороги для оценки на полной
            выборке HaluEval-QA подбираются через
            ``scripts/tune_thresholds.py``.

    Модели загружаются лениво — фактическое скачивание весов и
    инициализация происходят при первом вызове ``check_nli`` /
    ``check_embedding`` (или ``check_batch``).
    """

    def __init__(
        self,
        nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        nli_threshold: float = 0.5,
        similarity_threshold: float = 0.6,
        device: str = "auto",
        default_method: str = "nli",
    ) -> None:
        if default_method not in ("nli", "embedding", "combined"):
            raise ValueError(
                f"default_method должен быть 'nli', 'embedding' или 'combined', "
                f"получено {default_method!r}"
            )

        self.nli_model_name = nli_model
        self.embedding_model_name = embedding_model
        self.nli_threshold = float(nli_threshold)
        self.similarity_threshold = float(similarity_threshold)

        # Дефолтный метод выбран эмпирически: пилотный grid search
        # на 50 примерах HaluEval показал, что одиночный NLI с порогом
        # 0.5 даёт Q_prod ≈ 0.69, тогда как гибриды OR/AND и
        # embedding-only стабильно хуже на этой подвыборке.
        self.default_method = default_method

        self.device = _select_device(device)
        logger.info(
            "FactChecker: устройство — %s, метод по умолчанию — %s",
            self.device, self.default_method,
        )

        # Слоты для лениво загружаемых артефактов.
        self._nli_tokenizer = None
        self._nli_model = None
        self._nli_idx: Dict[str, int] = {}
        self._embedding_model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Ленивая загрузка моделей
    # ------------------------------------------------------------------

    def _load_nli(self) -> None:
        """Загрузить NLI-модель и токенизатор (один раз)."""
        if self._nli_model is not None:
            return

        logger.info("Загрузка NLI-модели: %s", self.nli_model_name)
        t0 = time.perf_counter()
        self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name
        ).to(self.device)
        self._nli_model.eval()
        logger.info(
            "NLI-модель загружена за %.2f с (%s)",
            time.perf_counter() - t0, self.nli_model_name,
        )

        # Определяем индексы классов по конфигу модели — порядок
        # contradiction/neutral/entailment может отличаться у разных
        # чекпойнтов, поэтому полагаться на жёсткий 0/1/2 нельзя.
        id2label = self._nli_model.config.id2label
        normalized = {label.upper(): idx for idx, label in id2label.items()}

        def _find(name: str) -> int:
            if name in normalized:
                return normalized[name]
            for k, v in normalized.items():
                if name in k:
                    return v
            raise RuntimeError(
                f"В id2label NLI-модели не найден класс {name!r}: {id2label!r}"
            )

        self._nli_idx = {
            "contradiction": _find("CONTRADICTION"),
            "neutral": _find("NEUTRAL"),
            "entailment": _find("ENTAILMENT"),
        }
        logger.info("Соответствие классов NLI: %s", self._nli_idx)

    def _load_embedding(self) -> None:
        """Загрузить модель sentence-эмбеддингов (один раз)."""
        if self._embedding_model is not None:
            return

        logger.info("Загрузка эмбеддера: %s", self.embedding_model_name)
        t0 = time.perf_counter()
        self._embedding_model = SentenceTransformer(
            self.embedding_model_name, device=self.device
        )
        logger.info(
            "Эмбеддер загружен за %.2f с (%s)",
            time.perf_counter() - t0, self.embedding_model_name,
        )

    # ------------------------------------------------------------------
    # Публичные методы верификации
    # ------------------------------------------------------------------

    def check_nli(self, claim: str, facts: List[str]) -> Verdict:
        """Проверить утверждение методом NLI.

        Для каждого факта из KB модель оценивает вероятность класса
        contradiction в паре (premise=fact, hypothesis=claim). Если
        максимум по KB превышает ``nli_threshold``, утверждение
        классифицируется как галлюцинация.
        """
        self._load_nli()
        return self._check_nli_internal(claim, facts, batch_size=8)

    def check_embedding(self, claim: str, facts: List[str]) -> Verdict:
        """Проверить утверждение методом cosine similarity эмбеддингов.

        Если максимум cosine similarity между утверждением и фактами KB
        ниже ``similarity_threshold``, утверждение считается
        неподтверждённым и классифицируется как галлюцинация.
        """
        self._load_embedding()
        if not facts:
            return self._empty_kb_verdict(claim, "embedding")
        fact_embeddings = self._embedding_model.encode(
            facts, convert_to_tensor=True, show_progress_bar=False
        )
        return self._check_embedding_internal(claim, facts, fact_embeddings)

    def score_nli(self, claim: str, facts: List[str]) -> Tuple[float, str]:
        """Вернуть «сырой» NLI-скор без применения порога.

        Вычисляет максимальную по базе знаний вероятность класса
        contradiction в парах (premise=fact, hypothesis=claim) и
        возвращает её вместе с фактом, на котором достигнут максимум.
        Полезно для grid search по порогу: сырые скоры считаются один
        раз, а пороги перебираются в Python без повторного запуска
        тяжёлой модели.

        Параметры:
            claim: проверяемое утверждение.
            facts: список фактов из базы знаний.

        Возвращает:
            Кортеж ``(max_contradiction_prob, matched_fact)``. Для
            пустого списка фактов возвращает ``(0.0, "")``.
        """
        self._load_nli()
        if not facts:
            return (0.0, "")
        probs = self._compute_nli_probs(claim, facts, batch_size=8)
        return (probs["max_contradiction"], facts[probs["max_idx"]])

    def score_embedding(self, claim: str, facts: List[str]) -> Tuple[float, str]:
        """Вернуть «сырой» embedding-скор без применения порога.

        Вычисляет максимум cosine similarity между утверждением и
        фактами KB и возвращает его вместе с наиболее похожим фактом.
        Семантика обратна NLI-скору: высокое значение — утверждение
        подтверждается базой знаний.

        Возвращает:
            Кортеж ``(max_cosine_similarity, matched_fact)``. Для
            пустого списка фактов возвращает ``(0.0, "")``.
        """
        self._load_embedding()
        if not facts:
            return (0.0, "")
        fact_embeddings = self._embedding_model.encode(
            facts, convert_to_tensor=True, show_progress_bar=False
        )
        sims = self._compute_embedding_sims(claim, fact_embeddings)
        return (sims["max_similarity"], facts[sims["max_idx"]])

    def check_combined(
        self, claim: str, facts: List[str], mode: str = "or"
    ) -> Verdict:
        """Объединить вердикты двух методов.

        Режим ``"or"`` (по умолчанию) ориентирован на recall —
        утверждение помечается как галлюцинация, если её обнаружил хотя
        бы один метод. Режим ``"and"`` ориентирован на precision —
        галлюцинация фиксируется только при согласии обоих методов.
        """
        nli_v = self.check_nli(claim, facts)
        emb_v = self.check_embedding(claim, facts)
        return self._combine_verdicts(claim, nli_v, emb_v, mode)

    def check(self, claim: str, facts: List[str]) -> Verdict:
        """Проверить утверждение методом по умолчанию (``self.default_method``).

        Удобный диспетчер: позволяет коду, не желающему явно выбирать
        метод, опираться на эмпирически лучший вариант, зафиксированный
        в конструкторе (по умолчанию — ``"nli"``).
        """
        if self.default_method == "nli":
            return self.check_nli(claim, facts)
        if self.default_method == "embedding":
            return self.check_embedding(claim, facts)
        return self.check_combined(claim, facts)

    def check_batch(
        self,
        claims: List[Claim],
        facts: List[str],
        method: str = "combined",
        mode: str = "or",
        batch_size: int = 8,
    ) -> List[Verdict]:
        """Пакетная проверка списка утверждений.

        Для метода ``"embedding"`` (и ``"combined"``) векторы фактов
        вычисляются один раз перед циклом и переиспользуются для всех
        утверждений; для ``"nli"`` пары (fact, claim) объединяются в
        батчи размера ``batch_size``. Прогресс отображается через
        ``tqdm``.
        """
        if method not in ("nli", "embedding", "combined"):
            raise ValueError(f"Неизвестный метод: {method!r}")
        if mode not in ("or", "and"):
            raise ValueError(f"Неизвестный режим объединения: {mode!r}")

        if method in ("nli", "combined"):
            self._load_nli()
        if method in ("embedding", "combined"):
            self._load_embedding()

        fact_embeddings = None
        if method in ("embedding", "combined") and facts:
            fact_embeddings = self._embedding_model.encode(
                facts, convert_to_tensor=True, show_progress_bar=False
            )

        verdicts: List[Verdict] = []
        for claim in tqdm(claims, desc=f"FactChecker[{method}]"):
            text = claim.text if isinstance(claim, Claim) else str(claim)

            if method == "nli":
                v = self._check_nli_internal(text, facts, batch_size=batch_size)
            elif method == "embedding":
                v = (
                    self._check_embedding_internal(text, facts, fact_embeddings)
                    if facts else self._empty_kb_verdict(text, "embedding")
                )
            else:  # combined
                nli_v = self._check_nli_internal(text, facts, batch_size=batch_size)
                emb_v = (
                    self._check_embedding_internal(text, facts, fact_embeddings)
                    if facts else self._empty_kb_verdict(text, "embedding")
                )
                v = self._combine_verdicts(text, nli_v, emb_v, mode)

            verdicts.append(v)
        return verdicts

    # ------------------------------------------------------------------
    # Внутренние реализации
    # ------------------------------------------------------------------

    def _compute_nli_probs(
        self, claim: str, facts: List[str], batch_size: int = 8
    ) -> Dict[str, Any]:
        """Низкоуровневый прогон NLI: forward pass без применения порога.

        Возвращает словарь с массивами вероятностей по трём классам,
        индексом и значением максимальной contradiction. Используется
        и для построения :class:`Verdict`, и для сырого скоринга в
        ``score_nli`` — это позволяет не дублировать тяжёлую часть.
        """
        contradiction_idx = self._nli_idx["contradiction"]
        entailment_idx = self._nli_idx["entailment"]
        neutral_idx = self._nli_idx["neutral"]

        contradiction_probs: List[float] = []
        entailment_probs: List[float] = []
        neutral_probs: List[float] = []

        with torch.no_grad():
            for start in range(0, len(facts), batch_size):
                batch_facts = facts[start:start + batch_size]
                # Пара передаётся как (premise=fact, hypothesis=claim).
                inputs = self._nli_tokenizer(
                    batch_facts,
                    [claim] * len(batch_facts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                logits = self._nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                contradiction_probs.extend(probs[:, contradiction_idx].cpu().tolist())
                entailment_probs.extend(probs[:, entailment_idx].cpu().tolist())
                neutral_probs.extend(probs[:, neutral_idx].cpu().tolist())

        max_idx = max(
            range(len(contradiction_probs)), key=lambda i: contradiction_probs[i]
        )
        return {
            "contradiction_probs": contradiction_probs,
            "entailment_probs": entailment_probs,
            "neutral_probs": neutral_probs,
            "max_idx": max_idx,
            "max_contradiction": float(contradiction_probs[max_idx]),
        }

    def _compute_embedding_sims(
        self, claim: str, fact_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """Низкоуровневый расчёт cosine similarity без применения порога."""
        claim_emb = self._embedding_model.encode(
            claim, convert_to_tensor=True, show_progress_bar=False
        )
        sims = util.cos_sim(claim_emb, fact_embeddings)[0]
        max_idx = int(torch.argmax(sims).item())
        return {
            "similarities": [float(s) for s in sims.cpu().tolist()],
            "max_idx": max_idx,
            "max_similarity": float(sims[max_idx].item()),
        }

    def _check_nli_internal(
        self, claim: str, facts: List[str], batch_size: int = 8
    ) -> Verdict:
        if not facts:
            return self._empty_kb_verdict(claim, "nli")

        s = self._compute_nli_probs(claim, facts, batch_size=batch_size)
        max_contra = s["max_contradiction"]
        is_h = max_contra > self.nli_threshold
        confidence = max_contra if is_h else 1.0 - max_contra

        return Verdict(
            claim=claim,
            method="nli",
            is_hallucination=is_h,
            confidence=float(confidence),
            matched_fact=facts[s["max_idx"]],
            details={
                "max_contradiction_prob": max_contra,
                "contradiction_probs": s["contradiction_probs"],
                "entailment_probs": s["entailment_probs"],
                "neutral_probs": s["neutral_probs"],
                "threshold": self.nli_threshold,
            },
        )

    def _check_embedding_internal(
        self,
        claim: str,
        facts: List[str],
        fact_embeddings: torch.Tensor,
    ) -> Verdict:
        s = self._compute_embedding_sims(claim, fact_embeddings)
        max_sim = s["max_similarity"]

        is_h = max_sim < self.similarity_threshold
        confidence = max_sim if not is_h else 1.0 - max_sim
        confidence = max(0.0, min(1.0, confidence))

        return Verdict(
            claim=claim,
            method="embedding",
            is_hallucination=is_h,
            confidence=float(confidence),
            matched_fact=facts[s["max_idx"]],
            details={
                "max_similarity": max_sim,
                "similarities": s["similarities"],
                "threshold": self.similarity_threshold,
            },
        )

    def _combine_verdicts(
        self, claim: str, nli_v: Verdict, emb_v: Verdict, mode: str
    ) -> Verdict:
        if mode == "or":
            is_h = nli_v.is_hallucination or emb_v.is_hallucination
        elif mode == "and":
            is_h = nli_v.is_hallucination and emb_v.is_hallucination
        else:
            raise ValueError(f"Неизвестный режим объединения: {mode!r}")

        # Усреднённая уверенность как простой и понятный для записки
        # вариант. При желании можно заменить на min/max в зависимости
        # от режима, но среднее даёт стабильно интерпретируемое число.
        confidence = (nli_v.confidence + emb_v.confidence) / 2.0

        matched_fact = (
            nli_v.matched_fact
            if nli_v.confidence >= emb_v.confidence
            else emb_v.matched_fact
        )

        return Verdict(
            claim=claim,
            method="combined",
            is_hallucination=is_h,
            confidence=float(confidence),
            matched_fact=matched_fact,
            details={
                "mode": mode,
                "nli": {
                    "is_hallucination": nli_v.is_hallucination,
                    "confidence": nli_v.confidence,
                    "matched_fact": nli_v.matched_fact,
                },
                "embedding": {
                    "is_hallucination": emb_v.is_hallucination,
                    "confidence": emb_v.confidence,
                    "matched_fact": emb_v.matched_fact,
                },
            },
        )

    @staticmethod
    def _empty_kb_verdict(claim: str, method: str) -> Verdict:
        """Сформировать вердикт при пустой базе знаний.

        Без фактов любое утверждение неподтверждаемо, поэтому помечаем
        его как галлюцинацию с максимальной уверенностью; одновременно
        фиксируем причину в ``details`` для прозрачности.
        """
        return Verdict(
            claim=claim,
            method=method,
            is_hallucination=True,
            confidence=1.0,
            matched_fact=None,
            details={"reason": "empty knowledge base"},
        )
