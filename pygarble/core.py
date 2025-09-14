import concurrent.futures
from enum import Enum
from typing import List, Union

from .strategies import (
    CharacterFrequencyStrategy,
    EntropyBasedStrategy,
    LanguageDetectionStrategy,
    PatternMatchingStrategy,
    StatisticalAnalysisStrategy,
    WordLengthStrategy,
)


class Strategy(Enum):
    CHARACTER_FREQUENCY = "character_frequency"
    WORD_LENGTH = "word_length"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ENTROPY_BASED = "entropy_based"
    LANGUAGE_DETECTION = "language_detection"


class GarbleDetector:
    def __init__(
        self,
        strategy: Strategy,
        threshold: float = 0.5,
        threads: int = None,
        **kwargs,
    ):
        self.strategy = strategy
        self.threshold = threshold
        self.threads = threads
        self.kwargs = kwargs
        self._strategy_instance = self._create_strategy_instance()

    def _create_strategy_instance(self):
        strategy_map = {
            Strategy.CHARACTER_FREQUENCY: CharacterFrequencyStrategy,
            Strategy.WORD_LENGTH: WordLengthStrategy,
            Strategy.PATTERN_MATCHING: PatternMatchingStrategy,
            Strategy.STATISTICAL_ANALYSIS: StatisticalAnalysisStrategy,
            Strategy.ENTROPY_BASED: EntropyBasedStrategy,
            Strategy.LANGUAGE_DETECTION: LanguageDetectionStrategy,
        }

        if self.strategy not in strategy_map:
            raise NotImplementedError(
                f"Strategy {self.strategy.value} is not implemented"
            )

        strategy_class = strategy_map[self.strategy]
        return strategy_class(**self.kwargs)

    def _process_text_proba(self, text: str) -> float:
        return self._strategy_instance.predict_proba(text)

    def _process_text_predict(self, text: str) -> bool:
        proba = self._strategy_instance.predict_proba(text)
        return proba >= self.threshold

    def _process_batch_threaded(self, texts: List[str], process_func) -> List:
        if self.threads is None or self.threads <= 1 or len(texts) < 10:
            return [process_func(text) for text in texts]

        max_workers = min(self.threads, len(texts))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [executor.submit(process_func, text) for text in texts]
            return [future.result() for future in futures]

    def predict(self, X: Union[str, List[str]]) -> Union[bool, List[bool]]:
        if isinstance(X, str):
            proba = self._strategy_instance.predict_proba(X)
            return proba >= self.threshold
        elif isinstance(X, list):
            return self._process_batch_threaded(X, self._process_text_predict)
        else:
            raise ValueError("Input must be a string or list of strings")

    def predict_proba(
        self, X: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        if isinstance(X, str):
            return self._strategy_instance.predict_proba(X)
        elif isinstance(X, list):
            return self._process_batch_threaded(X, self._process_text_proba)
        else:
            raise ValueError("Input must be a string or list of strings")
