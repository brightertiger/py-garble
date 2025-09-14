import re
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        if self._is_extremely_long_string(text):
            return True

        return self._predict_impl(text)

    def predict_proba(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        if self._is_extremely_long_string(text):
            return 1.0

        return self._predict_proba_impl(text)

    def _is_extremely_long_string(self, text: str) -> bool:
        max_length = self.kwargs.get("max_string_length", 1000)
        return len(text) > max_length and not re.search(r"\s", text)

    @abstractmethod
    def _predict_impl(self, text: str) -> bool:
        pass

    @abstractmethod
    def _predict_proba_impl(self, text: str) -> float:
        pass
