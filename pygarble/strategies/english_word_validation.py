import re
from spellchecker import SpellChecker
from .base import BaseStrategy


class EnglishWordValidationStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spell_checker = SpellChecker()

    def _tokenize_text(self, text: str):
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _predict_impl(self, text: str) -> bool:
        proba = self._predict_proba_impl(text)
        return proba >= 0.5

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize_text(text)
        if not words:
            return 0.0

        unknown_words = self.spell_checker.unknown(words)
        invalid_word_count = len(unknown_words)
        total_words = len(words)
        
        return invalid_word_count / total_words
