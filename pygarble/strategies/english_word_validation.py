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
        words = self._tokenize_text(text)
        if not words:
            return False

        unknown_words = self.spell_checker.unknown(words)
        valid_word_count = len(words) - len(unknown_words)
        total_words = len(words)
        
        threshold = self.kwargs.get("valid_word_threshold", 0.7)
        return (valid_word_count / total_words) >= threshold

    def _predict_proba_impl(self, text: str) -> float:
        words = self._tokenize_text(text)
        if not words:
            return 0.0

        unknown_words = self.spell_checker.unknown(words)
        valid_word_count = len(words) - len(unknown_words)
        total_words = len(words)
        
        return valid_word_count / total_words
