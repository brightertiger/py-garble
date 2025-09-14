import math

from .base import BaseStrategy


class EntropyBasedStrategy(BaseStrategy):
    def _get_char_counts(self, text: str):
        char_counts = {}
        for char in text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        return char_counts

    def _calculate_entropy(self, char_counts):
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        entropy = 0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy

    def _predict_impl(self, text: str) -> bool:
        char_counts = self._get_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return False

        entropy = self._calculate_entropy(char_counts)
        threshold = self.kwargs.get("entropy_threshold", 3.0)

        return entropy < threshold

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        entropy = self._calculate_entropy(char_counts)
        max_entropy = math.log2(len(char_counts)) if char_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return 1.0 - normalized_entropy
