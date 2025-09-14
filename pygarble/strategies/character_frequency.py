from .base import BaseStrategy


class CharacterFrequencyStrategy(BaseStrategy):
    def _get_char_counts(self, text: str):
        char_counts = {}
        for char in text.lower():
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        return char_counts

    def _predict_impl(self, text: str) -> bool:
        char_counts = self._get_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return False

        threshold = self.kwargs.get("frequency_threshold", 0.1)
        for count in char_counts.values():
            if count / total_chars > threshold:
                return True
        return False

    def _predict_proba_impl(self, text: str) -> float:
        char_counts = self._get_char_counts(text)
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return 0.0

        max_frequency = (
            max(char_counts.values()) / total_chars if char_counts else 0.0
        )
        return min(max_frequency, 1.0)
