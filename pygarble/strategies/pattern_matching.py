import re

from .base import BaseStrategy


class PatternMatchingStrategy(BaseStrategy):
    DEFAULT_PATTERNS = {
        "special_chars": r"[^a-zA-Z0-9\s]{3,}",
        "repeated_chars": r"(.)\1{4,}",
        "uppercase_sequence": r"[A-Z]{5,}",
        "long_numbers": r"[0-9]{8,}",
    }

    def _get_patterns(self):
        custom_patterns = self.kwargs.get("patterns", {})
        override_defaults = self.kwargs.get("override_defaults", False)

        if override_defaults:
            patterns = custom_patterns
        else:
            patterns = {**self.DEFAULT_PATTERNS, **custom_patterns}

        return patterns

    def _predict_impl(self, text: str) -> bool:
        patterns = self._get_patterns()

        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, text):
                return True
        return False

    def _predict_proba_impl(self, text: str) -> float:
        patterns = self._get_patterns()

        if not patterns:
            return 0.0

        matches = 0
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, text):
                matches += 1

        return min(matches / len(patterns), 1.0)
