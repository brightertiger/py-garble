from pygarble import GarbleDetector, Strategy


class TestStrategies:
    def test_character_frequency_detector(self):
        detector = GarbleDetector(
            Strategy.CHARACTER_FREQUENCY, frequency_threshold=0.3
        )
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("normal text") is False

    def test_word_length_detector(self):
        detector = GarbleDetector(
            Strategy.WORD_LENGTH, max_word_length=5, threshold=0.5
        )
        assert detector.predict("supercalifragilisticexpialidocious") is True
        assert detector.predict("short words") is False

    def test_pattern_matching_detector(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        assert detector.predict("AAAAA") is True
        assert detector.predict("normal text") is False

    def test_statistical_analysis_detector(self):
        detector = GarbleDetector(
            Strategy.STATISTICAL_ANALYSIS, alpha_threshold=0.3
        )
        assert detector.predict("123456789") is True
        assert detector.predict("normal text") is False

    def test_entropy_based_detector(self):
        detector = GarbleDetector(
            Strategy.ENTROPY_BASED, entropy_threshold=2.0
        )
        assert detector.predict("aaaaaaa") is True
        assert detector.predict("normal text") is False

    def test_character_frequency_proba(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        proba_high = detector.predict_proba("aaaaaaa")
        proba_low = detector.predict_proba("normal text")
        assert proba_high > proba_low
        assert 0.0 <= proba_high <= 1.0
        assert 0.0 <= proba_low <= 1.0

    def test_word_length_proba(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH, max_word_length=10)
        proba_long = detector.predict_proba(
            "supercalifragilisticexpialidocious"
        )
        proba_short = detector.predict_proba("short words")
        assert proba_long > proba_short
        assert 0.0 <= proba_long <= 1.0
        assert 0.0 <= proba_short <= 1.0

    def test_pattern_matching_proba(self):
        detector = GarbleDetector(Strategy.PATTERN_MATCHING)
        proba_pattern = detector.predict_proba("AAAAA")
        proba_normal = detector.predict_proba("normal text")
        assert proba_pattern > proba_normal
        assert 0.0 <= proba_pattern <= 1.0
        assert 0.0 <= proba_normal <= 1.0

    def test_statistical_analysis_proba(self):
        detector = GarbleDetector(Strategy.STATISTICAL_ANALYSIS)
        proba_numbers = detector.predict_proba("123456789")
        proba_text = detector.predict_proba("normal text")
        assert proba_numbers > proba_text
        assert 0.0 <= proba_numbers <= 1.0
        assert 0.0 <= proba_text <= 1.0

    def test_entropy_based_proba(self):
        detector = GarbleDetector(Strategy.ENTROPY_BASED)
        proba_repeated = detector.predict_proba("aaaaaaa")
        proba_diverse = detector.predict_proba("normal text")
        assert proba_repeated > proba_diverse
        assert 0.0 <= proba_repeated <= 1.0
        assert 0.0 <= proba_diverse <= 1.0


class TestStrategy:
    def test_strategy_enum_values(self):
        assert Strategy.CHARACTER_FREQUENCY.value == "character_frequency"
        assert Strategy.WORD_LENGTH.value == "word_length"
        assert Strategy.PATTERN_MATCHING.value == "pattern_matching"
        assert Strategy.STATISTICAL_ANALYSIS.value == "statistical_analysis"
        assert Strategy.ENTROPY_BASED.value == "entropy_based"
        assert Strategy.LANGUAGE_DETECTION.value == "language_detection"
