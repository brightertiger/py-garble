from pygarble import GarbleDetector, Strategy


class TestEdgeCases:
    def test_predict_proba_empty_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        assert detector.predict_proba("") == 0.0

    def test_predict_proba_whitespace_only(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH)
        assert detector.predict_proba("   ") == 0.0

    def test_empty_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        assert detector.predict("") is False

    def test_whitespace_only(self):
        detector = GarbleDetector(Strategy.WORD_LENGTH)
        assert detector.predict("   ") is False

    def test_extremely_long_string_no_spaces(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "a" * 1001
        assert detector.predict(long_string) is True
        assert detector.predict_proba(long_string) == 1.0

    def test_extremely_long_string_with_spaces(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = (
            "This is a normal sentence with spaces. " * 30
        )  # Long but has spaces
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_tabs(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word\t" * 250  # Long string with tabs
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_newlines(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word\n" * 250  # Long string with newlines
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_with_mixed_whitespace(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        long_string = "word \t\n " * 200  # Long string with mixed whitespace
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_extremely_long_string_custom_threshold(self):
        detector = GarbleDetector(
            Strategy.CHARACTER_FREQUENCY, max_string_length=500
        )
        long_string = "a" * 501
        assert detector.predict(long_string) is True
        assert detector.predict_proba(long_string) == 1.0

    def test_extremely_long_string_below_threshold(self):
        detector = GarbleDetector(
            Strategy.STATISTICAL_ANALYSIS, max_string_length=2000
        )
        long_string = "a" * 1001
        assert detector.predict(long_string) is False
        assert detector.predict_proba(long_string) < 1.0

    def test_base64_like_string(self):
        detector = GarbleDetector(Strategy.CHARACTER_FREQUENCY)
        base64_like = (
            "SGVsbG9Xb3JsZEhlbGxvV29ybGRIZWxsb1dvcmxk" * 50
        )  # Long base64-like string
        assert detector.predict(base64_like) is True
        assert detector.predict_proba(base64_like) == 1.0

    def test_english_word_validation_edge_cases(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0
        
        assert detector.predict("   ") is False
        assert detector.predict_proba("   ") == 0.0
        
        assert detector.predict("123") is False
        assert detector.predict_proba("123") == 0.0
        
        assert detector.predict("!@#") is False
        assert detector.predict_proba("!@#") == 0.0

    def test_english_word_validation_long_valid_text(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        long_valid_text = "This is a very long sentence with many valid English words that should be recognized by the spell checker and classified as not garbled text because it contains proper English vocabulary throughout the entire string"
        assert detector.predict(long_valid_text) is True
        assert detector.predict_proba(long_valid_text) > 0.8

    def test_english_word_validation_long_garbled_text(self):
        detector = GarbleDetector(Strategy.ENGLISH_WORD_VALIDATION)
        long_garbled_text = "asdfghjkl qwertyuiop zxcvbnm asdfghjkl qwertyuiop zxcvbnm asdfghjkl qwertyuiop zxcvbnm asdfghjkl qwertyuiop zxcvbnm"
        assert detector.predict(long_garbled_text) is False
        assert detector.predict_proba(long_garbled_text) < 0.3
