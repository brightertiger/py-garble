import os
import tempfile

from pygarble import GarbleDetector, Strategy


class TestLanguageDetection:
    def test_language_detection_strategy(self):
        detector = GarbleDetector(Strategy.LANGUAGE_DETECTION, threshold=0.5)
        result = detector.predict("This is normal English text")
        assert isinstance(result, bool)

    def test_language_detection_proba(self):
        detector = GarbleDetector(Strategy.LANGUAGE_DETECTION, threshold=0.5)
        result = detector.predict_proba("This is normal English text")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_language_detection_with_custom_params(self):
        detector = GarbleDetector(
            Strategy.LANGUAGE_DETECTION, target_language="en", threshold=0.3
        )
        result = detector.predict("Hello world")
        assert isinstance(result, bool)

    def test_language_detection_empty_string(self):
        detector = GarbleDetector(Strategy.LANGUAGE_DETECTION)
        assert detector.predict("") is False
        assert detector.predict_proba("") == 0.0

    def test_language_detection_custom_model_path(self):
        with tempfile.NamedTemporaryFile(
            suffix=".bin", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            detector = GarbleDetector(
                Strategy.LANGUAGE_DETECTION, model_path=tmp_path, threshold=0.5
            )
            assert detector._strategy_instance._model_path == tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_language_detection_custom_model_path_not_found(self):
        with tempfile.NamedTemporaryFile(
            suffix=".bin", delete=True
        ) as tmp_file:
            tmp_path = tmp_file.name

        detector = GarbleDetector(
            Strategy.LANGUAGE_DETECTION, model_path=tmp_path, threshold=0.5
        )

        import pytest

        with pytest.raises(FileNotFoundError):
            detector.predict("test text")
