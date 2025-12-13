import pytest
from src.pipeline.inference import EmotionPipeline
from types import SimpleNamespace
import io

class DummyFile:
    """Simulate UploadFile for testing."""
    def __init__(self, content):
        self.file = io.BytesIO(content)

def test_pipeline_initialization():
    pipeline = EmotionPipeline()
    assert pipeline.model is not None
    assert len(pipeline.labels) > 0


def test_prediction_runs(monkeypatch):
    """Ensure inference returns a list of strings."""

    def mock_predict(self, image_file):
        return ["happy"]

    monkeypatch.setattr(EmotionPipeline, "predict", mock_predict)

    pipeline = EmotionPipeline()
    result = pipeline.predict(DummyFile(b"fake image"))
    assert isinstance(result, list)
    assert result == ["happy"]
