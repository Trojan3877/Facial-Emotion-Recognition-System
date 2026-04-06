import torch
from src.model import EmotionCNN  # noqa: E402

def test_model_forward_pass():
    model = EmotionCNN()
    dummy = torch.randn(1, 1, 48, 48)  # grayscale (1 channel)
    out = model(dummy)

    assert out.shape[1] == 7, "❌ Output must have 7 emotion classes"
