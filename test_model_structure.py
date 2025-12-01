import torch
from model import EmotionCNN  # adjust to your model filename

def test_model_forward_pass():
    model = EmotionCNN()
    dummy = torch.randn(1, 3, 48, 48)
    out = model(dummy)

    assert out.shape[1] == 7, "❌ Output must have 7 emotion classes"
