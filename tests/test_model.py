"""
===============================================================================
UNIT TESTS — FACIAL EMOTION RECOGNITION (L6 STANDARD)

Tests:
    - Model forward pass output shape
    - EmotionModel wrapper loading
    - Inference output structure
    - Dataset output format
===============================================================================
"""

import pytest
import torch
import numpy as np

from src.model import EmotionCNN, EmotionModel
from src.dataset import FERDataset


# ==============================================================================
# MODEL FORWARD PASS
# ==============================================================================

def test_emotion_cnn_forward_pass():
    model = EmotionCNN()
    dummy_input = torch.randn(1, 1, 48, 48)
    output = model(dummy_input)

    assert output.shape == (1, 7)


# ==============================================================================
# INFERENCE WRAPPER STRUCTURE
# ==============================================================================

def test_emotion_model_output_structure(tmp_path):
    """
    Tests that EmotionModel returns structured output.
    """

    # Create temporary dummy weights file
    model = EmotionCNN()
    dummy_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), dummy_path)

    wrapper = EmotionModel(model_path=str(dummy_path))
    dummy_input = torch.randn(1, 1, 48, 48)

    result = wrapper.predict(dummy_input)

    assert isinstance(result, dict)
    assert "emotion" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


# ==============================================================================
# DATASET OUTPUT FORMAT
# ==============================================================================

def test_dataset_output_shape():

    import pandas as pd

    # Create minimal fake dataframe
    df = pd.DataFrame({
        "pixels": [" ".join(["0"] * (48*48))],
        "emotion": [0]
    })

    dataset = FERDataset(df)

    image, label = dataset[0]

    assert image.shape == (1, 48, 48)
    assert isinstance(label.item(), int)


# ==============================================================================
# INVALID INPUT TEST
# ==============================================================================

def test_invalid_input_shape():

    model = EmotionCNN()

    with pytest.raises(Exception):
        model(torch.randn(1, 48, 48))  # Missing channel dimension
