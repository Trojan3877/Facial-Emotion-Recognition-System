"""
================================================================================
FACIAL EMOTION RECOGNITION — MODEL SHIM

Re-exports model classes from src.src.modeling.model so that callers can use
the shorter `from src.model import ...` path.
================================================================================
"""

from src.src.modeling.model import (  # noqa: F401
    EmotionCNN,
    ResNetEmotion,
    EmotionModel,
    EMOTION_LABELS,
)
