"""
===============================================================================
FACIAL EMOTION RECOGNITION — CONFIGURATION (CV SYSTEM ONLY)

Purpose:
    Centralized configuration for model artifacts and inference parameters.

Design Principles:
    - Single source of truth
    - Environment variable overrides supported
    - No unrelated ML/LLM/RAG configuration
    - Clean separation of concerns

Future Improvements:
    - Replace static class with Pydantic BaseSettings for validation
===============================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --------------------------------------------------------------------------
    # MODEL ARTIFACTS
    # --------------------------------------------------------------------------
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "artifacts/models/emotion_model.pt"
    )

    DECISION_THRESHOLD: float = float(
        os.getenv("DECISION_THRESHOLD", 0.6)
    )

    # --------------------------------------------------------------------------
    # API SERVER CONFIG
    # --------------------------------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))


settings = Settings()
