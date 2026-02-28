"""
===============================================================================
FACIAL EMOTION RECOGNITION — CONFIGURATION SETTINGS (L6 STANDARD)

Purpose:
    Centralized configuration for model paths, inference thresholds,
    and deployment parameters.

Design Principles:
    - Single source of truth
    - No hardcoded logic inside model files
    - Environment-driven configuration when appropriate
    - Separation between CV and optional LLM modules

Future Improvements:
    - Replace static class with Pydantic BaseSettings
    - Add environment validation on startup
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

    # --------------------------------------------------------------------------
    # OPTIONAL LLM / RAG MODULES (ONLY IF USED)
    # --------------------------------------------------------------------------
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "gpt-4o-mini"
    )

    VECTOR_STORE_PATH: str = os.getenv(
        "VECTOR_STORE_PATH",
        "artifacts/vector_store/faiss_index"
    )

    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", 3))


settings = Settings()
