import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = "artifacts/models/emotion_model.pt"
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    VECTOR_STORE_PATH = "artifacts/vector_store/faiss_index"
    RAG_TOP_K = 3
    DECISION_THRESHOLD = 0.6

    # API  
    HOST = "0.0.0.0"
    PORT = 8000

    # OpenAI / Llama
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

settings = Settings()
