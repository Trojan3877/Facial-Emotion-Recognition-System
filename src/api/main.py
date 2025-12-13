from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.pipeline.inference import EmotionPipeline
from src.rag.context_retriever import EmotionContextRetriever
from src.llm.explainer import EmotionLLMExplainer


# Initialize components
pipeline = EmotionPipeline()
rag = EmotionContextRetriever()
llm = EmotionLLMExplainer()


# FastAPI App
app = FastAPI(
    title="Facial Emotion Recognition + LLM + RAG API",
    description="A production-ready API for emotion classification with LLM explanations.",
    version="1.0.0",
)


# CORS Support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Model
class EmotionRequest(BaseModel):
    emotions: list


@app.get("/")
def home():
    return {"status": "online", "message": "Emotion AI API Running Successfully!"}


@app.post("/predict")
async def predict_emotion(image: UploadFile = File(...)):
    """
    Endpoint: Perform CNN-based emotion prediction.
    """
    predictions = pipeline.predict(image)
    return {"predictions": predictions}


@app.post("/rag")
async def retrieve_context(request: EmotionRequest):
    """
    Endpoint: Retrieve psychology-backed context via RAG.
    """
    context = rag.retrieve(request.emotions)
    return {"context": context}


@app.post("/explain")
async def explain_emotions(request: EmotionRequest):
    """
    Endpoint: Generate LLM explanation using predictions + RAG context.
    """
    # Step 1: Retrieve psychology context
    rag_context = rag.retrieve(request.emotions)

    # Step 2: Generate LLM explanation
    explanation = llm.explain_emotions(
        emotions=request.emotions,
        rag_context=rag_context
    )

    return {
        "emotions": request.emotions,
        "context_used": rag_context,
        "explanation": explanation
    }


@app.post("/full-analysis")
async def full_analysis(image: UploadFile = File(...)):
    """
    Full pipeline:
    1. Predict emotion(s)
    2. Retrieve RAG psychology context
    3. Generate LLM explanation
    """
    predictions = pipeline.predict(image)
    
    rag_context = rag.retrieve(predictions)

    explanation = llm.explain_emotions(
        emotions=predictions,
        rag_context=rag_context
    )

    return {
        "predictions": predictions,
        "context_used": rag_context,
        "explanation": explanation
    }


# Run the server (local development)
if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
