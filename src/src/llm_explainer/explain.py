import json
from src.rag.context_retriever import EmotionContextRetriever
from src.inference.predictor import EmotionPredictor
from src.config.settings import settings

# MCP-compatible OpenAI client
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class LLMEmotionExplainer:
    """
    Generates psychology-backed explanations of detected emotions.
    Uses:
    - Emotion predictions from the CNN/ResNet model
    - RAG knowledge retrieval
    - An LLM for final synthesis
    """

    def __init__(self, model_name="gpt-4o-mini"):
        self.predictor = EmotionPredictor(use_resnet=settings.USE_RESNET)
        self.retriever = EmotionContextRetriever()
        self.model_name = model_name

    def generate_explanation(self, image_array):
        """
        Full pipeline:
        1. Run local model inference
        2. Retrieve psychology-based RAG context
        3. Ask LLM for a structured emotional analysis
        """

        prediction = self.predictor.predict_and_format_for_llm(image_array)

        if "error" in prediction:
            return prediction

        emotions = prediction["emotions"]
        confidences = prediction["confidences"]
        summary = prediction["summary"]

        # Retrieve contextual psychology info
        rag_context = self.retriever.retrieve(emotions)

        # Construct prompt
        prompt = f"""
You are an AI Emotion Specialist.

The user has uploaded an image. A local model detected the following emotions:
Emotions: {emotions}
Confidences: {confidences}
Summary: {summary}

Use the following RAG context to improve your explanation:
{rag_context}

Return a JSON dictionary with:
- "emotion_breakdown": explanation of each emotion
- "likely_cause": what may cause these emotional expressions
- "confidence_interpretation": meaning of confidence scores
- "final_summary": 3–5 sentence summary in plain English
"""

        # Call the LLM
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You specialize in emotion psychology and safe AI reasoning."},
                {"role": "user", "content": prompt}
            ]
        )

        llm_output = response.choices[0].message.content

        try:
            parsed = json.loads(llm_output)
        except json.JSONDecodeError:
            # fallback if LLM returns plain text
            parsed = {"llm_raw_output": llm_output}

        return {
            "predictions": {
                "emotions": emotions,
                "confidences": confidences,
                "summary": summary,
            },
            "rag_context": rag_context,
            "llm_explanation": parsed
        }
