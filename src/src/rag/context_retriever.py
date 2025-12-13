import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config.settings import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class EmotionContextRetriever:
    """
    A lightweight RAG retriever that provides psychology-backed
    context for emotions predicted by the model.

    Uses:
    - Local vector store (in-memory)
    - OpenAI embeddings API (or MCP-compatible model)
    """

    def __init__(self):
        self.model_name = "text-embedding-3-small"

        # Emotion psychology knowledge base (RAG corpus)
        self.corpus = {
            "happy": "Happiness is associated with joy, comfort, safety, and positive social connection.",
            "sad": "Sadness can reflect emotional pain, loss, disappointment, or a request for empathy.",
            "angry": "Anger may indicate frustration, injustice, boundary violations, or perceived threats.",
            "surprised": "Surprise often comes from unexpected events—positive or negative—and involves high alertness.",
            "fearful": "Fear is a protective emotional response tied to perceived danger or uncertainty.",
            "disgusted": "Disgust is linked to aversion, moral judgment, or sensing something unpleasant.",
            "neutral": "A neutral expression suggests calmness, baseline emotional state, or reflective thinking."
        }

        # Precompute embeddings for RAG corpus
        self.keys = list(self.corpus.keys())
        self.embeddings = self._embed_corpus()

    def _embed_corpus(self):
        """
        Embed the emotion corpus on initialization.
        """
        texts = list(self.corpus.values())

        embeddings = client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        vectors = [e.embedding for e in embeddings.data]
        return np.array(vectors)

    def _embed_query(self, emotions):
        """
        Converts predicted emotions into a single query embedding.
        """
        combined_query = " ".join(emotions)

        result = client.embeddings.create(
            model=self.model_name,
            input=combined_query
        )

        return np.array(result.data[0].embedding)

    def retrieve(self, predicted_emotions):
        """
        Retrieves the most relevant psychology context for the given emotions
        using cosine similarity over embedding vectors.
        """

        if isinstance(predicted_emotions, str):
            predicted_emotions = [predicted_emotions]

        # Embed the predicted emotions
        query_vec = self._embed_query(predicted_emotions)

        # Score each corpus entry
        scores = cosine_similarity([query_vec], self.embeddings)[0]

        # Rank highest scores
        ranked_idx = np.argsort(scores)[::-1]

        retrieved_chunks = []
        for idx in ranked_idx[:3]:
            emotion = self.keys[idx]
            text = self.corpus[emotion]
            retrieved_chunks.append({
                "emotion": emotion,
                "score": float(scores[idx]),
                "text": text
            })

        # Return combined context for LLM
        return json.dumps(retrieved_chunks, indent=4)
