from src.rag.context_retriever import EmotionContextRetriever

def test_rag_returns_context():
    rag = EmotionContextRetriever()
    result = rag.retrieve(["happy"])
    assert "Happiness" in result


def test_rag_handles_unknown_emotion():
    rag = EmotionContextRetriever()
    result = rag.retrieve(["unknown_emotion"])
    assert "No psychology context available" in result
