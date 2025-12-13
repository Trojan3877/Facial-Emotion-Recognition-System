from src.llm.explainer import EmotionLLMExplainer

def test_llm_explanation(monkeypatch):

    def mock_explain(self, emotions, rag_context):
        return "Mock explanation"

    monkeypatch.setattr(EmotionLLMExplainer, "explain_emotions", mock_explain)

    explainer = EmotionLLMExplainer()
    result = explainer.explain_emotions(["sad"], "context")
    assert result == "Mock explanation"
