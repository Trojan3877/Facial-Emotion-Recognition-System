from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]


def test_rag_endpoint():
    payload = {"emotions": ["happy"]}
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    assert "Happiness" in response.json()["context"]


def test_explain_endpoint(monkeypatch):

    def mock_explain(self, emotions, rag_context):
        return "Mock explanation"

    from src.llm.explainer import EmotionLLMExplainer
    monkeypatch.setattr(EmotionLLMExplainer, "explain_emotions", mock_explain)

    payload = {"emotions": ["sad"]}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    assert response.json()["explanation"] == "Mock explanation"
