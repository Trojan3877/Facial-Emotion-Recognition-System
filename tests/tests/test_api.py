import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rag_endpoint(client):
    payload = {"emotions": ["happy"]}
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    assert "Happiness" in response.json()["context"]


def test_explain_endpoint(client, monkeypatch):

    def mock_explain(self, emotions, rag_context):
        return "Mock explanation"

    from src.src.llm_explainer.explain import EmotionLLMExplainer
    monkeypatch.setattr(EmotionLLMExplainer, "explain_emotions", mock_explain)

    payload = {"emotions": ["sad"]}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    assert response.json()["explanation"] == "Mock explanation"
