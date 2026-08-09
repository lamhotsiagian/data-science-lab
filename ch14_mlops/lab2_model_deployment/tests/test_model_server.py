import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fastapi.testclient import TestClient
from model_server import app, registry
from train_and_save import train_and_save

@pytest.fixture(autouse=True)
def setup_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_v1.joblib")
    train_and_save("v1", model_path)
    registry.models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry.load_model("v1")
    yield
    if os.path.exists(model_path):
        os.remove(model_path)

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_predict():
    res = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert res.status_code == 200
    assert "prediction" in res.json()

def test_batch_predict():
    res = client.post("/predict/batch", json={"features_list": [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]})
    assert res.status_code == 200
    assert len(res.json()["predictions"]) == 2

def test_invalid_input():
    res = client.post("/predict", json={"features": "bad"})
    assert res.status_code == 422
    
def test_model_info():
    res = client.get("/model/info")
    assert res.status_code == 200
    assert "LogisticRegression" in res.json()["type"]
