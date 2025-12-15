from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"features": [1,2,3]})
    assert response.status_code == 200
