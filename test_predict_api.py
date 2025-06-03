import pytest
from fastapi.testclient import TestClient
from mlFlow_experiment import app

client = TestClient(app)

def test_predict_route():
    # Exemple de features (adapter selon le modèle attendu)
    features = [30, 175, 70, 2500, "M", "Oui", "Bac+2", "Ile-de-France", "Non", "Oui"]
    response = client.post("/predict", json={"data": features})
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)
