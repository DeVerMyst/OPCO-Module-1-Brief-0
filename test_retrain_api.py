import pytest
from fastapi.testclient import TestClient
from mlFlow_experiment import app

client = TestClient(app)

def test_retrain_route():
    # Exemple minimal de données valides pour le réentraînement (avec toutes les colonnes attendues)
    payload = {
        "data": [
            {
                "nom": "Dupont",
                "prenom": "Jean",
                "age": 35,
                "taille": 175,
                "poids": 70,
                "revenu_estime_mois": 2500,
                "sexe": "M",
                "sport_licence": "non",
                "niveau_etude": "bac+2",
                "region": "Île-de-France",
                "smoker": "non",
                "nationalité_francaise": "oui",
                "montant_pret": 10000
            },
            {
                "nom": "Martin",
                "prenom": "Claire",
                "age": 42,
                "taille": 168,
                "poids": 80,
                "revenu_estime_mois": 3200,
                "sexe": "F",
                "sport_licence": "oui",
                "niveau_etude": "bac",
                "region": "Occitanie",
                "smoker": "oui",
                "nationalité_francaise": "oui",
                "montant_pret": 15000
            }
        ]
    }
    response = client.post("/retrain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "run_id" in data
