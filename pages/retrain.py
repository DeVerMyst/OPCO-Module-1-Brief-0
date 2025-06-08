import streamlit as st
import requests
import os

st.title("Réentraînement du modèle de prêt")

st.write("Sélectionnez un fichier CSV pour réentraîner le modèle.")

# Sélection robuste du dossier data (compatible Docker et local)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
data_dir = os.path.abspath(data_dir)
if not os.path.exists(data_dir):
    st.error(f"Le dossier data/ est introuvable dans {data_dir}")
    st.stop()
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
data_path = st.selectbox("Fichier de données (CSV) *", csv_files)

# Option pour fine-tuning ou nouveau modèle
from_existing = st.radio(
    "Type de réentraînement :",
    ("Continuer à partir du modèle actuel (fine-tuning)", "Réentraîner un nouveau modèle")
)
from_existing_model = from_existing == "Continuer à partir du modèle actuel"

# Choix dynamique de l'URL de l'API (local ou Docker)
api_url = os.getenv("API_URL", "http://localhost:8000")

if st.button("Lancer le réentraînement", key="retrain_btn"):
    url = f"{api_url}/retrain"
    payload = {"data_path": os.path.join("data", data_path), "from_existing_model": from_existing_model}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            run_id = response.json()["run_id"]
            st.success(f"Réentraînement terminé ! Nouveau run_id : {run_id}")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

st.markdown("""
    <style>
    div.stButton > button#retrain_btn {
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
