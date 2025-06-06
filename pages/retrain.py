import streamlit as st
import requests
import os

st.title("Réentraînement du modèle de prêt")

st.write("Sélectionnez un fichier CSV pour réentraîner le modèle.")

# Sélection du fichier CSV
csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
data_path = st.selectbox("Fichier de données (CSV) *", csv_files)

# Option pour fine-tuning ou nouveau modèle
from_existing = st.radio(
    "Type de réentraînement :",
    ("Continuer à partir du modèle actuel (fine-tuning)", "Réentraîner un nouveau modèle")
)
from_existing_model = from_existing == "Continuer à partir du modèle actuel (fine-tuning)"

if st.button("Lancer le réentraînement", key="retrain_btn"):
    url = "http://localhost:8000/retrain"
    payload = {"data_path": f"data/{data_path}", "from_existing_model": from_existing_model}
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
