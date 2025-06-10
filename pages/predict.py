import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Prédiction de prêt", page_icon=":money_with_wings:")

st.title("Prédiction de montant de prêt avec FastIA et MLFlow")

st.write("Remplissez le formulaire ci-dessous pour obtenir une prédiction.")
st.warning("Les données recueillis sont utilisées uniquement pour la prédiction et ne sont pas stockées. Vous pouvez vous oposer à leur utilisation en fermant cette application.")
st.markdown('<span style="font-size: 0.9em;"><i>Les champs marqués d\'un * sont obligatoires.</i></span>', unsafe_allow_html=True)

# Charger les régions distinctes depuis le CSV
try:
    df = pd.read_csv("data/df_new.csv")
    regions = sorted(df['region'].dropna().unique())
except Exception:
    regions = ["Corse", "Île-de-France", "Normandie", "Auvergne-Rhône-Alpes", "Provence-Alpes-Côte d’Azur", "Bretagne", "Hauts-de-France", "Occitanie"]

# Champs du formulaire : 2 colonnes pour les 4 indicateurs numériques, puis 1 colonne pour le reste
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Âge *", min_value=0, max_value=120, value=30)
    taille = st.slider("Taille (cm) *", min_value=100, max_value=250, value=170)
with col2:
    poids = st.slider("Poids (kg) *", min_value=30, max_value=250, value=70)
    revenu_estime_mois = st.slider("Revenu estimé par mois *", min_value=0, max_value=20000, value=2500, step=100)

# Champs catégoriels sur une seule colonne
sexe = st.selectbox("Sexe *", ["M", "F"])
sport_licence = st.selectbox("Sport licencié ? *", ["Oui", "Non"])
niveau_etude = st.selectbox("Niveau d'étude *", ["Aucun", "Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"])
region = st.selectbox("Région *", regions)
smoker = st.selectbox("Fumeur ? *", ["Oui", "Non"])
nationalite_francaise = st.selectbox("Nationalité française ? *", ["Oui", "Non"])

# Préparer les données dans l'ordre attendu
features = [
    int(age),
    float(taille),
    float(poids),
    float(revenu_estime_mois),
    str(sexe),
    str(sport_licence),
    str(niveau_etude),
    str(region),
    str(smoker),
    str(nationalite_francaise)
]

# Bouton de validation vert
validate_btn = st.button("Prédire", key="predict_btn")
st.markdown("""
    <style>
    div.stButton > button#predict_btn {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

if validate_btn:
    payload = {"data": features}
    api_url = os.getenv("API_URL", "http://localhost:8000")
    # route to predict
    url = f"{api_url}/predict"
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Bravo! Montant de prêt estimé : {prediction:.2f} €")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
