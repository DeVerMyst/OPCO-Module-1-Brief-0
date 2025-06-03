import streamlit as st
import requests
import pandas as pd

# Charger les régions distinctes depuis le CSV
try:
    df = pd.read_csv("data/df_new.csv")
    regions = sorted(df['region'].dropna().unique())
except Exception:
    regions = ["Corse", "Île-de-France", "Normandie", "Auvergne-Rhône-Alpes", "Provence-Alpes-Côte d’Azur", "Bretagne", "Hauts-de-France", "Occitanie"]

st.title("Prédiction de montant de prêt avec FastIA")

st.write("Remplissez le formulaire ci-dessous pour obtenir une prédiction.")

# Champs du formulaire
age = st.number_input("Âge", min_value=0, max_value=120, value=30)
taille = st.number_input("Taille (cm)", min_value=100, max_value=250, value=170)
poids = st.number_input("Poids (kg)", min_value=30, max_value=250, value=70)
revenu_estime_mois = st.number_input("Revenu estimé par mois", min_value=0, value=2500)
sexe = st.selectbox("Sexe", ["M", "F"])
sport_licence = st.selectbox("Sport licencié ?", ["Oui", "Non"])
niveau_etude = st.selectbox("Niveau d'étude", ["Aucun", "Bac", "Bac+2", "Bac+3", "Bac+5", "Doctorat"])
region = st.selectbox("Région", regions)
smoker = st.selectbox("Fumeur ?", ["Oui", "Non"])
nationalite_francaise = st.selectbox("Nationalité française ?", ["Oui", "Non"])

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

if st.button("Prédire"):
    # Appel à l'API FastAPI
    url = "http://127.0.0.1:8000/predict"
    payload = {"data": features}
    # st.write(f"[DEBUG] Payload envoyé à l'API : {payload}")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Montant de prêt prédit : {prediction:.2f} €")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")


