import streamlit as st

st.set_page_config(page_title="Accueil", page_icon=":house:")
st.title("Accueil - Application Prédiction & Réentraînement")

st.write("Bienvenue sur l'application de prédiction de montant de prêt.")

st.markdown("""
- Accédez à la page **Prédiction** pour obtenir une estimation de prêt à partir de vos données.
- Accédez à la page **Réentraînement** pour mettre à jour le modèle avec de nouvelles données ou réinitialiser l'entraînement.

Utilisez le menu à gauche pour naviguer entre les pages.
""")