# Utilise une image officielle Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Exposer les ports pour FastAPI (8000) et Streamlit (8501)
EXPOSE 8001 8502

# Commande de démarrage multi-process (API + Streamlit)
CMD ["sh", "-c", "uvicorn mlFlow_api:app --host 0.0.0.0 --port 8001 & streamlit run app_streamlit.py --server.port 8502 --server.address 0.0.0.0"]
