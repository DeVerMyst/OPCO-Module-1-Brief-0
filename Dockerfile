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

# Exposer les ports pour FastAPI (8000), Streamlit (8501) et MLflow (5000)
EXPOSE 8000 8501 5000

# Commande de démarrage multi-process (MLflow + API + Streamlit)
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /app/mlartifacts & uvicorn mlFlow_api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
