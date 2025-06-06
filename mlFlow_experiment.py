import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Any
import numpy as np
import os

## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru, create_app
from datetime import datetime

logger = setup_loguru("logs/mlFlow_experiment.log")
app = create_app()
today_str = datetime.now().strftime("%Y%m%d_%H%M")

# Force url for MLFlow
mlflow.set_tracking_uri('http://localhost:5000')


settings = {
    "description": "not set ",
    "dataversion": "df_new.csv", # version de données de base d'entrainnement
    # "best_model": "1a22f130b64744ba9b235c1b6be1a7be",  # ID du meilleur modèle
    "wanted_train_cycle": 3,  # nombre d'entraînements à effectuer 3 est le meilleur
    "epochs": 50,  
    "train_seed": 42,  
}

# Paramètres d'entraînement
wanted_train_cycle = settings.get("wanted_train_cycle", 1)  # nombre d'entraînements à effectuer
artifact_path = "linear_regression_model"

# prediction_model = None  # Variable to hold the prediction model
# prediction_model = "1a22f130b64744ba9b235c1b6be1a7be"  # Variable to hold the BEST predicted model
# prediction_model = None  # Variable to hold the BEST predicted model
# prediction_model = "60ca87cde38a42dab673c4c6491ba076"  # Variable to hold the BEST predicted model
prediction_model = "3366968e6add45ddaad09162c63578e5"  # from df_modifie


def MLFlow_train_model(options, model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
    """
    Entraîne un modèle et le log le loss et ou le model.
    Args:
        options (dict): Options pour l'entraînement et le logging.
        model: Le modèle à entraîner.
        X (DataFrame): Les données d'entrée.
        y (Series): Les étiquettes cibles.
        X_val (DataFrame, optional): Les données de validation. Si None, pas de validation.
        y_val (Series, optional): Les étiquettes cibles de validation. Si None, pas de validation.
        epochs (int): Nombre d'époques pour l'entraînement.
        batch_size (int): Taille du batch pour l'entraînement.
        verbose (int): Niveau de verbosité pour l'entraînement.
        Returns:
        model: Le modèle entraîné.
        hist: L'historique de l'entraînement.
    """
    model, hist = train_model(model, X, y, X_val, y_val, epochs, batch_size, verbose)
    
    step_base_name = options.get("step_base_name", f"model_{today_str}_ml_{options.get('step', 'default')}")
    if options.get("save_model", False):
        # sauvegarder le modèle
        joblib.dump(model, join('models', f'{step_base_name}.pkl'))
        logger.info(f"Model saved as {step_base_name}.pkl")
        
    if options.get("save_cost", False):
        # sauvegarder le drawloss
        draw_loss(hist, join('figures',f'{step_base_name}.jpg'))
        
    
    return model, hist

def MLFlow_load_model(runId, artifactPath="linear_regression_model"):
    """
    Load a model from MLFlow using the run ID.
    
    Parameters:
    runId (str): The ID of the MLFlow run from which to load the model.
    
    Returns:
    model: The loaded model.
    """
    model_uri = f"runs:/{runId}/{artifactPath}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def MLFlow_make_prediction(model, X):
    """
    Make predictions using a loaded model.
    
    Parameters:
    model: The loaded model.
    X: The input features for which to make predictions.
    
    Returns:
    preds: The predictions made by the model.
    """
    preds = model.predict(X)
    # print("Predictions : ", preds)
    return preds
    

### Function to train and log a model iteratively in MLFlow
def train_and_log_iterative(run_idx, info, run_id=None):
    """
    Entraîne un modèle et le log dans MLFlow, en utilisant un run_id pour charger un modèle précédent si disponible.
    """
    df = pd.read_csv(join('data', info["dataversion"]))
    X_train, X_test, y_train, y_test = prepare_data(df, run_idx)
    
    run_desc = f"Performance for run {run_idx}/{wanted_train_cycle}"
    
    run_id = train_and_log_model(X_train, y_train, X_test, y_test, run_desc, run_id, run_idx, artifact_path)
    return run_id

def prepare_data(df, run_idx=0):
    """
    Prend un DataFrame, applique le préprocessing et split en train/test.
    Retourne X_train, X_test, y_train, y_test
    """
    X, y, _ = preprocessing(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42+run_idx)  # Ajout de run_idx pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)  # Ajout de run_idx pour la reproductibilité
    return X_train, X_test, y_train, y_test

def train_and_log_model(X_train, y_train, X_test, y_test, run_desc, model_id=None, run_idx=0, artifact_path="linear_regression_model"):
    """
    Entraîne un modèle, loggue dans MLflow, retourne le model_id.
    """
     # Charger le modèle du run précédent ou créer un nouveau modèle
    if model_id is not None:
        logger.info(f"Loading model from previous model_id: {model_id}")
        model = MLFlow_load_model(model_id, artifact_path)
    else:
        logger.info("No previous model_id, creating new model.")
        model = create_nn_model(X_train.shape[1])
        model_id = "None"  # Reset model_id if no previous model is loaded
    
    step_base_name = f"model_{today_str}_{run_idx}_{model_id}"
    model, hist = MLFlow_train_model({
        "save_model": False, # should be False when tests are finished
        "save_cost": True, # should be False when tests are finished
        "step_base_name": step_base_name,
        "step": run_idx
    }, model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50, batch_size=32, verbose=0)
    
    preds = MLFlow_make_prediction(model, X_test)
    
    perf = evaluate_performance(y_test, preds)
    print_data(perf, exp_name=run_desc)
    logger.info(f"Model performances: {perf}")
    

    with mlflow.start_run() as run:
        mlflow.log_param("description", run_desc)
        mlflow.log_param("data_version", settings.get("dataversion", "df_old.csv"))
        mlflow.log_param("random_state", settings.get("train_seed", 42))
        mlflow.log_param("previous_run_id", model_id if model_id else "None")
        mlflow.log_metric("mse", perf['MSE'])
        mlflow.log_metric("mae", perf['MAE'])
        mlflow.log_metric("r2", perf['R²'])
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Run {run_idx + 1} terminé, run_id={run.info.run_id}")
        return run.info.run_id
             
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        
        # run_id = None
        run_id = prediction_model
        
        for i in range(wanted_train_cycle):
            logger.info(f"Starting training iteration {i} of {wanted_train_cycle}")
            run_id = train_and_log_iterative(i, settings, run_id)
    else:
        print("Aucune action lancée. Pour entraîner, lancez : python mlFlow_experiment.py train")
        
        
@app.get("/health")
async def health(request: Request):
    """
    Endpoint de santé pour vérifier que l'application fonctionne.
    """
    logger.info(f"Route '{request.url.path}' called by {request.client.host}")
    return {"status": "healthy", "message": "API is running"}


class PredictRequest(BaseModel):
    data: List[Any]  # Liste des features pour une seule instance

@app.post("/predict")
async def predict(request: Request, payload: PredictRequest):
    """
    Endpoint pour faire une prédiction à partir d'un modèle MLflow sauvegardé.
    """
    logger.info(f"Route '{request.url.path}' called with data: {payload.data}")
    try:
        # # Charger le modèle MLflow le plus récent (dernier run)
        # client = mlflow.tracking.MlflowClient()
        # runs = client.search_runs(experiment_ids=["0"], order_by=["attributes.start_time DESC"], max_results=1)
        # if not runs:
        #     raise HTTPException(status_code=404, detail="Aucun modèle MLflow trouvé.")
        run_id = prediction_model
        model = MLFlow_load_model(run_id, artifact_path)
        
        logger.info(f"Model loaded from MLflow run ID: {run_id}")
                
        # Charger le préprocesseur
        preprocessor = joblib.load(join('models','preprocessor.pkl'))
        # Colonnes attendues par le préprocesseur
        numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
        categorical_cols = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]
        columns = numerical_cols + categorical_cols
        
        
        # Transformer les données d'entrée en DataFrame
        X_input = pd.DataFrame([payload.data], columns=columns)
        X_processed = preprocessor.transform(X_input)
        # Prédiction
        y_pred = model_predict(model, X_processed)
        
        predict = np.asarray(y_pred).squeeze().item()
        
        logger.info(f"Prediction made: {y_pred}, prediction value: {predict}")
        return {"prediction": float(predict)}
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RetrainRequest(BaseModel):
    data_path: str  # Chemin du fichier CSV à utiliser comme nouvelle source de données
    from_existing_model: bool = True  # True: fine-tuning, False: nouveau modèle

@app.post("/retrain")
async def retrain(request: Request, payload: RetrainRequest):
    """
    Réentraîne le modèle à partir d'un fichier CSV fourni (data_path) et d'une option pour fine-tuning ou nouveau modèle.
    """
    
    settings = {
        "description": "not set ",
        "dataversion": "df_new.csv", # version de données de base d'entrainnement
        # "best_model": "1a22f130b64744ba9b235c1b6be1a7be",  # ID du meilleur modèle
        "wanted_train_cycle": 3,  # nombre d'entraînements à effectuer 3 est le meilleur
        "epochs": 50,  
        "train_seed": 42,  
    }
    
    logger.info(f"Route '{request.url.path}' called for retraining with data_path={payload.data_path}, from_existing_model={payload.from_existing_model}")
    
    
    try:
        # Charger le dataset fourni
        df = pd.read_csv(payload.data_path)
        logger.info(f"Données chargées pour réentraînement: shape={df.shape}, colonnes={df.columns.tolist()}")
        run_id = prediction_model if payload.from_existing_model else None
        # Mettre à jour la version des données avec seulement le nom du fichier
        settings["dataversion"] = os.path.basename(payload.data_path)
        for i in range(wanted_train_cycle):
            logger.info(f"Starting training iteration {i} of {wanted_train_cycle} (from_existing_model={payload.from_existing_model})")
            run_id = train_and_log_iterative(i, settings, run_id)
            
        # Mettre à jour le modèle de prédiction avec le dernier run_id
        prediction_model = run_id  # Mettre à jour le modèle de prédiction avec le dernier 
        logger.info(f"Nouveau modèle de prédiction mis à jour avec run_id: {run_id}")
        return {"status": "success", "nouveau modèle actif pour la prévision : ": run_id}
    except Exception as e:
        logger.error(f"Erreur lors du réentraînement: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    