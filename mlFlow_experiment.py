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

## Base initialisation for Loguru and FastAPI
from myapp_base import setup_loguru, create_app
from datetime import datetime

logger = setup_loguru("logs/mlFlow_experiment.log")
app = create_app()
today_str = datetime.now().strftime("%Y%m%d")

# Force url for MLFlow
mlflow.set_tracking_uri('http://localhost:5000')


info = {
    "description": "not set ",
    "dataversion": "df_new.csv",
}

# prediction_model = None  # Variable to hold the prediction model
prediction_model = "d84d764851cd421093d42b738e8dd140"  # Variable to hold the prediction model


def MLFlow_train_model(options, model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
    """
    Train a model and log it to MLFlow.
    
    Parameters:
    model: The model to be trained.
    X_train: The training features.
    y_train: The training labels.
    Returns:
    model: The trained model.
    """
    model, hist = train_model(model, X, y, X_val, y_val, epochs, batch_size, verbose)
    
    if options.get("save_model", False):
        
        step_base_name = options.get("step_base_name", f"model_{today_str}_ml_{options.get('step', 'default')}")
        # sauvegarder le drawloss
        draw_loss(hist, join('figures',f'{step_base_name}.jpg'))
        # sauvegarder le modèle
        joblib.dump(model, join('models', f'{step_base_name}.pkl'))
        logger.info(f"Model saved as {step_base_name}.pkl")
    
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
    

def MLFlow_save_run(model, preds, X_test, y_test, info, artifactPath="linear_regression_model"):
    """
    Save a model to MLFlow.
    
    Parameters:
    model: The model to be saved.
    artifactPath (str): The path where the model will be saved in MLFlow.
    
    Returns:
    runId (str): The ID of the MLFlow run in which the model was saved.
    """
    with mlflow.start_run() as run:
        mlflow.log_param("description", info["description"])
        mlflow.log_param("data_version", info["dataversion"])
        mlflow.log_param("current_train", info["current_train"])
        mlflow.log_param("train_count", info["wanted_train"])
           
        perf = evaluate_performance(y_test, preds)
        print_data(perf)
   
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", model.score(X_test, y_test))
        
        mlflow.sklearn.log_model(model, artifactPath)
        runId = run.info.run_id
        print(f"Model saved with run ID: {runId}")
    return runId


def MLFlow_analyse_dataset(info):
    """
    Analyse the dataset and log the results to MLFlow.
    
    Parameters:
    df: The dataset to be analysed.
    """
    # load and prepare dataset
    df = pd.read_csv(join('data',info["dataversion"]))
    X, y, _ = preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # load from specific model 
    if info["runId"] is not None:
        model = MLFlow_load_model(info["runId"])
        # retrouver les paramètres du modèle
        params = model.get_params()
        logger.info(f"Model loaded with run ID: {info['runId']} Model parameters: {params}")
        
    else:
        logger.info("No run ID provided, training a new model.")
        model = LinearRegression()
        
   
    MLFlow_train_model(model, X_train, y_train)
    
    preds = MLFlow_make_prediction(model, X_test)   
    perf = evaluate_performance(y_test, preds)
    print_data(perf)
    logger.info(f"Model performances: {perf}")
    
    # Save the model and metrics to MLFlow
    info["runId"] = MLFlow_save_run(model, preds, X_test, y_test, info, artifactPath="linear_regression_model")
    return info


### Function to train and log a model iteratively in MLFlow
def train_and_log_iterative(run_idx, info, run_id=None):
    """
    Entraîne un modèle et le log dans MLFlow, en utilisant un run_id pour charger un modèle précédent si disponible.
    Args:
        run_idx (int): L'index de l'itération d'entraînement
        info (dict): Dictionnaire contenant les informations de version des données et d'autres paramètres
        run_id (str, optional): L'ID du run précédent à charger. Si None, un nouveau modèle sera créé.
        Returns:
        str: L'ID du run MLFlow après l'entraînement.
        """
        
    df = pd.read_csv(join('data', info["dataversion"]))
    X, y, _ = preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Charger le modèle du run précédent ou créer un nouveau modèle
    if run_id is not None:
        logger.info(f"Loading model from previous run_id: {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_path}")
    else:
        logger.info("No previous run_id, creating new model.")
        model = create_nn_model(X_train.shape[1])
        run_id = today_str
        
    # step_base_name = f"model_{run_idx}_{run_id}"
    step_base_name = f"model_{run_id}"
    # Réentraîner le modèle
    # model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)
    
    model, hist = MLFlow_train_model({
        "save_model": True,
        "step_base_name": step_base_name,
        "step": run_idx
    }, model, X_train, y_train, X_val=X_test, y_val=y_test)
    
    
    
    # # sauvegarder le drawloss
    # draw_loss(hist, join('figures',f'{step_base_name}.jpg'))
    # # sauvegarder le modèle (Debug)
    # joblib.dump(model, join('models',f'{step_base_name}.pkl'))

    
    
    preds = model.predict(X_test)
    perf = evaluate_performance(y_test, preds)
    print_data(perf, exp_name=f"Performance for run {run_idx}/{wanted_train}")
    logger.info(f"Model performances: {perf}")
    with mlflow.start_run() as run:
        mlflow.log_param("description", f"Training iteration {run_idx}/{wanted_train}")
        mlflow.log_param("data_version", info["dataversion"])
        mlflow.log_param("random_state", 42)
        mlflow.log_param("previous_run_id", run_id if run_id else "None")
        mlflow.log_metric("mse", perf['MSE'])
        mlflow.log_metric("mae", perf['MAE'])
        mlflow.log_metric("r2", perf['R²'])
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Run {run_idx + 1} terminé, run_id={run.info.run_id}")
        return run.info.run_id
            
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Paramètres d'entraînement
        wanted_train = 3  # nombre d'entraînements à effectuer
        artifact_path = "linear_regression_model"
        run_id = None
        
        for i in range(wanted_train):
            logger.info(f"Starting training iteration {i} of {wanted_train}")
            run_id = train_and_log_iterative(i, info, run_id)
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
async def predict(payload: PredictRequest, uri: Request):
    """
    Endpoint pour faire une prédiction à partir d'un modèle MLflow sauvegardé.
    """
    logger.info(f"Route '{uri.url.path}' called with data: {payload.data}")
    try:
        # # Charger le modèle MLflow le plus récent (dernier run)
        # client = mlflow.tracking.MlflowClient()
        # runs = client.search_runs(experiment_ids=["0"], order_by=["attributes.start_time DESC"], max_results=1)
        # if not runs:
        #     raise HTTPException(status_code=404, detail="Aucun modèle MLflow trouvé.")
        run_id = prediction_model
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/linear_regression_model")
        
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