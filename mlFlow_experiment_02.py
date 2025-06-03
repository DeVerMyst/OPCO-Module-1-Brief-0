import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from modules.preprocess import preprocessing
from modules.evaluate import evaluate_performance

from models.models import create_nn_model, train_model, model_predict
from modules.print_draw import print_data, draw_loss
import joblib



import pandas as pd
from os.path import join as join
from myapp_base import setup_loguru, create_app



logger = setup_loguru("logs/mlFlow_experiment02.log")
app = create_app()

mlflow.set_tracking_uri('http://localhost:5000')

# Paramètres d'entraînement
wanted_train = 3  # nombre d'entraînements à effectuer
info = {
    "description": "not set ",
    "dataversion": "df_new.csv",
}

artifact_path = "linear_regression_model"
run_id = None


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
        
    step_base_name = f"model_2025_06_ml_{run_idx}_{run_id}"
    # Réentraîner le modèle
    model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)
    
    # sauvegarder le drawloss
    draw_loss(hist, join('figures',f'{step_base_name}.jpg'))
    # sauvegarder le modèle (Debug)
    joblib.dump(model, join('models',f'{step_base_name}.pkl'))

    
    
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
    run_id = None
    for i in range(wanted_train):
        logger.info(f"Starting training iteration {i} of {wanted_train}")
        run_id = train_and_log_iterative(i, info, run_id)
