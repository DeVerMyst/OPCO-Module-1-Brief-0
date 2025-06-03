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

## Base initialisation of Loguru and FastAPI
from myapp_base import setup_loguru, create_app

logger = setup_loguru("logs/mlFlow_experiment.log")
app = create_app()


# Force url for MLFlow
mlflow.set_tracking_uri('http://localhost:5000')


#make object with info : dataversion, wanted_train
info = {
    "description" : "not set ",
    "dataversion": "df_old.csv",
    "wanted_train": 3,
    "current_train": 0,  # This will be incremented with each training iteration
    "runId": None  # This will be set after the first run
}


def MLFlow_train_model(model, X_train, y_train):
    """
    Train a model and log it to MLFlow.
    
    Parameters:
    model: The model to be trained.
    X_train: The training features.
    y_train: The training labels.
    Returns:
    model: The trained model.
    """
    model.fit(X_train, y_train), 
    return model

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
        info["current_train"] = params.get("current_train", 0)
        
    else:
        logger.info("No run ID provided, training a new model.")
        model = LinearRegression()
        info["current_train"] = 0
        
    model.fit(X_train, y_train)
    info["current_train"] += 1
    info["description"] = f"Training iteration {info['current_train']/info['wanted_train']}"
    
    
    # if info["current_train"] is not None:
    # else:
    #     info["description"] = "Initial training"
    logger.info(f"Training model with description: {info['description']}")
    
    MLFlow_train_model(model, X_train, y_train)
    
    preds = MLFlow_make_prediction(model, X_test)   
    perf = evaluate_performance(y_test, preds)
    print_data(perf)
    logger.info(f"Model performances: {perf}")
    
    # Save the model and metrics to MLFlow
    info["runId"] = MLFlow_save_run(model, preds, X_test, y_test, info, artifactPath="linear_regression_model")
    return info

while info["current_train"] < info["wanted_train"]:
    logger.info(f"Starting training iteration {info['current_train']} of {info['wanted_train']}")
    info = MLFlow_analyse_dataset(info)
