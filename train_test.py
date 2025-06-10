import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# Générer un jeu de données de régression synthétique
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

mlflow.set_tracking_uri('http://localhost:5000')

# X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)




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
    print("Predictions : ", preds)
    return preds
    
def MLFlow_save_model(model, artifactPath="linear_regression_model"):
    """
    Save a model to MLFlow.
    
    Parameters:
    model: The model to be saved.
    artifactPath (str): The path where the model will be saved in MLFlow.
    
    Returns:
    runId (str): The ID of the MLFlow run in which the model was saved.
    """
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifactPath)
        runId = run.info.run_id
        print(f"Model saved with run ID: {runId}")
    return runId

def MLFlow_make_run(model, X_test, y_test, info="no information for run given", artifactPath="linear_regression_model"):
    """
    Save a model to MLFlow.
    
    Parameters:
    model: The model to be saved.
    artifactPath (str): The path where the model will be saved in MLFlow.
    
    Returns:
    runId (str): The ID of the MLFlow run in which the model was saved.
    """
    with mlflow.start_run() as run:
        mlflow.log_param("description", info)
        
        preds = MLFlow_make_prediction(model, X_test)
   
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", model.score(X_test, y_test))
        
        mlflow.sklearn.log_model(model, artifactPath)
        runId = run.info.run_id
        print(f"Model saved with run ID: {runId}")
    return runId

MLFlow_make_run(model, X_test, y_test, info="First try", artifactPath="linear_regression_model")

# # MLflow tracking
# with mlflow.start_run():
#     alpha = 0.5  # exemple fictif si on utilisait Ridge
#     mlflow.log_param("alpha", alpha)
        
#     preds = MLFlow_make_prediction(model, X_test)
   
#     mse = mean_squared_error(y_test, preds)
#     mae = mean_absolute_error(y_test, preds)
    
    
    
#     mlflow.log_metric("mse", mse)
#     mlflow.log_metric("mae", mse)
#     mlflow.log_metric("r2", model.score(X_test, y_test))
    
#     mlflow.sklearn.log_model(model, "linear_regression_model")
    
    
