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
# model = LinearRegression()
# model.fit(X_train, y_train)


# MLflow tracking
with mlflow.start_run():
    alpha = 0.5  # exemple fictif si on utilisait Ridge
    mlflow.log_param("alpha", alpha)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
   
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mse)
    mlflow.log_metric("r2", model.score(X_test, y_test))
    
    mlflow.sklearn.log_model(model, "linear_regression_model")
    
    
