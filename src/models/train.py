import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_house_model(df, n_trees=100, seed=42):
    """
    This is the function your pipeline is looking for!
    It takes the cleaned dataframe and trains the model.
    """

    n_cl = ['Area', 'Bedrooms', 'Baths', 'Location']
    X_encoded = pd.get_dummies(df[n_cl], columns=['Location'])
    y = df['Price']


    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=seed
    )


    with mlflow.start_run(run_name="RandomForest_Train_Step", nested=True):
        rf_model = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_trees)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_r2", r2)

        mlflow.sklearn.log_model(rf_model, "house_model_v1")

        print(f"✅ Training complete. R2: {r2:.2f}")
        

    return rf_model, X_test, y_test