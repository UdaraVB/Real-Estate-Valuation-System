import os
import sys
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


from src.data.preprocess import clean_pakistan_real_estate
from src.models.train import train_house_model
from src.models.evaluate import full_evaluation

def main(args):

    db_path = os.path.join(project_root, "mlflow.db")
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True) 
    
    mlflow_uri = args.mlflow_uri or f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)
    print(f"📡 Tracking to: {mlflow_uri}")

    with mlflow.start_run(run_name="RF_Final_Run"):

        print("Loading raw data...")
        df_raw = pd.read_csv(args.input)
        print(f"Loaded {len(df_raw)} rows.")

        print("🔧 Cleaning data and handling Crore/Lakh logic...")
        df_clean = clean_pakistan_real_estate(df_raw)
        print(f"Clean rows: {len(df_clean)}")

        print(f"Training Random Forest (n={args.n_estimators})...")
        model, X_test, y_test = train_house_model(
            df_clean, 
            n_trees=args.n_estimators, 
            seed=42
        )
        
        model_save_path = os.path.join(artifacts_dir, "model.joblib")
        joblib.dump(model, model_save_path)
        print(f"Model saved physically to: {model_save_path}")
        
        feature_metadata_path = os.path.join(artifacts_dir, "feature_metadata.pkl")
        joblib.dump(list(X_test.columns), feature_metadata_path)
        print(f"Feature metadata saved to: {feature_metadata_path}")


        #EVALUATION
        print("Running final evaluation suite...")
        y_pred = model.predict(X_test)
        metrics = full_evaluation(y_test, y_pred)
        
        mlflow.log_metrics({
            "r2_score": metrics['r2'],
            "mae_pkr": metrics['mae']
        })

        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", type=str, default="Karachi_House_Prices")
    parser.add_argument("--mlflow_uri", type=str, default=None, help="e.g. sqlite:///mlflow.db")

    args = parser.parse_args()
    main(args)