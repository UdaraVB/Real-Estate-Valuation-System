import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_house_model(X_train, y_train):
    """
    Performs Grid Search to find the best hyperparameters.
    Returns the best model found.
    """
    mlflow.sklearn.autolog()

    param_grid = {
        'n_estimators': [100, 200],    
        'max_depth': [None, 10, 20],      
        'min_samples_split': [2, 5],      
        'max_features': ['sqrt', None]    
    }

    rf = RandomForestRegressor(random_state=42)
    

    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, 
        scoring='r2'
    )

    with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning", nested=True):
        print("Starting Grid Search... this may take a few minutes.")
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\n✅ Best Parameters: {best_params}")
        print(f"✅ Best CV R2 Score: {best_score:.4f}")

    return grid_search.best_estimator_