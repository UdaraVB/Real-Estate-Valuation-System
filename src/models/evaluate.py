import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def full_evaluation(y_true, y_pred):
    """
    Computes and prints the key regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(y_true_arr, 1))) * 100
    
    print("\n--- Model Evaluation Report ---")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} PKR")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} PKR")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}