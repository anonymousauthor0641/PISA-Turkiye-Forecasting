import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def dtw_distance(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    matrix = np.full((len(a) + 1, len(p) + 1), np.inf)
    matrix[0, 0] = 0.0

    for i in range(1, len(a) + 1):
        for j in range(1, len(p) + 1):
            cost = abs(a[i - 1] - p[j - 1])
            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1],
            )
    return float(matrix[-1, -1])

def calculate_metrics(actual, predicted):
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    mse = mean_squared_error(actual_arr, pred_arr)
    r2 = r2_score(actual_arr, pred_arr) if len(actual_arr) >= 2 else np.nan
    return {
        "R2": float(r2),
        "DTW": dtw_distance(actual_arr, pred_arr),
        "MAE": float(mean_absolute_error(actual_arr, pred_arr)),
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
    }
