import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(train_values, horizon: int):
    values = np.asarray(train_values, dtype=float).reshape(-1)
    if len(values) < 4:
        return np.repeat(values[-1], horizon)

    candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)]
    best_fit = None
    best_aic = np.inf

    for order in candidate_orders:
        try:
            fitted = ARIMA(values, order=order).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_fit = fitted
                best_aic = fitted.aic
        except Exception:
            continue

    if best_fit is None:
        return np.repeat(values[-1], horizon)

    return np.asarray(best_fit.forecast(steps=horizon), dtype=float)
