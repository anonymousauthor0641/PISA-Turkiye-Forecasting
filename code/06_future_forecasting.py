import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import FORECAST_YEARS, LOOKBACK, RANDOM_SEEDS, SCENARIOS, TABLE_DIR
from models.arima_model import arima_forecast
from models.prophet_model import prophet_forecast
from models.recurrent_models import recurrent_one_step_forecast

def fit_and_forecast_scenario(
    processed,
    future_exog,
    target,
    model_name,
    exog_columns,
    seed=11,
):
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(processed[[target]]).reshape(-1)

    if exog_columns:
        exog_scaler = MinMaxScaler()
        x_scaled = exog_scaler.fit_transform(processed[list(exog_columns)])
        future_x_scaled = exog_scaler.transform(future_exog[list(exog_columns)])
    else:
        x_scaled = np.empty((len(processed), 0))
        future_x_scaled = np.empty((len(future_exog), 0))

    if model_name == "ARIMA":
        pred_scaled = arima_forecast(y_scaled, len(FORECAST_YEARS))

    elif model_name == "Prophet":
        train_exog = pd.DataFrame(x_scaled, columns=exog_columns) if exog_columns else None
        test_exog = pd.DataFrame(future_x_scaled, columns=exog_columns) if exog_columns else None
        pred_scaled = prophet_forecast(
            processed["Cycle"].to_numpy(),
            y_scaled,
            np.asarray(FORECAST_YEARS),
            train_exog,
            test_exog,
        )

    elif model_name in {"LSTM", "GRU"}:
        feature_history = np.column_stack([y_scaled, x_scaled])
        target_history = y_scaled.copy()
        predictions = []

        for step in range(len(FORECAST_YEARS)):
            next_window = feature_history[-LOOKBACK:]
            pred = recurrent_one_step_forecast(
                model_name, feature_history, target_history,
                next_window, seed
            )
            predictions.append(pred)
            next_row = np.concatenate([[pred], future_x_scaled[step]])
            feature_history = np.vstack([feature_history, next_row])
            target_history = np.append(target_history, pred)

        pred_scaled = np.asarray(predictions)
    else:
        raise ValueError(model_name)

    return target_scaler.inverse_transform(
        np.asarray(pred_scaled).reshape(-1, 1)
    ).reshape(-1)

def run_forward_forecasts(processed, future_exog):
    rows = []
    for target in ("Mathematics", "Science"):
        for scenario_name, exog_columns in SCENARIOS.items():
            for model_name in ("ARIMA", "Prophet", "LSTM", "GRU"):
                try:
                    preds = fit_and_forecast_scenario(
                        processed, future_exog, target,
                        model_name, exog_columns, RANDOM_SEEDS[0]
                    )
                    row = {
                        "Target": target,
                        "Scenario": scenario_name,
                        "Model": model_name,
                    }
                    for year, value in zip(FORECAST_YEARS, preds):
                        row[str(year)] = float(value)
                    rows.append(row)
                except Exception as exc:
                    rows.append({
                        "Target": target,
                        "Scenario": scenario_name,
                        "Model": model_name,
                        "Error": str(exc),
                    })

    result = pd.DataFrame(rows)
    result.to_excel(TABLE_DIR / "forward_forecasts.xlsx", index=False)
    result.to_csv(TABLE_DIR / "forward_forecasts.csv", index=False)
    return result
