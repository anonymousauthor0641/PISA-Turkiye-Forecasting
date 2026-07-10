import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import LOOKBACK, RANDOM_SEEDS, SCENARIOS, TABLE_DIR
from models.arima_model import arima_forecast
from models.prophet_model import prophet_forecast
from models.recurrent_models import recurrent_one_step_forecast
from utils.metrics import calculate_metrics

def scale_fold(train, test, target, exog_columns):
    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(train[[target]]).reshape(-1)
    y_test = target_scaler.transform(test[[target]]).reshape(-1)

    if exog_columns:
        exog_scaler = MinMaxScaler()
        x_train = exog_scaler.fit_transform(train[list(exog_columns)])
        x_test = exog_scaler.transform(test[list(exog_columns)])
    else:
        x_train = np.empty((len(train), 0))
        x_test = np.empty((len(test), 0))

    return y_train, y_test, x_train, x_test

def rolling_backtest(
    df,
    target,
    model_name,
    scenario_name,
    exog_columns,
    seed=11,
    minimum_train_size=LOOKBACK + 1,
):
    actual_scaled = []
    predicted_scaled = []

    for test_index in range(minimum_train_size, len(df)):
        train = df.iloc[:test_index].copy()
        test = df.iloc[[test_index]].copy()
        y_train, y_test, x_train, x_test = scale_fold(
            train, test, target, exog_columns
        )

        if model_name == "Naive Persistence":
            prediction = float(y_train[-1])

        elif model_name == "ARIMA":
            prediction = float(arima_forecast(y_train, 1)[0])

        elif model_name == "Prophet":
            train_exog = pd.DataFrame(x_train, columns=exog_columns) if exog_columns else None
            test_exog = pd.DataFrame(x_test, columns=exog_columns) if exog_columns else None
            prediction = float(prophet_forecast(
                train["Cycle"].to_numpy(),
                y_train,
                test["Cycle"].to_numpy(),
                train_exog,
                test_exog,
            )[0])

        elif model_name in {"LSTM", "GRU"}:
            features = np.column_stack([y_train, x_train])
            next_window = features[-LOOKBACK:]
            prediction = recurrent_one_step_forecast(
                model_name, features, y_train, next_window, seed
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        actual_scaled.append(float(y_test[0]))
        predicted_scaled.append(prediction)

    return calculate_metrics(actual_scaled, predicted_scaled)

def run_all_backtests(processed):
    rows = []
    for target in ("Mathematics", "Science"):
        for scenario_name, exog_columns in SCENARIOS.items():
            for model_name in ("Naive Persistence", "ARIMA", "Prophet", "LSTM", "GRU"):
                if model_name == "Naive Persistence" and scenario_name != "baseline":
                    continue
                try:
                    metrics = rolling_backtest(
                        processed, target, model_name, scenario_name,
                        exog_columns, RANDOM_SEEDS[0]
                    )
                    rows.append({
                        "Target": target,
                        "Scenario": scenario_name,
                        "Model": model_name,
                        **metrics,
                    })
                except Exception as exc:
                    rows.append({
                        "Target": target,
                        "Scenario": scenario_name,
                        "Model": model_name,
                        "Error": str(exc),
                    })

    result = pd.DataFrame(rows)
    result.to_excel(TABLE_DIR / "rolling_backtesting_results.xlsx", index=False)
    result.to_csv(TABLE_DIR / "rolling_backtesting_results.csv", index=False)
    return result
