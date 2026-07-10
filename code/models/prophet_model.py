import numpy as np
import pandas as pd

def prophet_forecast(
    train_years,
    train_values,
    future_years,
    train_exog=None,
    future_exog=None,
):
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError("Prophet is not installed.") from exc

    frame = pd.DataFrame({
        "ds": pd.to_datetime([f"{int(y)}-01-01" for y in train_years]),
        "y": np.asarray(train_values, dtype=float),
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    if train_exog is not None and not train_exog.empty:
        for column in train_exog.columns:
            model.add_regressor(column)
            frame[column] = train_exog[column].to_numpy(dtype=float)

    model.fit(frame)

    future = pd.DataFrame({
        "ds": pd.to_datetime([f"{int(y)}-01-01" for y in future_years])
    })

    if future_exog is not None and not future_exog.empty:
        for column in future_exog.columns:
            future[column] = future_exog[column].to_numpy(dtype=float)

    return model.predict(future)["yhat"].to_numpy(dtype=float)
