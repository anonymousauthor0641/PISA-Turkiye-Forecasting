import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import DATASET_DIR, FORECAST_YEARS, VARIABLE_ORDER
from data_source import PISA_RAW

def linear_trend_value(years, values, target_year):
    years_arr = np.asarray(years, dtype=float).reshape(-1, 1)
    values_arr = np.asarray(values, dtype=float)
    model = LinearRegression()
    model.fit(years_arr, values_arr)
    return float(model.predict(np.array([[target_year]], dtype=float))[0])

def prepare_processed_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # WEALTH 2003: nearest-year estimation using 2006.
    df.loc[df["Cycle"] == 2003, "WEALTH"] = float(
        df.loc[df["Cycle"] == 2006, "WEALTH"].iloc[0]
    )

    # HEDRES 2022 and WEALTH 2022: linear trend extension.
    for variable in ("HEDRES", "WEALTH"):
        history = df.loc[(df["Cycle"] < 2022) & df[variable].notna(), ["Cycle", variable]]
        estimate = linear_trend_value(history["Cycle"], history[variable], 2022)
        df.loc[df["Cycle"] == 2022, variable] = estimate

    return df

def forecast_future_exogenous_values(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in FORECAST_YEARS:
        row = {"Cycle": year}
        for variable in VARIABLE_ORDER:
            row[variable] = linear_trend_value(
                processed["Cycle"], processed[variable], year
            )
        rows.append(row)
    return pd.DataFrame(rows)

def save_datasets(raw, processed, future_exog):
    raw.to_excel(DATASET_DIR / "01_pisa_turkiye_raw.xlsx", index=False)
    processed.to_excel(DATASET_DIR / "02_pisa_turkiye_processed.xlsx", index=False)
    future_exog.to_excel(
        DATASET_DIR / "03_future_socioeconomic_inputs_exploratory.xlsx",
        index=False,
    )
    raw.to_csv(DATASET_DIR / "01_pisa_turkiye_raw.csv", index=False)
    processed.to_csv(DATASET_DIR / "02_pisa_turkiye_processed.csv", index=False)
    future_exog.to_csv(
        DATASET_DIR / "03_future_socioeconomic_inputs_exploratory.csv",
        index=False,
    )

if __name__ == "__main__":
    processed = prepare_processed_dataset(PISA_RAW)
    future_exog = forecast_future_exogenous_values(processed)
    save_datasets(PISA_RAW, processed, future_exog)
    print(processed)
    print(future_exog)
