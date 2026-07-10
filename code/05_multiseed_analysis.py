import pandas as pd

from config import RANDOM_SEEDS, TABLE_DIR
from importlib import import_module

rolling_module = import_module("04_rolling_backtesting")
rolling_backtest = rolling_module.rolling_backtest

def run_multiseed_analysis(processed):
    raw_rows = []

    for target in ("Mathematics", "Science"):
        for model_name in ("LSTM", "GRU"):
            for seed in RANDOM_SEEDS:
                try:
                    metrics = rolling_backtest(
                        processed,
                        target=target,
                        model_name=model_name,
                        scenario_name="baseline",
                        exog_columns=[],
                        seed=seed,
                    )
                    raw_rows.append({
                        "Target": target,
                        "Model": model_name,
                        "Seed": seed,
                        **metrics,
                    })
                except Exception as exc:
                    raw_rows.append({
                        "Target": target,
                        "Model": model_name,
                        "Seed": seed,
                        "Error": str(exc),
                    })

    raw = pd.DataFrame(raw_rows)
    raw.to_excel(TABLE_DIR / "multiseed_raw_results.xlsx", index=False)

    valid = raw.dropna(subset=["R2", "DTW", "MAE", "MSE", "RMSE"])
    rows = []
    for (target, model), group in valid.groupby(["Target", "Model"]):
        row = {"Target": target, "Model": model}
        for metric in ("R2", "DTW", "MAE", "MSE", "RMSE"):
            row[metric] = f"{group[metric].mean():.3f} ± {group[metric].std(ddof=1):.3f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_excel(TABLE_DIR / "multiseed_summary.xlsx", index=False)
    summary.to_csv(TABLE_DIR / "multiseed_summary.csv", index=False)
    return summary
