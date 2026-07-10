import importlib

from config import OUTPUT_ROOT, FORECAST_YEARS, RANDOM_SEEDS, LOOKBACK, SCENARIOS
from data_source import PISA_RAW
from utils.reproducibility import save_json

pre = importlib.import_module("01_data_preprocessing")
vif_mod = importlib.import_module("02_vif_analysis")
backtest_mod = importlib.import_module("04_rolling_backtesting")
seed_mod = importlib.import_module("05_multiseed_analysis")
forecast_mod = importlib.import_module("06_future_forecasting")
figure_mod = importlib.import_module("07_generate_figures")

def main():
    processed = pre.prepare_processed_dataset(PISA_RAW)
    future_exog = pre.forecast_future_exogenous_values(processed)
    pre.save_datasets(PISA_RAW, processed, future_exog)

    vif = vif_mod.calculate_vif(processed)
    rolling = backtest_mod.run_all_backtests(processed)
    multiseed = seed_mod.run_multiseed_analysis(processed)
    forecasts = forecast_mod.run_forward_forecasts(processed, future_exog)
    figure_mod.create_historical_forecast_figures(
        processed, forecasts, selected_model="GRU"
    )

    metadata = {
        "forecast_years": list(FORECAST_YEARS),
        "random_seeds": list(RANDOM_SEEDS),
        "lookback": LOOKBACK,
        "scenarios": SCENARIOS,
        "interpretation": (
            "Exploratory country-level forecasting of officially reported OECD PISA "
            "national mean indicators."
        ),
    }
    save_json(metadata, OUTPUT_ROOT / "run_metadata.json")

    print("Pipeline completed.")
    print(f"Outputs: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()
