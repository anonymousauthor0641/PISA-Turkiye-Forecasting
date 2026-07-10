import numpy as np
import matplotlib.pyplot as plt

from config import FIGURE_DIR, FORECAST_YEARS, SCENARIOS

def scenario_label(name):
    return {
        "baseline": "Baseline",
        "hedres": "HEDRES",
        "hedres_homepos": "HEDRES + HOMEPOS",
        "hedres_homepos_wealth": "HEDRES + HOMEPOS + WEALTH",
    }[name]

def create_historical_forecast_figures(processed, forecasts, selected_model="GRU"):
    for target in ("Mathematics", "Science"):
        for scenario_name in SCENARIOS:
            subset = forecasts[
                (forecasts["Target"] == target)
                & (forecasts["Scenario"] == scenario_name)
                & (forecasts["Model"] == selected_model)
            ]
            if subset.empty or "Error" in subset.columns and subset["Error"].notna().any():
                continue

            forecast_values = subset.iloc[0][[str(y) for y in FORECAST_YEARS]].astype(float)

            hist_years = processed["Cycle"].to_numpy()
            hist_values = processed[target].to_numpy()

            forecast_years = np.array([hist_years[-1], *FORECAST_YEARS])
            forecast_line = np.array([hist_values[-1], *forecast_values.to_numpy()])

            plt.figure(figsize=(9, 5.5))
            plt.plot(
                hist_years, hist_values,
                marker="o", linestyle="-",
                label="Historical observations",
            )
            plt.plot(
                forecast_years, forecast_line,
                marker="s", linestyle="--",
                label=f"{selected_model} exploratory forecasts",
            )
            plt.axvline(2022, linestyle=":", linewidth=1)
            plt.xlabel("PISA cycle")
            plt.ylabel(f"{target} literacy score")
            plt.title(
                f"Türkiye PISA {target} literacy: historical observations and "
                f"{selected_model} forecasts\nScenario: {scenario_label(scenario_name)}"
            )
            plt.legend()
            plt.tight_layout()
            filename = (
                f"{target.lower()}_{scenario_name}_{selected_model.lower()}_"
                "historical_and_forecast.png"
            )
            plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
            plt.close()
