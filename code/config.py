from pathlib import Path

OUTPUT_ROOT = Path("outputs")
DATASET_DIR = OUTPUT_ROOT / "datasets"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

for directory in (DATASET_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEEDS = (11, 22, 33, 44, 55)
FORECAST_YEARS = (2025, 2028, 2031)
LOOKBACK = 4

LSTM_GRU_CONFIG = {
    "epochs": 100,
    "batch_size": 4,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "activation": "tanh",
    "units": 16,
    "lookback": LOOKBACK,
}

VARIABLE_ORDER = ["HEDRES", "HOMEPOS", "WEALTH"]

SCENARIOS = {
    "baseline": [],
    "hedres": ["HEDRES"],
    "hedres_homepos": ["HEDRES", "HOMEPOS"],
    "hedres_homepos_wealth": ["HEDRES", "HOMEPOS", "WEALTH"],
}
