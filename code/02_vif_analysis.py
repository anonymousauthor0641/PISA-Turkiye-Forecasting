import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import TABLE_DIR, VARIABLE_ORDER

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    x = df[VARIABLE_ORDER].astype(float)
    x_with_const = np.column_stack([np.ones(len(x)), x.to_numpy()])

    rows = []
    for idx, variable in enumerate(VARIABLE_ORDER, start=1):
        vif = float(variance_inflation_factor(x_with_const, idx))
        r_squared = 1.0 - (1.0 / vif)
        rows.append({"Variable": variable, "R_squared": r_squared, "VIF": vif})

    result = pd.DataFrame(rows)
    result.to_excel(TABLE_DIR / "vif_analysis.xlsx", index=False)
    result.to_csv(TABLE_DIR / "vif_analysis.csv", index=False)
    return result
