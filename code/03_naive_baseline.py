import numpy as np
from utils.metrics import calculate_metrics

def naive_persistence_predictions(series):
    values = np.asarray(series, dtype=float)
    return values[:-1], values[1:]

def evaluate_naive_persistence(series):
    predicted, actual = naive_persistence_predictions(series)
    return calculate_metrics(actual, predicted)
