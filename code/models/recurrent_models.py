import numpy as np

from config import LSTM_GRU_CONFIG, LOOKBACK
from utils.reproducibility import set_global_seed

def create_supervised_sequences(feature_matrix, target, lookback=LOOKBACK):
    x, y = [], []
    for idx in range(lookback, len(target)):
        x.append(feature_matrix[idx - lookback:idx])
        y.append(target[idx])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

def build_recurrent_model(model_name, input_shape, seed):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, GRU, LSTM
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed.") from exc

    set_global_seed(seed)
    layer_class = LSTM if model_name.upper() == "LSTM" else GRU

    model = Sequential([
        layer_class(
            units=LSTM_GRU_CONFIG["units"],
            activation=LSTM_GRU_CONFIG["activation"],
            input_shape=input_shape,
        ),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LSTM_GRU_CONFIG["learning_rate"]),
        loss=LSTM_GRU_CONFIG["loss"],
    )
    return model

def recurrent_one_step_forecast(
    model_name,
    train_features,
    train_target,
    next_window,
    seed,
):
    x_train, y_train = create_supervised_sequences(
        train_features, train_target, LOOKBACK
    )
    if len(x_train) == 0:
        return float(train_target[-1])

    model = build_recurrent_model(
        model_name,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        seed=seed,
    )
    model.fit(
        x_train,
        y_train,
        epochs=LSTM_GRU_CONFIG["epochs"],
        batch_size=LSTM_GRU_CONFIG["batch_size"],
        verbose=0,
        shuffle=False,
    )
    prediction = model.predict(next_window[np.newaxis, ...], verbose=0)
    return float(prediction.reshape(-1)[0])
