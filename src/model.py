"""Definição e treinamento do modelo TensorFlow/Keras.

Fase 4 do projeto. Treina um MLP que recebe as features padronizadas da Fase 3
e prevê um par de gols (casa, visitante) por partida.

- Split **temporal** (não aleatório): treino = jogos antigos, validação e teste =
  jogos recentes. Evita usar o futuro para prever o passado.
- Saída: 2 neurônios (gols do time A e do time B). A conversão para inteiros
  ≥ 0 é feita na Fase 6 (`src/predict.py`).
- Perda MSE, otimizador Adam, `EarlyStopping` na validação, sementes fixas.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, RANDOM_SEED, set_seeds
from src.features import (
    FEATURE_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMNS,
    TRAIN_PATH,
    build_features,
    scale_features,
)

# --- Parâmetros da Fase 4 ---------------------------------------------------

VAL_START_YEAR = 2018   # validação: 2018–2021
TEST_START_YEAR = 2022  # teste: 2022+

EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.2
PATIENCE = 25

MODEL_PATH = OUTPUTS_DIR / "model.keras"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"


# --- Dados ------------------------------------------------------------------


def load_training_frame() -> pd.DataFrame:
    """Carrega a tabela de treino da Fase 3 (gera se ainda não existir)."""
    if not TRAIN_PATH.exists() or not SCALER_PATH.exists():
        build_features()
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_scaler() -> dict[str, dict[str, float]]:
    return json.loads(SCALER_PATH.read_text(encoding="utf-8"))


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide por data: treino < 2018, validação 2018–2021, teste >= 2022."""
    year = df["date"].dt.year
    train = df[year < VAL_START_YEAR]
    val = df[(year >= VAL_START_YEAR) & (year < TEST_START_YEAR)]
    test = df[year >= TEST_START_YEAR]
    return train, val, test


def to_xy(
    df: pd.DataFrame, scaler: dict[str, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de features padronizadas (X) e alvos de gols (y)."""
    x = scale_features(df, scaler)
    y = df[TARGET_COLUMNS].to_numpy(dtype="float32")
    return x.astype("float32"), y


# --- Modelo -----------------------------------------------------------------


def build_model(n_features: int):
    """MLP: 128 → 64 → 32 → 2 (gols casa/visitante), com dropout."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(DROPOUT),
            layers.Dense(64, activation="relu"),
            layers.Dropout(DROPOUT),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(TARGET_COLUMNS)),  # saída linear de gols
        ],
        name="copa2026_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(
    save: bool = True,
) -> tuple["object", dict, dict[str, np.ndarray]]:
    """Treina o modelo com split temporal e retorna (modelo, histórico, splits)."""
    set_seeds(RANDOM_SEED)
    from tensorflow import keras

    df = load_training_frame()
    scaler = load_scaler()
    train_df, val_df, test_df = temporal_split(df)

    x_train, y_train = to_xy(train_df, scaler)
    x_val, y_val = to_xy(val_df, scaler)
    x_test, y_test = to_xy(test_df, scaler)

    print(
        f"Split temporal -> treino: {len(x_train):,} | "
        f"val: {len(x_val):,} | teste: {len(x_test):,}"
    )

    model = build_model(x_train.shape[1])

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=0,
    )

    val_mae = float(model.evaluate(x_val, y_val, verbose=0)[1])
    test_mae = float(model.evaluate(x_test, y_test, verbose=0)[1])
    print(
        f"Épocas treinadas: {len(history.history['loss'])} | "
        f"MAE val: {val_mae:.3f} | MAE teste: {test_mae:.3f}"
    )

    splits = {
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val,
    }

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
        HISTORY_PATH.write_text(
            json.dumps(
                {
                    "epochs": len(history.history["loss"]),
                    "val_mae": val_mae,
                    "test_mae": test_mae,
                    "loss": [float(v) for v in history.history["loss"]],
                    "val_loss": [float(v) for v in history.history["val_loss"]],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Modelo salvo em {MODEL_PATH.name}")

    return model, history.history, splits


if __name__ == "__main__":
    train_model()
