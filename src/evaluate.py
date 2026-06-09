"""Avaliação do modelo via pontuação do bolão e baselines.

Fase 5 do projeto. Mede no conjunto de **teste** (jogos a partir de 2022, que o
modelo nunca viu) o desempenho segundo a regra oficial do bolão:

- **5 pontos** se acertar o placar exato;
- **2 pontos** se acertar só o resultado (vitória/empate/derrota);
- **0** caso contrário.

Compara o modelo com três baselines simples (roadmap):

1. sempre empate 1×1;
2. vitória do time de maior Elo por 2×1;
3. placar médio histórico (gols médios casa/visitante).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, set_seeds
from src.features import scale_features
from src.model import (
    MODEL_PATH,
    load_scaler,
    load_training_frame,
    temporal_split,
    train_model,
)

EVAL_PATH = OUTPUTS_DIR / "evaluation.json"


# --- Conversão e pontuação --------------------------------------------------


def to_scores(pred: np.ndarray) -> np.ndarray:
    """Converte a saída contínua do modelo em gols inteiros ≥ 0."""
    return np.clip(np.rint(pred), 0, None).astype(int)


def bolao_points(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pontos do bolão por partida (5 placar exato, 2 resultado, 0 erro)."""
    exact = np.all(y_true == y_pred, axis=1)
    res_true = np.sign(y_true[:, 0] - y_true[:, 1])
    res_pred = np.sign(y_pred[:, 0] - y_pred[:, 1])
    result_ok = res_true == res_pred
    return np.where(exact, 5, np.where(result_ok, 2, 0))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas de avaliação para um conjunto de previsões."""
    points = bolao_points(y_true, y_pred)
    res_true = np.sign(y_true[:, 0] - y_true[:, 1])
    res_pred = np.sign(y_pred[:, 0] - y_pred[:, 1])
    return {
        "pontos_por_jogo": float(points.mean()),
        "pontos_totais": int(points.sum()),
        "placar_exato_pct": float((np.all(y_true == y_pred, axis=1)).mean() * 100),
        "resultado_pct": float((res_true == res_pred).mean() * 100),
        "mae_gols": float(np.abs(y_true - y_pred).mean()),
    }


# --- Baselines --------------------------------------------------------------


def baseline_draw(df: pd.DataFrame) -> np.ndarray:
    """Sempre 1×1."""
    return np.tile([1, 1], (len(df), 1))


def baseline_elo(df: pd.DataFrame) -> np.ndarray:
    """Time de maior Elo vence por 2×1 (empate técnico se Elo igual)."""
    a_stronger = df["elo_a"].to_numpy() >= df["elo_b"].to_numpy()
    return np.where(a_stronger[:, None], [2, 1], [1, 2])


def baseline_mean_score(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Placar médio histórico do treino (gols médios casa/visitante)."""
    score = [int(round(train["gols_a"].mean())), int(round(train["gols_b"].mean()))]
    return np.tile(score, (len(df), 1))


# --- Orquestração -----------------------------------------------------------


def run_evaluation(save: bool = True) -> dict[str, dict[str, float]]:
    """Avalia modelo e baselines no conjunto de teste e imprime o comparativo."""
    set_seeds()
    from tensorflow import keras

    df = load_training_frame()
    scaler = load_scaler()
    train_df, _, test_df = temporal_split(df)

    if not MODEL_PATH.exists():
        train_model()
    model = keras.models.load_model(MODEL_PATH)

    y_true = test_df[["gols_a", "gols_b"]].to_numpy()
    x_test = scale_features(test_df, scaler).astype("float32")
    y_model = to_scores(model.predict(x_test, verbose=0))

    results = {
        "Modelo (MLP)": metrics(y_true, y_model),
        "Baseline 1×1": metrics(y_true, baseline_draw(test_df)),
        "Baseline maior Elo 2×1": metrics(y_true, baseline_elo(test_df)),
        "Baseline placar médio": metrics(
            y_true, baseline_mean_score(test_df, train_df)
        ),
    }

    print(f"\n--- Fase 5: avaliação ({len(test_df):,} jogos de teste, >= 2022) ---")
    header = f"{'':24} {'pts/jogo':>9} {'exato%':>7} {'result%':>8} {'MAE':>6}"
    print(header)
    for name, m in results.items():
        print(
            f"{name:24} {m['pontos_por_jogo']:9.3f} "
            f"{m['placar_exato_pct']:6.1f}% {m['resultado_pct']:7.1f}% "
            f"{m['mae_gols']:6.3f}"
        )

    best = max(results, key=lambda k: results[k]["pontos_por_jogo"])
    print(f"\nMelhor: {best}")
    if best.startswith("Modelo"):
        print("O modelo supera os baselines — seguir para a Fase 6.")
    else:
        print("AVISO: o modelo não bateu o baseline — revisar features/modelo.")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return results


if __name__ == "__main__":
    run_evaluation()
