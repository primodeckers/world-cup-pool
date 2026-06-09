"""Geração das previsões e exportação do JSON final.

Fase 6 do projeto. Carrega o modelo treinado (Fase 4), prevê os 24 jogos da 1ª
rodada de 2026 a partir das features de inferência (Fase 3), converte a saída em
gols inteiros ≥ 0 e exporta o `bolao_copa.txt` no formato exigido.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BOLAO_PATH, OUTPUTS_DIR, ROUND1_MATCHES, set_seeds
from src.evaluate import to_scores
from src.features import INFERENCE_PATH, build_features, scale_features
from src.model import MODEL_PATH, load_scaler

NOME = "RENE ESTEVAM DECKERS"
TURMA = "DEEP LEARNING E PROCESSAMENTO DE LINGUAGEM NATURAL - 2º BIM 2026"
PREVISOES_PATH = OUTPUTS_DIR / "previsoes.json"


def build_results_payload(
    predictions: list[tuple[int, int]],
    nome: str,
    turma: str,
) -> dict[str, Any]:
    """Monta o dicionário no formato exigido pelo bolão."""
    if len(predictions) != len(ROUND1_MATCHES):
        raise ValueError("É necessário exatamente 24 previsões.")

    resultados: dict[str, Any] = {}
    for idx, ((team_a, team_b), (gols_a, gols_b)) in enumerate(
        zip(ROUND1_MATCHES, predictions, strict=True),
        start=1,
    ):
        resultados[f"jogo{idx}"] = {
            team_a: {"gols": int(gols_a)},
            team_b: {"gols": int(gols_b)},
        }

    return {"nome": nome, "turma": turma, "resultados": resultados}


def export_predictions(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Salva o JSON de previsões."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = path or BOLAO_PATH

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return output_path


def predict_round1() -> list[tuple[int, int]]:
    """Prevê os 24 jogos de 2026 e devolve os pares de gols inteiros, em ordem."""
    set_seeds()
    from tensorflow import keras

    if not INFERENCE_PATH.exists():
        build_features()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} não encontrado. Treine o modelo (python -m src.model)."
        )

    inference = pd.read_csv(INFERENCE_PATH)
    # Garante a ordem oficial jogo1..jogo24.
    inference = inference.sort_values(
        "jogo", key=lambda s: s.str.removeprefix("jogo").astype(int)
    )

    scaler = load_scaler()
    model = keras.models.load_model(MODEL_PATH)
    x = scale_features(inference, scaler).astype("float32")
    scores = to_scores(model.predict(x, verbose=0))
    return [(int(a), int(b)) for a, b in scores]


def generate_bolao(nome: str = NOME, turma: str = TURMA) -> dict[str, Any]:
    """Gera as previsões, grava o `bolao_copa.txt` e uma cópia em `outputs/`."""
    predictions = predict_round1()
    payload = build_results_payload(predictions, nome=nome, turma=turma)
    export_predictions(payload, BOLAO_PATH)
    export_predictions(payload, PREVISOES_PATH)

    print(f"\n--- Fase 6: previsões ({len(predictions)} jogos) ---")
    for (team_a, team_b), (gols_a, gols_b) in zip(ROUND1_MATCHES, predictions):
        print(f"  {team_a} {gols_a}-{gols_b} {team_b}")
    print(f"\nGravado em {BOLAO_PATH.name} e {PREVISOES_PATH.name}")
    return payload


if __name__ == "__main__":
    generate_bolao()
