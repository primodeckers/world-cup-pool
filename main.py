"""Pipeline principal — executar com: python main.py"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import (
    BOLAO_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    ROUND1_MATCHES,
    set_seeds,
)
from src.data import download_all, ensure_data_dirs, print_summary
from src.evaluate import run_evaluation
from src.features import build_features, print_features_summary
from src.model import train_model
from src.predict import generate_bolao


def validate_bolao() -> dict:
    """Valida estrutura do bolao_copa.txt."""
    with BOLAO_PATH.open(encoding="utf-8") as file:
        bolao = json.load(file)

    if len(bolao["resultados"]) != 24:
        raise ValueError("bolao_copa.txt deve conter exatamente 24 jogos.")

    for idx, (team_a, team_b) in enumerate(ROUND1_MATCHES, start=1):
        jogo = bolao["resultados"][f"jogo{idx}"]
        if team_a not in jogo or team_b not in jogo:
            raise ValueError(f"jogo{idx}: siglas incorretas (esperado {team_a} x {team_b}).")

    if bolao["nome"] == "SEU NOME COMPLETO":
        print("AVISO: atualize nome e turma em bolao_copa.txt antes da entrega.")

    return bolao


def main(force_download: bool = False) -> None:
    set_seeds(RANDOM_SEED)
    ensure_data_dirs()
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Semente: {RANDOM_SEED}")
    print(f"Dados brutos: {DATA_RAW_DIR}\n")

    validate_bolao()
    print("bolao_copa.txt: estrutura OK\n")

    print("=== Fase 2: coleta de dados ===")
    download_all(force=force_download)
    print_summary()

    print("\n=== Fase 3: engenharia de features ===")
    tables = build_features()
    print_features_summary(tables)

    print("\n=== Fase 4: treino do modelo ===")
    train_model()

    print("\n=== Fase 5: avaliação ===")
    run_evaluation()

    print("\n=== Fase 6: previsões + JSON ===")
    generate_bolao()
    print("\nPipeline completo. bolao_copa.txt pronto para entrega.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Copa 2026")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Rebaixa os CSVs mesmo se já existirem",
    )
    args = parser.parse_args()

    try:
        main(force_download=args.force_download)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
