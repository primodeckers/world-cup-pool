"""Constantes e utilitários compartilhados do projeto."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
BOLAO_PATH = ROOT_DIR / "bolao_copa.txt"

TEAM_NAMES: dict[str, str] = {
    "MEX": "México",
    "RSA": "África do Sul",
    "KOR": "Coreia do Sul",
    "CZE": "Tchéquia",
    "CAN": "Canadá",
    "SUI": "Suíça",
    "QAT": "Catar",
    "BIH": "Bósnia e Herzegovina",
    "BRA": "Brasil",
    "MAR": "Marrocos",
    "HAI": "Haiti",
    "SCO": "Escócia",
    "USA": "Estados Unidos",
    "PAR": "Paraguai",
    "AUS": "Austrália",
    "TUR": "Turquia",
    "GER": "Alemanha",
    "CUW": "Curaçao",
    "CIV": "Costa do Marfim",
    "ECU": "Equador",
    "NED": "Países Baixos",
    "JPN": "Japão",
    "TUN": "Tunísia",
    "SWE": "Suécia",
    "BEL": "Bélgica",
    "EGY": "Egito",
    "IRN": "Irã",
    "NZL": "Nova Zelândia",
    "ESP": "Espanha",
    "CPV": "Cabo Verde",
    "KSA": "Arábia Saudita",
    "URU": "Uruguai",
    "FRA": "França",
    "SEN": "Senegal",
    "NOR": "Noruega",
    "IRQ": "Iraque",
    "ARG": "Argentina",
    "ALG": "Argélia",
    "AUT": "Áustria",
    "JOR": "Jordânia",
    "POR": "Portugal",
    "UZB": "Uzbequistão",
    "COL": "Colômbia",
    "COD": "RD Congo",
    "ENG": "Inglaterra",
    "CRO": "Croácia",
    "GHA": "Gana",
    "PAN": "Panamá",
}

# Ordem oficial: jogo1 .. jogo24 (time casa/listado primeiro, time visitante segundo)
ROUND1_MATCHES: list[tuple[str, str]] = [
    ("MEX", "RSA"),
    ("KOR", "CZE"),
    ("CAN", "BIH"),
    ("USA", "PAR"),
    ("HAI", "SCO"),
    ("AUS", "TUR"),
    ("BRA", "MAR"),
    ("QAT", "SUI"),
    ("CIV", "ECU"),
    ("GER", "CUW"),
    ("NED", "JPN"),
    ("SWE", "TUN"),
    ("KSA", "URU"),
    ("ESP", "CPV"),
    ("IRN", "NZL"),
    ("BEL", "EGY"),
    ("FRA", "SEN"),
    ("IRQ", "NOR"),
    ("ARG", "ALG"),
    ("AUT", "JOR"),
    ("GHA", "PAN"),
    ("ENG", "CRO"),
    ("POR", "COD"),
    ("UZB", "COL"),
]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fixa sementes e ativa o modo determinístico para reprodutibilidade.

    As variáveis de ambiente precisam ser definidas antes do primeiro import do
    TensorFlow; por isso `set_seeds()` é chamada no início de cada etapa, antes
    de qualquer uso do TF.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Desliga reordenações de ponto flutuante do oneDNN (não determinísticas).
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass  # versões antigas do TF
    except ImportError:
        pass
