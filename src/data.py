"""Carregamento e download de dados brutos."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import DATA_RAW_DIR, ROUND1_MATCHES, TEAM_NAMES
from src.team_names import to_team_code

# Corte: só dados disponíveis antes da entrega (10/06/2026)
DATA_CUTOFF = date(2026, 6, 10)

DATA_SOURCES: dict[str, dict[str, str]] = {
    "international_results": {
        "filename": "international_results.csv",
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "description": "Histórico de partidas entre seleções (gols, torneio, mando)",
        "source": "https://github.com/martj42/international_results",
        "relevance": "Treino do modelo e cálculo de forma recente",
    },
    "fifa_ranking": {
        "filename": "fifa_ranking_historical.csv",
        "url": "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/ranking_fifa_historical.csv",
        "description": "Ranking FIFA masculino histórico (pontos e posição)",
        "source": "https://github.com/Dato-Futbol/fifa-ranking (dados oficiais FIFA)",
        "relevance": "Força relativa das seleções antes de cada jogo",
    },
    "elo_ratings": {
        "filename": "elo_ratings_yearly.csv",
        "url": "https://raw.githubusercontent.com/JGravier/soccer-elo/main/csv/ranking_soccer_1901-2023.csv",
        "description": "Elo anual das seleções (World Football Elo Ratings)",
        "source": "https://github.com/JGravier/soccer-elo (compilado de eloratings.net)",
        "relevance": "Proxy de força histórica complementar ao ranking FIFA",
    },
}


def ensure_data_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    """Baixa um arquivo se ainda não existir localmente."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_all(force: bool = False) -> dict[str, Path]:
    """Baixa todas as fontes definidas em DATA_SOURCES."""
    ensure_data_dirs()
    downloaded: dict[str, Path] = {}

    for key, meta in DATA_SOURCES.items():
        path = DATA_RAW_DIR / meta["filename"]
        print(f"Baixando {key}...")
        download_file(meta["url"], path, force=force)
        downloaded[key] = path
        print(f"  -> {path.name} ({path.stat().st_size // 1024} KB)")

    write_sources_manifest()
    return downloaded


def write_sources_manifest() -> Path:
    """Documenta fontes, URLs e data de corte."""
    manifest = {
        "data_cutoff": DATA_CUTOFF.isoformat(),
        "sources": DATA_SOURCES,
        "notes": (
            "Dados usados apenas até a véspera da entrega. "
            "Elo anual vai até 2023; ranking FIFA até ~2024. "
            "Forma recente será derivada de international_results na Fase 3."
        ),
    }
    path = DATA_RAW_DIR / "sources.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_csv(filename: str) -> pd.DataFrame:
    """Carrega um CSV de data/raw/."""
    path = DATA_RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Execute download_all() primeiro."
        )
    return pd.read_csv(path)


def load_international_results() -> pd.DataFrame:
    df = load_csv(DATA_SOURCES["international_results"]["filename"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.date < DATA_CUTOFF].copy()
    df["home_code"] = df["home_team"].map(to_team_code)
    df["away_code"] = df["away_team"].map(to_team_code)
    return df


def load_fifa_ranking() -> pd.DataFrame:
    df = load_csv(DATA_SOURCES["fifa_ranking"]["filename"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.date < DATA_CUTOFF].copy()
    df["team_code"] = df["team"].map(to_team_code)
    if "team_short" in df.columns:
        df["team_code"] = df["team_code"].fillna(df["team_short"])
    return df


def load_elo_ratings() -> pd.DataFrame:
    df = load_csv(DATA_SOURCES["elo_ratings"]["filename"])
    df["team_code"] = df["team"].map(to_team_code)
    df["rating"] = (
        df["rating"]
        .astype(str)
        .str.replace("\u2212", "-", regex=False)
        .astype(float)
    )
    return df


def load_world_cup_matches() -> pd.DataFrame:
    """Subset de jogos de Copa do Mundo para treino."""
    df = load_international_results()
    mask = df["tournament"].str.contains("World Cup", case=False, na=False)
    return df.loc[mask].copy()


def summarize_datasets() -> dict[str, Any]:
    """Resumo rápido dos dados baixados."""
    results = load_international_results()
    fifa = load_fifa_ranking()
    elo = load_elo_ratings()
    world_cup = load_world_cup_matches()

    teams_2026 = {code for pair in ROUND1_MATCHES for code in pair}
    mapped_in_results = results["home_code"].dropna().unique()
    coverage = sorted(teams_2026 & set(mapped_in_results))

    return {
        "international_results_rows": len(results),
        "international_results_date_range": (
            results["date"].min().date().isoformat(),
            results["date"].max().date().isoformat(),
        ),
        "world_cup_matches": len(world_cup),
        "fifa_ranking_rows": len(fifa),
        "fifa_latest_date": fifa["date"].max().date().isoformat(),
        "elo_years": (int(elo["year"].min()), int(elo["year"].max())),
        "teams_2026_with_history": coverage,
        "teams_2026_missing_history": sorted(teams_2026 - set(coverage)),
    }


def print_summary() -> None:
    summary = summarize_datasets()
    print("\n--- Resumo dos dados ---")
    print(f"Partidas internacionais: {summary['international_results_rows']:,}")
    print(
        f"  Período: {summary['international_results_date_range'][0]} "
        f"a {summary['international_results_date_range'][1]}"
    )
    print(f"Jogos de Copa do Mundo: {summary['world_cup_matches']:,}")
    print(f"Ranking FIFA: {summary['fifa_ranking_rows']:,} registros")
    print(f"  Última data: {summary['fifa_latest_date']}")
    print(f"Elo anual: {summary['elo_years'][0]}-{summary['elo_years'][1]}")
    print(
        f"Seleções 2026 com histórico mapeado: "
        f"{len(summary['teams_2026_with_history'])}/48"
    )
    if summary["teams_2026_missing_history"]:
        missing = ", ".join(summary["teams_2026_missing_history"])
        print(f"  Sem histórico mapeado ainda: {missing}")

    for code in sorted(summary["teams_2026_with_history"]):
        print(f"    {code} ({TEAM_NAMES[code]})")


if __name__ == "__main__":
    download_all()
    print_summary()
