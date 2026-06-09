"""Engenharia de features para treino e inferência.

Fase 3 do projeto. Constrói, a partir dos dados brutos da Fase 2, uma tabela de
features por partida em **espaço de siglas** (sem vazamento temporal):

- diferença de Elo (rating do ano anterior à partida);
- diferença de pontos do ranking FIFA (valor mais recente *antes* da partida);
- forma recente de cada seleção (médias de gols pró/contra e pontos por jogo
  nas últimas N partidas anteriores ao jogo);
- mando / jogo em campo neutro.

Alvos: gols do time da casa (listado) e do visitante.

A mesma lógica gera a tabela de treino (histórico) e a tabela de inferência
(24 jogos da 1ª rodada da Copa 2026).
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, ROUND1_MATCHES
from src.data import (
    DATA_CUTOFF,
    load_elo_ratings,
    load_fifa_ranking,
    load_international_results,
)
from src.team_names import NAME_VARIANTS

# --- Parâmetros da Fase 3 ---------------------------------------------------

FORM_WINDOW = 10            # nº de partidas anteriores usadas na forma recente
FORM_MIN_MATCHES = 3        # mínimo de jogos para considerar a forma válida
TRAIN_START = date(1994, 1, 1)  # era moderna (ranking FIFA começa em 1992)

# País-sede de 2026: jogos do anfitrião listado como casa não são neutros.
HOSTS_2026 = {"USA", "MEX", "CAN"}

# Data de referência para os jogos de 2026 (usa todos os dados até a véspera).
INFERENCE_DATE = DATA_CUTOFF

FEATURE_COLUMNS: list[str] = [
    "elo_diff",
    "fifa_diff",
    "form_gf_diff",
    "form_ga_diff",
    "form_ppg_diff",
    "elo_a",
    "elo_b",
    "fifa_a",
    "fifa_b",
    "form_gf_a",
    "form_ga_a",
    "form_ppg_a",
    "form_gf_b",
    "form_ga_b",
    "form_ppg_b",
    "neutral",
]

TARGET_COLUMNS: list[str] = ["gols_a", "gols_b"]

TRAIN_PATH = DATA_PROCESSED_DIR / "train_features.csv"
INFERENCE_PATH = DATA_PROCESSED_DIR / "inference_features.csv"
MEDIANS_PATH = DATA_PROCESSED_DIR / "feature_medians.json"
SCALER_PATH = DATA_PROCESSED_DIR / "feature_scaler.json"


# --- Resolução de siglas ----------------------------------------------------


def build_name_code_map() -> dict[str, str]:
    """Mapa nome -> sigla unindo o ``team_short`` do FIFA com ``NAME_VARIANTS``.

    O ``team_short`` do FIFA é a sigla oficial de cada seleção; cobre ~200
    seleções. O ``NAME_VARIANTS`` (curado) trata as divergências de grafia entre
    datasets (ex.: "Ivory Coast" vs "Côte d'Ivoire") e tem prioridade.
    """
    fifa = load_fifa_ranking()
    fifa_map = (
        fifa.dropna(subset=["team_short"])
        .drop_duplicates("team")
        .set_index("team")["team_short"]
        .to_dict()
    )
    return {**fifa_map, **NAME_VARIANTS}


def _resolve_series(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Converte uma coluna de nomes de seleções em siglas (NaN se desconhecido)."""
    return names.str.strip().map(mapping)


# --- Forma recente ----------------------------------------------------------


def _long_team_matches(results: pd.DataFrame) -> pd.DataFrame:
    """Reorganiza partidas em formato longo: uma linha por seleção por jogo."""
    home = pd.DataFrame(
        {
            "match_id": results.index,
            "date": results["date"],
            "side": "a",
            "code": results["code_a"],
            "gf": results["home_score"],
            "ga": results["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": results.index,
            "date": results["date"],
            "side": "b",
            "code": results["code_b"],
            "gf": results["away_score"],
            "ga": results["home_score"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.where(
        long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0)
    )
    return long


def _add_rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    """Adiciona médias móveis da forma usando apenas jogos *anteriores*."""
    long = long.sort_values(["code", "date", "match_id"]).copy()
    grp = long.groupby("code", sort=False)

    def shifted_mean(col: str) -> pd.Series:
        return grp[col].transform(
            lambda s: s.shift().rolling(FORM_WINDOW, min_periods=FORM_MIN_MATCHES).mean()
        )

    long["form_gf"] = shifted_mean("gf")
    long["form_ga"] = shifted_mean("ga")
    long["form_ppg"] = shifted_mean("points")
    return long


# --- Junções as-of (FIFA / Elo) ---------------------------------------------


def _attach_fifa(results: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Anexa pontos FIFA mais recentes *antes* de cada partida (as-of por sigla)."""
    fifa_clean = (
        fifa.dropna(subset=["team_code"])
        .loc[:, ["team_code", "date", "total_points"]]
        .sort_values("date")
    )
    out = results.copy()
    for side in ("a", "b"):
        left = (
            out[["date", f"code_{side}"]]
            .rename(columns={f"code_{side}": "team_code"})
            .reset_index()
            .sort_values("date")
        )
        merged = pd.merge_asof(
            left,
            fifa_clean,
            on="date",
            by="team_code",
            direction="backward",
            allow_exact_matches=False,
        )
        out[f"fifa_{side}"] = merged.set_index("index")["total_points"]
    return out


def _attach_elo(results: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Anexa o Elo do ano anterior à partida (evita usar o ano corrente)."""
    elo_clean = (
        elo.dropna(subset=["team_code"])
        .loc[:, ["team_code", "year", "rating"]]
        .astype({"year": "int64"})
        .sort_values("year")
    )
    out = results.copy()
    out["lookup_year"] = (out["date"].dt.year - 1).astype("int64")
    for side in ("a", "b"):
        left = (
            out[["lookup_year", f"code_{side}"]]
            .rename(columns={f"code_{side}": "team_code", "lookup_year": "year"})
            .reset_index()
            .sort_values("year")
        )
        merged = pd.merge_asof(
            left,
            elo_clean,
            on="year",
            by="team_code",
            direction="backward",
        )
        out[f"elo_{side}"] = merged.set_index("index")["rating"]
    return out.drop(columns="lookup_year")


# --- Montagem da tabela de features -----------------------------------------


def _finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as colunas de diferença a partir dos valores por seleção."""
    df = df.copy()
    df["elo_diff"] = df["elo_a"] - df["elo_b"]
    df["fifa_diff"] = df["fifa_a"] - df["fifa_b"]
    df["form_gf_diff"] = df["form_gf_a"] - df["form_gf_b"]
    df["form_ga_diff"] = df["form_ga_a"] - df["form_ga_b"]
    df["form_ppg_diff"] = df["form_ppg_a"] - df["form_ppg_b"]
    return df


def assemble_training_table() -> pd.DataFrame:
    """Constrói a tabela completa de treino com features + alvos (sem imputar)."""
    mapping = build_name_code_map()
    results = load_international_results().reset_index(drop=True)
    fifa = load_fifa_ranking()
    elo = load_elo_ratings()

    results["code_a"] = _resolve_series(results["home_team"], mapping)
    results["code_b"] = _resolve_series(results["away_team"], mapping)
    fifa["team_code"] = _resolve_series(fifa["team"], mapping)
    elo["team_code"] = _resolve_series(elo["team"], mapping)

    # Só partidas modernas com ambas as seleções identificadas e placar válido.
    mask = (
        results["code_a"].notna()
        & results["code_b"].notna()
        & (results["date"].dt.date >= TRAIN_START)
        & results["home_score"].notna()
        & results["away_score"].notna()
    )
    results = results.loc[mask].reset_index(drop=True)

    # Forma recente.
    long = _add_rolling_form(_long_team_matches(results))
    form_cols = ["form_gf", "form_ga", "form_ppg"]
    for side in ("a", "b"):
        side_form = (
            long[long["side"] == side]
            .set_index("match_id")[form_cols]
            .rename(columns={c: f"{c}_{side}" for c in form_cols})
        )
        results = results.join(side_form)

    # Ranking FIFA e Elo (as-of, sem vazamento).
    results = _attach_fifa(results, fifa)
    results = _attach_elo(results, elo)

    results["neutral"] = results["neutral"].astype(int)
    results = _finalize_features(results)

    results["gols_a"] = results["home_score"].astype(int)
    results["gols_b"] = results["away_score"].astype(int)

    keep = (
        ["date", "code_a", "code_b"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )
    return results[keep]


def _team_latest_features(
    code: str,
    results: pd.DataFrame,
    fifa: pd.DataFrame,
    elo: pd.DataFrame,
    as_of: date,
) -> dict[str, float]:
    """Forma, FIFA e Elo mais recentes de uma seleção até ``as_of``."""
    cutoff = pd.Timestamp(as_of)

    # Forma: últimas FORM_WINDOW partidas da seleção antes do corte.
    played = results[
        ((results["code_a"] == code) | (results["code_b"] == code))
        & (results["date"] < cutoff)
    ].sort_values("date").tail(FORM_WINDOW)
    is_home = played["code_a"] == code
    gf = np.where(is_home, played["home_score"], played["away_score"]).astype(float)
    ga = np.where(is_home, played["away_score"], played["home_score"]).astype(float)
    pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    if len(played) >= FORM_MIN_MATCHES:
        form_gf, form_ga, form_ppg = gf.mean(), ga.mean(), pts.mean()
    else:
        form_gf = form_ga = form_ppg = np.nan

    # FIFA: pontuação mais recente antes do corte.
    fifa_team = fifa[(fifa["team_code"] == code) & (fifa["date"] < cutoff)]
    fifa_pts = (
        fifa_team.sort_values("date")["total_points"].iloc[-1]
        if not fifa_team.empty
        else np.nan
    )

    # Elo: rating do ano mais recente disponível antes do ano do corte.
    elo_team = elo[(elo["team_code"] == code) & (elo["year"] < cutoff.year)]
    elo_rating = (
        elo_team.sort_values("year")["rating"].iloc[-1]
        if not elo_team.empty
        else np.nan
    )

    return {
        "elo": elo_rating,
        "fifa": fifa_pts,
        "form_gf": form_gf,
        "form_ga": form_ga,
        "form_ppg": form_ppg,
    }


def assemble_inference_table(as_of: date = INFERENCE_DATE) -> pd.DataFrame:
    """Constrói as features dos 24 jogos da 1ª rodada de 2026 (sem imputar)."""
    mapping = build_name_code_map()
    results = load_international_results().reset_index(drop=True)
    fifa = load_fifa_ranking()
    elo = load_elo_ratings()

    results["code_a"] = _resolve_series(results["home_team"], mapping)
    results["code_b"] = _resolve_series(results["away_team"], mapping)
    fifa["team_code"] = _resolve_series(fifa["team"], mapping)
    elo["team_code"] = _resolve_series(elo["team"], mapping)

    rows: list[dict] = []
    for idx, (code_a, code_b) in enumerate(ROUND1_MATCHES, start=1):
        fa = _team_latest_features(code_a, results, fifa, elo, as_of)
        fb = _team_latest_features(code_b, results, fifa, elo, as_of)
        rows.append(
            {
                "jogo": f"jogo{idx}",
                "code_a": code_a,
                "code_b": code_b,
                "elo_a": fa["elo"],
                "elo_b": fb["elo"],
                "fifa_a": fa["fifa"],
                "fifa_b": fb["fifa"],
                "form_gf_a": fa["form_gf"],
                "form_ga_a": fa["form_ga"],
                "form_ppg_a": fa["form_ppg"],
                "form_gf_b": fb["form_gf"],
                "form_ga_b": fb["form_ga"],
                "form_ppg_b": fb["form_ppg"],
                "neutral": 0 if code_a in HOSTS_2026 else 1,
            }
        )

    df = _finalize_features(pd.DataFrame(rows))
    keep = ["jogo", "code_a", "code_b"] + FEATURE_COLUMNS
    return df[keep]


# --- Imputação e persistência -----------------------------------------------


def _impute(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col in df:
            df[col] = df[col].fillna(medians[col])
    return df


def fit_scaler(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calcula média/desvio das features no treino (padronização StandardScaler).

    Salvo como JSON (independente de máquina) para reaplicar nos 24 jogos sem
    refitar — evita vazamento dos dados de inferência na normalização.
    """
    scaler: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0)) or 1.0  # evita divisão por zero
        scaler[col] = {"mean": mean, "std": std}
    return scaler


def scale_features(df: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> np.ndarray:
    """Aplica a padronização salva e devolve a matriz X pronta para o modelo."""
    cols = []
    for col in FEATURE_COLUMNS:
        params = scaler[col]
        cols.append((df[col] - params["mean"]) / params["std"])
    return np.column_stack(cols)


def build_features(as_of: date = INFERENCE_DATE) -> dict[str, pd.DataFrame]:
    """Gera, imputa e persiste as tabelas de treino e inferência.

    As medianas usadas na imputação são calculadas no conjunto de treino e
    reaproveitadas na inferência, garantindo reprodutibilidade.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train = assemble_training_table()
    medians = {col: float(train[col].median()) for col in FEATURE_COLUMNS}
    train = _impute(train, medians)

    inference = assemble_inference_table(as_of=as_of)
    inference = _impute(inference, medians)

    scaler = fit_scaler(train)

    train.to_csv(TRAIN_PATH, index=False)
    inference.to_csv(INFERENCE_PATH, index=False)
    MEDIANS_PATH.write_text(
        json.dumps(medians, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    SCALER_PATH.write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "train": train,
        "inference": inference,
        "medians": medians,
        "scaler": scaler,
    }


def print_features_summary(tables: dict[str, pd.DataFrame]) -> None:
    train, inference = tables["train"], tables["inference"]
    print("\n--- Fase 3: features ---")
    print(f"Treino: {len(train):,} partidas, {len(FEATURE_COLUMNS)} features")
    print(f"  Período: {train['date'].min().date()} a {train['date'].max().date()}")
    print(f"  Gols (casa) média: {train['gols_a'].mean():.2f} | "
          f"visitante média: {train['gols_b'].mean():.2f}")
    print(f"Inferência: {len(inference)} jogos de 2026")
    print(
        f"  Arquivos: {TRAIN_PATH.name}, {INFERENCE_PATH.name}, "
        f"{MEDIANS_PATH.name}, {SCALER_PATH.name}"
    )


if __name__ == "__main__":
    from src.config import set_seeds

    set_seeds()
    tables = build_features()
    print_features_summary(tables)
