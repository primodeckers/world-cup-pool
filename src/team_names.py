"""Mapeamento de nomes de seleções para siglas oficiais da Copa 2026."""

from __future__ import annotations

from src.config import TEAM_NAMES

# Sigla -> confederação (Copa 2026)
CONFEDERATION: dict[str, str] = {
    "MEX": "CONCACAF",
    "RSA": "CAF",
    "KOR": "AFC",
    "CZE": "UEFA",
    "CAN": "CONCACAF",
    "SUI": "UEFA",
    "QAT": "AFC",
    "BIH": "UEFA",
    "BRA": "CONMEBOL",
    "MAR": "CAF",
    "HAI": "CONCACAF",
    "SCO": "UEFA",
    "USA": "CONCACAF",
    "PAR": "CONMEBOL",
    "AUS": "AFC",
    "TUR": "UEFA",
    "GER": "UEFA",
    "CUW": "CONCACAF",
    "CIV": "CAF",
    "ECU": "CONMEBOL",
    "NED": "UEFA",
    "JPN": "AFC",
    "TUN": "CAF",
    "SWE": "UEFA",
    "BEL": "UEFA",
    "EGY": "CAF",
    "IRN": "AFC",
    "NZL": "OFC",
    "ESP": "UEFA",
    "CPV": "CAF",
    "KSA": "AFC",
    "URU": "CONMEBOL",
    "FRA": "UEFA",
    "SEN": "CAF",
    "NOR": "UEFA",
    "IRQ": "AFC",
    "ARG": "CONMEBOL",
    "ALG": "CAF",
    "AUT": "UEFA",
    "JOR": "AFC",
    "POR": "UEFA",
    "UZB": "AFC",
    "COL": "CONMEBOL",
    "COD": "CAF",
    "ENG": "UEFA",
    "CRO": "UEFA",
    "GHA": "CAF",
    "PAN": "CONCACAF",
}

# Variantes de nomes em datasets -> sigla oficial
NAME_VARIANTS: dict[str, str] = {
    "Mexico": "MEX",
    "México": "MEX",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Korea Republic": "KOR",
    "Korea DPR": "PRK",
    "North Korea": "PRK",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Czechoslovakia": "CZE",
    "Canada": "CAN",
    "Switzerland": "SUI",
    "Qatar": "QAT",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Brasil": "BRA",
    "Morocco": "MAR",
    "Haiti": "HAI",
    "Scotland": "SCO",
    "United States": "USA",
    "USA": "USA",
    "Paraguay": "PAR",
    "Australia": "AUS",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Germany": "GER",
    "Curaçao": "CUW",
    "Curacao": "CUW",
    "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Ecuador": "ECU",
    "Netherlands": "NED",
    "Holland": "NED",
    "Japan": "JPN",
    "Tunisia": "TUN",
    "Sweden": "SWE",
    "Belgium": "BEL",
    "Egypt": "EGY",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Spain": "ESP",
    "Cape Verde": "CPV",
    "Saudi Arabia": "KSA",
    "Uruguay": "URU",
    "France": "FRA",
    "Senegal": "SEN",
    "Norway": "NOR",
    "Iraq": "IRQ",
    "Argentina": "ARG",
    "Algeria": "ALG",
    "Austria": "AUT",
    "Jordan": "JOR",
    "Portugal": "POR",
    "Uzbekistan": "UZB",
    "Colombia": "COL",
    "DR Congo": "COD",
    "Congo DR": "COD",
    "Congo": "COG",
    "England": "ENG",
    "Croatia": "CRO",
    "Ghana": "GHA",
    "Panama": "PAN",
}

# Nome canônico (valor de TEAM_NAMES) -> sigla
for code, name in TEAM_NAMES.items():
    NAME_VARIANTS.setdefault(name, code)


def to_team_code(name: str) -> str | None:
    """Converte nome do dataset para sigla oficial, se conhecido."""
    if not name or not isinstance(name, str):
        return None
    cleaned = name.strip()
    if cleaned in NAME_VARIANTS:
        return NAME_VARIANTS[cleaned]
    if len(cleaned) == 3 and cleaned.isupper() and cleaned in TEAM_NAMES:
        return cleaned
    return None
