"""Lecture des CSV PPMI et conversion des dates (format MM/YYYY après désidentification)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .tables import ALL, IGNORED_DIRS, OPTIONAL


def find_file(raw_dir: Path, key: str) -> Path | None:
    """Fichier le plus récent pour une table ; ignore les dossiers de rebut et les doublons « (1) »."""
    hits = [p for p in Path(raw_dir).rglob(ALL.get(key, key))
            if not set(p.parts) & IGNORED_DIRS and "(1)" not in p.name]
    return sorted(hits)[-1] if hits else None


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for key in ALL:
        path = find_file(raw_dir, key)
        if path is not None:
            tables[key] = pd.read_csv(path, low_memory=False)
        elif key not in OPTIONAL:
            raise FileNotFoundError(f"Table obligatoire absente : {key} ({ALL[key]}) dans {raw_dir}")
    return tables


def parse_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%m/%Y", errors="coerce")


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
