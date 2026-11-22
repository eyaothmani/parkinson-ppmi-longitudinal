"""Inventaire des CSV disponibles : dimensions, patients, visites, taux de manquants (utilisé par l'EDA)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .tables import IGNORED_DIRS


def inventory(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).rglob("*.csv")):
        if set(path.parts) & IGNORED_DIRS or "(1)" in path.name:
            continue
        df = pd.read_csv(path, low_memory=False)
        rows.append({"dossier": path.parent.name if path.parent != Path(raw_dir) else "(racine)",
                     "fichier": path.name, "lignes": len(df), "colonnes": df.shape[1],
                     "patients": df["PATNO"].nunique() if "PATNO" in df else None,
                     "visites": df["EVENT_ID"].nunique() if "EVENT_ID" in df else None,
                     "manquants_pct": round(df.isna().mean().mean() * 100, 1)})
    return pd.DataFrame(rows)
