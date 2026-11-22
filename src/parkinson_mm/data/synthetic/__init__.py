"""Jeu de données SYNTHÉTIQUE au schéma PPMI (mêmes fichiers, colonnes et codes), pour la CI et la démo.

Aucune donnée réelle. Les distributions sont seulement plausibles (progression ~2 pts/an, écart OFF−ON ~10 pts,
codes 101, visites à distance, modalités partielles) afin d'exercer TOUTES les règles du pipeline.
Les résultats obtenus dessus n'ont aucune valeur scientifique.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .imaging import write_imaging
from .patients import make_patients, write_patient_tables
from .visits import write_visit_tables

FOLDERS = ("Motor___MDS-UPDRS", "Subject_Demographics", "DaTSCAN", "Genetic_Status", "MR_Analysis_Results",
           "Medical", "Non-motor_Assessments", "Biospecimen_Analysis")


def generate(out: str | Path, n_pd: int = 160, n_hc: int = 30, seed: int = 0) -> Path:
    out = Path(out)
    for folder in FOLDERS:
        (out / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    patients = make_patients(n_pd, n_hc, rng)
    write_patient_tables(out, patients, rng)
    write_visit_tables(out, patients, rng)
    write_imaging(out, patients, rng)
    (out / "README.md").write_text("Données SYNTHÉTIQUES au schéma PPMI — aucune donnée réelle.\n", encoding="utf-8")
    return out
