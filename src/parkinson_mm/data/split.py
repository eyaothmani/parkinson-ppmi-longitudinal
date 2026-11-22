"""Découpage au niveau PATIENT, fait AVANT toute construction de features ou tout prétraitement.

Toutes les visites d'un patient sont dans le même ensemble : sinon le modèle « reconnaîtrait » le patient
et les performances seraient surestimées (ADR-0005). Le test n'est utilisé qu'une fois, à la fin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_patient_split(long: pd.DataFrame, test_size: float = 0.2, n_folds: int = 5, seed: int = 42,
                       min_visits: int = 2) -> pd.DataFrame:
    """PATNO | split (dev/test) | fold (0..k-1 pour le dev, -1 pour le test).
    Stratifié sur les tertiles du nombre de visites pour que les ensembles aient des suivis comparables."""
    counts = long.groupby("PATNO").size()
    p = counts[counts >= min_visits].rename("n").reset_index()
    rng = np.random.default_rng(seed)
    p["stratum"] = pd.qcut(p["n"].rank(method="first"), 3, labels=False)
    p["draw"] = rng.random(len(p))
    p["split"] = "dev"
    for _, group in p.groupby("stratum"):
        n_test = int(round(len(group) * test_size))
        p.loc[group.sort_values("draw").index[:n_test], "split"] = "test"
    dev = p[p["split"] == "dev"].sort_values(["stratum", "draw"])
    p["fold"] = -1
    p.loc[dev.index, "fold"] = np.arange(len(dev)) % n_folds
    return p[["PATNO", "split", "fold"]].sort_values("PATNO").reset_index(drop=True)


def attach_split(samples: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    out = samples.drop(columns=["split", "fold"], errors="ignore").merge(splits, on="PATNO", how="left")
    if out["split"].isna().any():
        raise ValueError("Des échantillons n'ont pas de split : relancer l'étape split.")
    return out.astype({"fold": int})


def assert_no_leakage(splits: pd.DataFrame) -> None:
    """Aucun patient en double, aucun patient commun entre dev et test."""
    if splits["PATNO"].duplicated().any():
        raise AssertionError("Un patient apparaît plusieurs fois dans la table de split.")
    dev = set(splits.loc[splits["split"] == "dev", "PATNO"])
    test = set(splits.loc[splits["split"] == "test", "PATNO"])
    if dev & test:
        raise AssertionError(f"{len(dev & test)} patient(s) à la fois en dev et en test.")
