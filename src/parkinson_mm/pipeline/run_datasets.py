"""Quels datasets déclarer pour chaque type de run (ADR-0019) : une seule définition, utilisée par les étapes du
pipeline (mêmes sous-ensembles, mêmes empreintes pour tous les runs)."""
from __future__ import annotations

import pandas as pd

from ..training.datasets import Data

LONG = "data/processed/pd_longitudinal.csv"
NEXT = "data/processed/pd_next_visit_samples.csv"
HORIZONS = "data/processed/pd_multi_horizon_samples.csv"
LABELS = {NEXT: ("visite_suivante", "y_next"), HORIZONS: ("multi_horizon", None)}


def cross_validation(samples: pd.DataFrame, file: str = NEXT, fold=None) -> list[Data]:
    """Run d'un modèle : tout le dev ; run d'un pli : plis d'entraînement (training) + pli tenu à part (validation)."""
    label, target = LABELS[file]
    dev = samples[samples["split"] == "dev"]
    if fold is None:
        return [Data(dev, f"{label}:dev", "training", file, target)]
    fold = int(fold)
    return [Data(dev[dev["fold"] != fold], f"{label}:dev sans pli {fold}", "training", file, target),
            Data(dev[dev["fold"] == fold], f"{label}:pli {fold}", "validation", file, target)]


def out_of_fold(samples: pd.DataFrame) -> list[Data]:
    """Sprint 6 : prédictions hors-pli du dev (chaque exemple prédit par un modèle qui ne l'a pas vu) ; pas de test."""
    return [Data(samples[samples["split"] == "dev"], "visite_suivante:dev (hors-pli)", "validation", NEXT, "y_next")]


def test_evaluation(samples: pd.DataFrame, file: str = NEXT) -> list[Data]:
    """Run TEST : modèle réentraîné sur tout le dev, évalué une fois sur les patients de test."""
    label, target = LABELS[file]
    return [Data(samples[samples["split"] == "dev"], f"{label}:dev", "training", file, target),
            Data(samples[samples["split"] == "test"], f"{label}:test", "test", file, target)]


def state_model(long: pd.DataFrame, samples: pd.DataFrame, splits: pd.DataFrame) -> list[Data]:
    """Objectif 1 : modèle mixte ajusté sur les visites des patients du dev, vérifié sur la visite suivante du test."""
    dev = splits.loc[splits["split"] == "dev", "PATNO"]
    return [Data(long[long["PATNO"].isin(dev)], "longitudinal:dev", "training", LONG),
            Data(samples[samples["split"] == "test"], "visite_suivante:test", "test", NEXT, "y_next")]


def statistics(long: pd.DataFrame) -> list[Data]:
    """Objectif 3 : analyse statistique de toute la cohorte (pas de prédiction, donc pas de test)."""
    return [Data(long, "longitudinal:cohorte", "training", LONG)]
