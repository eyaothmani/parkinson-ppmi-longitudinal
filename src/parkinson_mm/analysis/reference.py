"""Persistance ajustée à l'état de l'examen : la référence juste de l'objectif 2.

Les modèles appris connaissent l'état de l'examen de la visite à prédire (non traité, OFF ou ON). La persistance
brute ne le connaît pas, alors qu'un examen ON est en moyenne plus bas qu'un examen OFF. Cette référence ajoute au
dernier score la variation médiane observée pour la même transition d'état (ex. « OFF -> ON »), apprise sur les
AUTRES plis : aucune information du pli évalué n'est utilisée.
"""
from __future__ import annotations

import pandas as pd

NAME = "Persistance ajustée à l'état"


def transitions(samples: pd.DataFrame) -> pd.Series:
    """État de l'examen courant -> état de l'examen à prédire, ex. « OFF -> ON »."""
    return samples["med_state"].astype(str) + " -> " + samples["target_state"].astype(str)


def group_key(p: pd.DataFrame, columns) -> pd.Series:
    """Clé de groupe texte, ex. « 24 | OFF -> ON » ; un seul groupe si aucune colonne."""
    return p[list(columns)].astype(str).agg(" | ".join, axis=1) if columns else pd.Series("", index=p.index)


def shifted_persistence(p: pd.DataFrame, by=("transition",), statistic: str = "median", within=()) -> pd.Series:
    """`p` : une ligne par prédiction hors-pli (fold, y, y_last et les colonnes `by`). Dernier score + variation
    médiane du même groupe, apprise sur les autres plis. Groupe jamais vu : variation de ces plis au même `within`
    (ex. même horizon), sinon variation globale."""
    key, base, delta = group_key(p, by), group_key(p, within), p["y"] - p["y_last"]
    pred = pd.Series(index=p.index, dtype=float)
    for fold in sorted(p["fold"].unique()):
        learn, held = p["fold"] != fold, p["fold"] == fold
        shift, fallback = (delta[learn].groupby(k[learn]).agg(statistic) for k in (key, base))
        pred[held] = p.loc[held, "y_last"] + key[held].map(shift).fillna(base[held].map(fallback))
    return pred.clip(0, 132)


def adjusted_persistence(persistence: pd.DataFrame, samples: pd.DataFrame, statistic: str = "median") -> pd.DataFrame:
    """`persistence` : prédictions hors-pli de la persistance (une ligne par échantillon, colonne `fold`).
    Retourne les mêmes échantillons, prédits par la persistance ajustée."""
    p = persistence.assign(transition=transitions(samples).reindex(persistence["sample_id"]).to_numpy())
    pred = shifted_persistence(p, ("transition",), statistic)
    return p.drop(columns="transition").assign(model=NAME, family="baseline", pred=pred)


def errors_by_transition(oof: pd.DataFrame) -> pd.DataFrame:
    """MAE de chaque modèle selon la transition d'état : d'où vient l'avantage des modèles sur la persistance ?"""
    err = oof.assign(err=(oof["pred"] - oof["y"]).abs())
    table = err.pivot_table(index="transition", columns="model", values="err", aggfunc="mean")
    n = err.drop_duplicates("sample_id").groupby("transition").size().rename("n")
    return table.join(n).sort_values("n", ascending=False).round(3).reset_index()


def changed_share(samples: pd.DataFrame, sample_ids) -> float:
    """Part des échantillons dont l'état de l'examen change entre la visite courante et la visite à prédire."""
    s = samples.loc[sample_ids]
    return float((s["med_state"] != s["target_state"]).mean())
