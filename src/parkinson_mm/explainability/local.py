"""Explication LOCALE par groupe de variables (ADR-0016) : on donne à un groupe la valeur typique des patients
d'entraînement (médiane ; état ON/OFF le plus fréquent) et on mesure de combien la prédiction change. Effet > 0 : ce
groupe pousse vers un score plus élevé qu'un profil typique. Même méthode pour les 4 modèles ; non additif, pas causal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .permutation import GROUPS, SEQUENCE_CHANNELS

ONE_HOT = [["target_off", "target_on", "target_untreated"], ["med_off", "med_on"]]   # un état n'est jamais « à moitié »
METHOD = ("Effet = prédiction réelle − prédiction si les valeurs renseignées de ce groupe prenaient la valeur typique "
          "des patients d'entraînement (points de MDS-UPDRS III) ; une variable non saisie n'a pas d'effet compté. "
          "Non additif ; ce que le modèle utilise, pas une cause.")


def reference_values(samples: pd.DataFrame, X: np.ndarray, mask: np.ndarray) -> dict:
    """Valeurs typiques du dev : médianes des colonnes des groupes et des canaux des séquences (visites observées)."""
    columns = list(dict.fromkeys(c for cols in GROUPS.values() for c in cols if c in samples))
    tabular = samples[columns].apply(pd.to_numeric, errors="coerce").median().to_dict()
    for block in ONE_HOT:
        present = [c for c in block if c in samples]
        if present:
            mode = samples[present].astype(float).value_counts().idxmax()
            tabular.update(zip(present, mode if isinstance(mode, tuple) else (mode,), strict=True))
    with warnings.catch_warnings():                 # un canal jamais observé -> NaN, ignoré
        warnings.simplefilter("ignore", RuntimeWarning)
        sequence = dict(zip(SEQUENCE_CHANNELS, np.nanmedian(X[mask.astype(bool)], axis=0).tolist(), strict=True))
    sequence.update({c: tabular[c] for c in ONE_HOT[1] if c in tabular})
    clean = lambda d: {k: float(v) for k, v in d.items() if pd.notna(v)}  # noqa: E731
    return {"tabular": clean(tabular), "sequence": clean(sequence)}


def neutralize(samples, X, mask, columns, reference) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Seules les valeurs RENSEIGNÉES prennent la valeur typique : une variable non saisie reste non saisie
    (sinon l'effet mesuré serait celui de l'indicateur « valeur manquante », pas celui du patient)."""
    s, Xn = samples.copy(), X.copy()
    tab = [c for c in columns if c in s and c in reference["tabular"]]
    for c in tab:
        s[c] = s[c].where(s[c].isna(), reference["tabular"][c])
    seq = [(i, reference["sequence"][c]) for i, c in enumerate(SEQUENCE_CHANNELS)
           if c in columns and c in reference["sequence"]]
    for i, value in seq:
        Xn[..., i] = np.where((mask > 0) & ~np.isnan(X[..., i]), value, X[..., i])
    return s, Xn, bool(tab or seq)


def explain_local(model, samples, X, mask, reference) -> pd.DataFrame:
    """Effet moyen de chaque groupe sur la prédiction de ces échantillons, du plus fort au plus faible (|effet|)."""
    base, rows = np.asarray(model.predict(samples, X, mask), dtype=float), []
    for group, columns in GROUPS.items():
        s, Xn, used = neutralize(samples, X, mask, columns, reference)
        if used:
            other = np.asarray(model.predict(s, Xn, mask), dtype=float)
            rows.append({"groupe": group, "effet": float(np.mean(base - other))})
    out = pd.DataFrame(rows)
    return out.reindex(out["effet"].abs().sort_values(ascending=False, kind="stable").index).reset_index(drop=True)
