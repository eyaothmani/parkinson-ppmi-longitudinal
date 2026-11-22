"""Matrice de design reconstruite à partir des NOMS des coefficients statsmodels.

On évite de garder l'objet patsy du modèle (non sérialisable) : les noms suffisent pour les formules
utilisées ici — constante, variables numériques, indicatrices C(var, ...)[T.niveau] et interactions a:b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def term_column(term: str, rows: pd.DataFrame) -> np.ndarray:
    if term == "Intercept":
        return np.ones(len(rows))
    if ":" in term:                                   # interaction = produit des colonnes
        return np.prod([term_column(part, rows) for part in term.split(":")], axis=0)
    if "[T." in term:                                 # ex. C(med, Treatment('untreated'))[T.ON]
        variable = term[2:].split(",")[0].split(")")[0].strip()
        level = term.split("[T.", 1)[1][:-1]
        return (rows[variable].astype(str) == level).to_numpy(float)
    return rows[term].to_numpy(float)


def design_matrix(terms: list[str], rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack([term_column(term, rows) for term in terms])
