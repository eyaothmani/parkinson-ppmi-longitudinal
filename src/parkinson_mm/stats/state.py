"""Objectif 1 — score d'état actuel : où en est le patient AUJOURD'HUI ?

Le score mesuré à une visite est bruité (fluctuations, évaluateur) et baisse sous traitement (examen ON).
Le score d'état est ce que le modèle mixte estime pour CE patient, en condition « sans traitement »,
à partir de ses visites passées uniquement (filtrage : il se met à jour à chaque nouvelle visite) :
    niveau(t) = x(t)·β + b0 + b1·t          vitesse = β_temps + b1   (points UPDRS III / an)
avec b = (b0, b1) estimé par BLUP :  E[b | y] = D Zᵀ V⁻¹ (y − Xβ),  V = Z D Zᵀ + σ² I.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import design_matrix

REFERENCE = "untreated"          # condition de référence du niveau : sans effet symptomatique du traitement


class StateModel:
    """β, D et σ² d'un modèle mixte déjà ajusté (sur le dev). Objet simple, donc sérialisable pour l'API."""

    def __init__(self, result, fill: dict):
        self.terms, self.beta = list(result.fe_params.index), result.fe_params.to_numpy()
        self.D, self.sigma2 = result.cov_re.to_numpy(), float(result.scale)
        self.fill = {k: float(v) for k, v in fill.items()}   # covariable de base manquante -> médiane du dev

    def design(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = design_matrix(self.terms, rows.fillna(self.fill))
        Z = np.column_stack([np.ones(len(rows)), rows["time_years"].to_numpy(float)])
        return X, Z

    def posterior(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Moyenne et covariance des effets aléatoires du patient, sachant ses visites passées."""
        X, Z = self.design(history)
        V = Z @ self.D @ Z.T + self.sigma2 * np.eye(len(Z))
        K = self.D @ Z.T @ np.linalg.inv(V)
        return K @ (history["np3"].to_numpy(float) - X @ self.beta), self.D - K @ Z @ self.D

    def predict(self, history: pd.DataFrame, at: pd.DataFrame, noise: bool) -> tuple[np.ndarray, np.ndarray]:
        """Moyenne et écart-type à des instants `at` ; noise=True ajoute le bruit de mesure (visite future)."""
        return self._at(*self.posterior(history), at, noise)

    def _at(self, b, P, at: pd.DataFrame, noise: bool) -> tuple[np.ndarray, np.ndarray]:
        X, Z = self.design(at)
        var = np.einsum("ij,jk,ik->i", Z, P, Z) + (self.sigma2 if noise else 0.0)
        return X @ self.beta + Z @ b, np.sqrt(var)

    def state(self, history: pd.DataFrame) -> dict:
        """Score d'état à la dernière visite connue : niveau de référence et vitesse, avec leurs écarts-types."""
        b, P = self.posterior(history)
        now = history.tail(1).assign(med=REFERENCE)
        level, level_sd = self._at(b, P, now, noise=False)
        return {"time_years": float(now["time_years"].iloc[0]), "np3_mesure": float(history["np3"].iloc[-1]),
                "niveau_actuel": float(level[0]), "niveau_sd": float(level_sd[0]),
                "vitesse": float(self.beta[self.terms.index("time_years")] + b[1]),
                "vitesse_sd": float(np.sqrt(P[1, 1]))}
