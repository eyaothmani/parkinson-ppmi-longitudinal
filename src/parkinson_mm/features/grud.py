"""Entrées du GRU-D (Che et al., 2018) : chaque variable garde la trace de son ABSENCE, au lieu d'une imputation muette.

Pour chaque visite et chaque variable : valeur standardisée (0 si non mesurée), masque (1 = mesurée), temps écoulé
depuis la dernière mesure de cette variable (années) et dernière valeur observée. Statistiques apprises sur le train ;
après standardisation, la moyenne empirique vaut 0, valeur vers laquelle le GRU-D fait décroître une mesure ancienne.
"""
from __future__ import annotations

import warnings

import numpy as np

from .preprocessing import CLIP

MAX_ELAPSED_YEARS = 10.0


def elapsed(t: np.ndarray, m: np.ndarray, visit: np.ndarray) -> np.ndarray:
    """δ_k = (t_k − t_{k−1}) + δ_{k−1} si la variable manquait à k−1, sinon (t_k − t_{k−1}) ; 0 à la première visite."""
    delta = np.zeros(m.shape, dtype=np.float32)
    for k in range(1, m.shape[1]):
        step = np.nan_to_num(t[:, k] - t[:, k - 1])[:, None]
        carried = step + np.where(m[:, k - 1], 0.0, delta[:, k - 1])
        delta[:, k] = np.where(visit[:, k - 1, None] > 0, carried, 0.0)
    return np.minimum(delta, MAX_ELAPSED_YEARS)


def forward_fill(z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Dernière valeur observée de chaque variable (0 = moyenne du train tant qu'aucune mesure n'existe)."""
    last = np.zeros_like(z)
    for k in range(z.shape[1]):
        last[:, k] = np.where(m[:, k], z[:, k], last[:, k - 1] if k else 0.0)
    return last


class GRUDScaler:
    def fit(self, X: np.ndarray, visit: np.ndarray) -> GRUDScaler:
        observed = X[visit.astype(bool)]
        with warnings.catch_warnings():   # variable jamais observée : moyenne 0, écart-type 1
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nan_to_num(np.nanmean(observed, axis=0))
            std = np.nan_to_num(np.nanstd(observed, axis=0))
        self.std_ = np.where(std > 1e-6, std, 1.0)
        return self

    def transform(self, X: np.ndarray, visit: np.ndarray, time_index: int) -> dict:
        m = ~np.isnan(X) & (visit[..., None] > 0)
        z = np.where(m, np.clip((np.nan_to_num(X) - self.mean_) / self.std_, -CLIP, CLIP), 0.0).astype(np.float32)
        return {"x": z, "m": m.astype(np.float32), "delta": elapsed(X[..., time_index], m, visit),
                "x_last": forward_fill(z, m), "visit": visit.astype(np.float32)}
