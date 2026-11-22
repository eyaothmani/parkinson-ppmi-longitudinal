"""Prétraitements appris sur le TRAIN uniquement, puis appliqués tels quels à la validation et au test."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

CLIP = 5.0   # z-scores bornés : une valeur extrême (ex. dosage censuré) ne doit pas saturer un réseau


class TabularPreprocessor:
    """Imputation par la médiane + standardisation + indicateurs « valeur manquante »."""

    def __init__(self, columns: list[str], add_indicators: bool = True):
        self.columns, self.add_indicators = columns, add_indicators

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        x = df.reindex(columns=self.columns).astype(float)
        self.median_ = x.median()
        filled = x.fillna(self.median_)
        self.mean_, self.std_ = filled.mean(), filled.std().replace(0, 1).fillna(1)
        self.flagged_ = [c for c in self.columns if x[c].isna().any()] if self.add_indicators else []
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.columns).astype(float)
        flags = x[self.flagged_].isna().to_numpy(float)
        z = ((x.fillna(self.median_) - self.mean_) / self.std_).fillna(0).clip(-CLIP, CLIP).to_numpy()
        return np.hstack([z, flags]).astype(np.float32)


class SequenceScaler:
    """Même logique pour les séquences : statistiques calculées sur les visites réellement observées."""

    def fit(self, X: np.ndarray, mask: np.ndarray) -> SequenceScaler:
        observed = X[mask.astype(bool)]
        with warnings.catch_warnings():   # une variable jamais observée donne une médiane NaN -> 0
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nan_to_num(np.nanmedian(observed, axis=0))
        filled = np.where(np.isnan(observed), self.median_, observed)
        self.mean_, self.std_ = filled.mean(0), filled.std(0) + 1e-6
        return self

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = np.clip((np.where(np.isnan(X), self.median_, X) - self.mean_) / self.std_, -CLIP, CLIP)
        return (z * mask[..., None]).astype(np.float32)
