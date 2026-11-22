"""Entrées et cibles des modèles multi-horizon, prétraitées avec les statistiques du TRAIN uniquement (ADR-0018)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.groups import DEMO, STEP_CLIN
from ..features.grud import GRUDScaler
from ..features.horizons import SUMMARY, condition_columns
from ..features.preprocessing import TabularPreprocessor

TIME_INDEX = STEP_CLIN.index("time_years")


def static_columns(months, extra=()) -> list[str]:
    """Résumé, démographie, conditions de chaque horizon ; `extra` = variables multimodales (ADR-0021)."""
    return [*SUMMARY, *DEMO, *condition_columns(months), *extra]


def horizon_view(samples: pd.DataFrame, h: int) -> pd.DataFrame:
    """Colonnes de l'horizon h renommées en colonnes génériques : un LightGBM par horizon, mêmes noms de variables."""
    renamed = {f"{name}_{h}": f"h_{name}" for name in ("gap", "off", "on", "untreated")}
    return samples.rename(columns=renamed)


def tabular_columns(extra=()) -> list[str]:
    return [*STEP_CLIN, *SUMMARY, *DEMO, "h_gap", "h_off", "h_on", "h_untreated", *extra]


def variation(samples: pd.DataFrame, months) -> np.ndarray:
    """(n, horizons) : MDS-UPDRS III à l'horizon − score à l'origine ; NaN si aucune mesure dans la fenêtre."""
    return np.column_stack([samples[f"y_{h}"] - samples["np3"] for h in months]).astype(float)


class HorizonInputs:
    def __init__(self, months, auxiliary, extra=()):
        self.months, self.auxiliary, self.extra = list(months), list(auxiliary), list(extra)

    def fit(self, samples: pd.DataFrame, X: np.ndarray, visit: np.ndarray) -> HorizonInputs:
        self.seq_ = GRUDScaler().fit(X, visit)
        self.static_ = TabularPreprocessor(static_columns(self.months, self.extra)).fit(samples)
        aux = self._aux_raw(samples)
        self.aux_mean_, self.aux_std_ = np.nanmean(aux, axis=(0, 1)), np.nanstd(aux, axis=(0, 1)) + 1e-6
        return self

    def transform(self, samples: pd.DataFrame, X: np.ndarray, visit: np.ndarray) -> dict:
        return {**self.seq_.transform(X, visit, TIME_INDEX), "static": self.static_.transform(samples)}

    def auxiliary_targets(self, samples: pd.DataFrame) -> np.ndarray:
        """(n, horizons, cibles auxiliaires) standardisées ; NaN = non mesurée (exclue de la perte)."""
        return (self._aux_raw(samples) - self.aux_mean_) / self.aux_std_

    def _aux_raw(self, samples: pd.DataFrame) -> np.ndarray:
        if not self.auxiliary:
            return np.full((len(samples), len(self.months), 1), np.nan)
        return np.stack([np.column_stack([samples[f"{a}_{h}"] for a in self.auxiliary]) for h in self.months],
                        axis=1).astype(float)
