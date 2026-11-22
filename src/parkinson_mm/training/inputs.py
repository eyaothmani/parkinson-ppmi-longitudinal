"""Entrées des modèles appris : séquences de visites et variables statiques, prétraitées sur le train."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.groups import DEMO, HISTORY, STEP_CLIN
from ..features.preprocessing import SequenceScaler, TabularPreprocessor

STATIC = [*HISTORY, *DEMO]
TABULAR = [*STEP_CLIN, *HISTORY, *DEMO]   # LightGBM : visite courante + historique + démographie (M1)


class NetworkInputs:
    def fit(self, samples: pd.DataFrame, X: np.ndarray, mask: np.ndarray) -> NetworkInputs:
        self.seq_ = SequenceScaler().fit(X, mask)
        self.static_ = TabularPreprocessor(STATIC).fit(samples)
        return self

    def transform(self, samples: pd.DataFrame, X: np.ndarray, mask: np.ndarray) -> dict:
        return {"seq": self.seq_.transform(X, mask), "static": self.static_.transform(samples)}

    def dims(self, arrays: dict) -> dict:
        return {"step": arrays["seq"].shape[-1], "static": arrays["static"].shape[1]}
