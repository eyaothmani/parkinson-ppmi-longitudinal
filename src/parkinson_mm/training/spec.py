"""Description d'une expérience (ModelSpec) et modèle entraîné prêt à prédire (FittedModel)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.baselines import persistence


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                 # baseline | ml | dl
    algo: str                   # persistence | lgbm | mlp | gru (ADR-0015)
    question: str = "default"   # experiment MLflow

    @classmethod
    def from_params(cls, entry: dict) -> ModelSpec:
        return cls(entry["name"], entry["family"], entry["algo"], entry.get("question", "default"))


@dataclass
class FittedModel:
    """Modèle + prétraitement appris ensemble : c'est l'objet sérialisé et servi par l'API."""
    spec: ModelSpec
    estimators: list = field(default_factory=list)   # 1 modèle tabulaire, ou n réseaux (ensemble de graines)
    inputs: object = None                              # prétraitement appris sur le train
    y_scale: float = 1.0
    info: dict = field(default_factory=dict)

    def predict(self, samples: pd.DataFrame, X=None, mask=None) -> np.ndarray:
        current = persistence(samples)
        if self.spec.family == "baseline":
            return current
        if self.spec.family == "ml":
            return current + self.estimators[0].predict(self.inputs.transform(samples))
        from .loop import predict

        arrays = self.inputs.transform(samples, X, mask)
        return current + np.mean([predict(net, arrays, self.y_scale) for net in self.estimators], axis=0)
