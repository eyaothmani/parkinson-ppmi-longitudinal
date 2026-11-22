"""Modèle multi-horizon entraîné (ADR-0018) : il prédit, pour chaque visite d'origine, les quantiles 10/50/90 % du
MDS-UPDRS III à 12, 24 et 36 mois. Persistance et LightGBM ne donnent que la médiane (bornes NaN)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .horizon_inputs import horizon_view


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    family: str                 # baseline | ml | dl
    algo: str                   # persistence | lgbm | mlp | grud
    options: tuple = ()         # réglages et ablations : (("lr", 0.001), ("masks", False)) ...

    @classmethod
    def from_params(cls, entry: dict) -> HorizonSpec:
        return cls(entry["name"], entry["family"], entry["algo"], tuple(sorted((entry.get("options") or {}).items())))

    def option(self, key: str, default):
        return dict(self.options).get(key, default)

    def with_options(self, **options) -> HorizonSpec:
        """Nouvelle spécification ; les options passées ici remplacent celles déjà présentes."""
        return HorizonSpec(self.name, self.family, self.algo, tuple(sorted({**dict(self.options), **options}.items())))


@dataclass
class HorizonModel:
    spec: HorizonSpec
    months: list
    estimators: list = field(default_factory=list)   # une entrée par graine (réseau, ou un LightGBM par horizon)
    inputs: object = None                              # prétraitement appris sur le train
    y_scale: float = 1.0
    info: dict = field(default_factory=dict)

    def predict_each(self, samples: pd.DataFrame, X=None, visit=None) -> list[np.ndarray]:
        """Une prédiction (n, horizons, 3) par graine : la dispersion entre graines mesure la stabilité."""
        base = samples["np3"].to_numpy(dtype=float)[:, None, None]
        if self.spec.family == "baseline":
            return [np.broadcast_to(base, (len(samples), len(self.months), 1)) * np.array([np.nan, 1.0, np.nan])]
        if self.spec.family == "ml":
            views = [horizon_view(samples, h) for h in self.months]
            out = []
            for models in self.estimators:
                delta = np.column_stack([m.predict(pre.transform(v)) for m, pre, v in zip(models, self.inputs, views,
                                                                                          strict=True)])
                out.append(base + delta[..., None] * np.array([np.nan, 1.0, np.nan]))
            return out
        from .horizon_loop import predict_quantiles

        arrays = self.inputs.transform(samples, X, visit)
        return [base + predict_quantiles(net, arrays) * self.y_scale for net in self.estimators]

    def predict(self, samples: pd.DataFrame, X=None, visit=None) -> np.ndarray:
        return np.mean(self.predict_each(samples, X, visit), axis=0)
