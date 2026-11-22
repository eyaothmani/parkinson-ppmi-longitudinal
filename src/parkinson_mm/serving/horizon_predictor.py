"""Prédicteur multi-horizon (ADR-0018) : historique de visites d'un patient -> MDS-UPDRS III à 12, 24 et 36 mois.

Même format d'entrée que le prédicteur de la visite suivante (une ligne par visite passée ; `target_state` = état de
mesure prévu aux visites futures ; `gap_months` est ignoré). Sortie : une ligne par patient et par horizon, médiane et
intervalle 10–90 % (NaN pour la persistance et LightGBM). Outil de recherche, non diagnostique.
"""
from __future__ import annotations

import pickle

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ..features.horizons import add_conditions
from .predictor import NextVisitPredictor


class HorizonPredictor:
    def __init__(self, model, max_seq_len: int = 12, metadata: dict | None = None):
        self.model, self.max_seq_len, self.metadata = model, max_seq_len, metadata or {}

    def prepare(self, visits: pd.DataFrame):
        """Même recodage que la visite suivante, puis délai nominal et état prévu pour chaque horizon."""
        samples, X, visit = NextVisitPredictor(None, self.max_seq_len).prepare(visits.assign(gap_months=12.0))
        for h in self.model.months:
            samples = add_conditions(samples, h, h, samples["target_state"])
        return samples, X, visit

    def predict_frame(self, visits: pd.DataFrame) -> pd.DataFrame:
        samples, X, visit = self.prepare(visits)
        q = np.clip(self.model.predict(samples, X, visit), 0, 132)
        months = self.model.months
        return pd.DataFrame({"PATNO": np.repeat(samples["PATNO"].to_numpy(), len(months)),
                             "horizon_mois": np.tile(months, len(samples)), "prediction": q[..., 1].ravel(),
                             "borne_basse_10": q[..., 0].ravel(), "borne_haute_90": q[..., 2].ravel()})

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> HorizonPredictor:
        with open(path, "rb") as f:
            return pickle.load(f)


class HorizonPyfunc(mlflow.pyfunc.PythonModel):
    """Wrapper MLflow : prétraitement et modèle livrés ensemble dans un seul artefact."""

    def load_context(self, context):
        self.predictor = HorizonPredictor.load(context.artifacts["predictor"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.predictor.predict_frame(model_input)
