"""Wrapper MLflow pyfunc : le prétraitement et le modèle sont livrés ensemble dans un seul artefact."""
from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

from .predictor import NextVisitPredictor


class NextVisitPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = NextVisitPredictor.load(context.artifacts["predictor"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.predictor.predict_frame(model_input)


def example_input(n_visits: int = 3) -> pd.DataFrame:
    """Exemple 100 % fictif : sert de signature MLflow et de documentation du format d'entrée.
    PATNO reste entier : identifiant jamais manquant (l'avertissement MLflow sur les colonnes entières
    vise les variables pouvant contenir des NaN ; toutes les mesures sont donc en float)."""
    t = np.arange(n_visits) * 0.5
    return pd.DataFrame({"PATNO": 1, "time_years": t, "np3": 20 + 2 * t,
                         "med_state": ["untreated"] + ["OFF"] * (n_visits - 1), "nhy": 2.0, "updrs2": 6.0,
                         "age": 62 + t, "male": 1.0, "age_bl": 62.0, "gap_months": 12.0, "target_state": "OFF"})
