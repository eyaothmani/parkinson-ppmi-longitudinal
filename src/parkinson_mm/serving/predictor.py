"""Prédicteur de service : historique de visites d'un patient -> UPDRS III prédit à la visite suivante.

Entrée : une ligne par visite passée. Obligatoires : PATNO, time_years, np3, med_state, gap_months, target_state.
Les autres variables connues du modèle sont facultatives (imputées comme à l'entraînement).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..features.groups import BIO, DAT, DEMO, GEN, MRI, STEP_CLIN
from ..features.samples import make_inference_samples
from ..features.sequences import build_sequences
from .intervals import bounds

REQUIRED = ["PATNO", "time_years", "np3", "med_state", "gap_months", "target_state"]
PATIENT_COLUMNS = [*DEMO, *BIO, *GEN, *MRI]
DISCLAIMER = "Outil de recherche (cohorte PPMI), non diagnostique : ne pas utiliser pour une décision individuelle."


class NextVisitPredictor:
    def __init__(self, model, max_seq_len: int = 12, interval=(np.nan, np.nan), metadata: dict | None = None,
                 reference: dict | None = None):
        self.model, self.max_seq_len = model, max_seq_len
        self.interval = interval              # par horizon : quantiles des résidus de CV (jamais du test)
        self.metadata = metadata or {}
        self.reference = reference or {}      # valeurs typiques du dev : point de comparaison de /explain

    def prepare(self, visits: pd.DataFrame):
        """Visites -> échantillon (dernière visite), séquences et masque, recodés exactement comme à l'entraînement."""
        missing = [c for c in REQUIRED if c not in visits]
        if missing:
            raise ValueError(f"Colonnes obligatoires manquantes : {missing}")
        v = visits.reindex(columns=sorted(set(visits.columns) | set(STEP_CLIN + DAT + PATIENT_COLUMNS)))
        v = v.sort_values(["PATNO", "time_years"]).assign(months_from_bl=lambda d: d["time_years"] * 12)
        v["visit_date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(v["months_from_bl"] * 30.44, unit="D")
        v["on_treatment"] = v["on_treatment"].fillna((v["med_state"] != "untreated").astype(float))
        patients = v.groupby("PATNO")[PATIENT_COLUMNS].last().reset_index()
        last = v.groupby("PATNO").tail(1).set_index("PATNO")
        samples, history = make_inference_samples(v.drop(columns=PATIENT_COLUMNS), patients,
                                                  last["gap_months"], last["target_state"])
        return (samples, *build_sequences(history, samples, STEP_CLIN, self.max_seq_len))

    def predict_frame(self, visits: pd.DataFrame) -> pd.DataFrame:
        samples, X, mask = self.prepare(visits)
        pred = np.asarray(self.model.predict(samples, X, mask), dtype=float)
        low, high = bounds(self.interval, samples["gap_months"])
        return pd.DataFrame({"PATNO": samples["PATNO"], "prediction": np.clip(pred, 0, 132),
                             "lower": np.clip(pred + low, 0, 132), "upper": np.clip(pred + high, 0, 132)})

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> NextVisitPredictor:
        with open(path, "rb") as f:
            return pickle.load(f)
