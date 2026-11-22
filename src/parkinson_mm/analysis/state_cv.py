"""Objectif 1 en validation croisée : pour chaque pli, le modèle mixte est réajusté sur les SEULS patients des autres
plis du dev, puis le score d'état prédit la visite suivante des patients du pli tenu à part. Les prédictions sont
donc comparables, échantillon par échantillon, à celles des 4 modèles et des deux persistances."""
from __future__ import annotations

import pandas as pd

from ..pipeline.common import read_long
from ..stats.factors import build_patient_factors
from ..stats.lme import prepare
from ..stats.state_apply import fit_state_model, forecast_samples

NAME = "Score d'état (modèle mixte) – CV"
Z80 = 1.2816                                     # quantile normal de l'intervalle à 80 % (comme l'étape state)


def state_out_of_fold(cfg, samples: pd.DataFrame, folds) -> pd.DataFrame:
    long, patients = read_long(cfg)
    d = prepare(long, build_patient_factors(long, patients), cfg.stats.min_visits)
    splits = pd.read_csv(cfg.path("processed") / "splits.csv")
    parts = []
    for fold in folds:
        learn = splits.loc[(splits["split"] == "dev") & (splits["fold"] != fold), "PATNO"]
        held = samples[(samples["split"] == "dev") & (samples["fold"] == fold)]
        model = fit_state_model(d[d["PATNO"].isin(learn)])
        parts.append(forecast_samples(model, d, held).assign(fold=fold))
    out = pd.concat(parts, ignore_index=True)
    out["inside_80"] = ((out["y"] - out["pred"]).abs() <= Z80 * out["pred_sd"]).astype(float)
    return out.assign(model=NAME, family="statistique")
