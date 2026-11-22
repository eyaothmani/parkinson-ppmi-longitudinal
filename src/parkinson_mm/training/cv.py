"""Validation croisée groupée par patient : prédictions hors-pli (out-of-fold) de chaque modèle."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..evaluation.metrics import regression_metrics
from .fit import fit


def cross_validate(specs, samples: pd.DataFrame, X, mask, cfg=None, folds=None, seed: int = 42,
                   log=print, on_fold=None) -> pd.DataFrame:
    """Le test n'est JAMAIS utilisé ici : seuls les patients du dev, répartis en plis.
    `on_fold(spec, fold, metrics, info)` permet de tracer chaque pli (runs MLflow imbriqués)."""
    dev = (samples["split"] == "dev").to_numpy()
    folds = folds if folds is not None else sorted(samples.loc[dev, "fold"].unique())
    parts = []
    for spec in specs:
        for fold in folds:
            held_out = np.where(dev & (samples["fold"] == fold).to_numpy())[0]
            train = np.where(dev & (samples["fold"] != fold).to_numpy())[0]
            model = fit(spec, samples, X, mask, train, cfg, seed + int(fold))
            started = time.time()
            pred = model.predict(samples.iloc[held_out], X[held_out], mask[held_out])
            part = samples.iloc[held_out][["PATNO", "np3", "y_next", "gap_months"]].rename(
                columns={"np3": "y_last", "y_next": "y"}).assign(model=spec.name, family=spec.family, fold=fold,
                                                                 sample_id=held_out, pred=pred)
            parts.append(part)
            if on_fold:
                info = {**model.info, "predict_seconds": round(time.time() - started, 3)}
                on_fold(spec, int(fold), regression_metrics(part["y"], part["y_last"], part["pred"]), info)
        mine = pd.concat([p for p in parts if p["model"].iat[0] == spec.name])
        m = regression_metrics(mine["y"], mine["y_last"], mine["pred"])
        log(f"{spec.name:<38} MAE={m['MAE']:.3f}  R²Δ={m['R2_delta']:.3f}")
    return pd.concat(parts, ignore_index=True)
