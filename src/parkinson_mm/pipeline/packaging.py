"""Empaquetage d'un modèle évalué : intervalle par horizon (résidus de CV), couverture sur le test, métadonnées."""
from __future__ import annotations

import pandas as pd

from ..monitoring.drift import reference_quantiles
from ..serving.intervals import coverage_by_horizon, fit_intervals
from ..serving.predictor import NextVisitPredictor
from .common import data_version


def package(cfg, model, name: str, test_mae: float, predictions, samples, dev, reference: dict):
    """Renvoie (prédicteur, couverture par horizon en tableau, couverture par horizon en dict, métadonnées)."""
    oof = pd.read_csv(cfg.path("reports") / "tables" / "models" / "cv_predictions.csv")
    oof = oof[oof["model"] == name]       # intervalle empirique par horizon : résidus de CV, jamais du test
    intervals = fit_intervals(oof["y"] - oof["pred"], oof["gap_months"], cfg.evaluation.interval_coverage)
    test = predictions[predictions["model"] == name]
    cover = coverage_by_horizon(test["y"], test["pred"], test["gap_months"], intervals)
    by_horizon = {h: round(float(c), 3) for h, c in zip(cover["horizon"], cover["couverture"], strict=True)}
    metadata = {"model": name, "test_MAE": round(test_mae, 4), "interval_coverage_test": by_horizon["global"],
                "interval_coverage_test_by_horizon": by_horizon, "interval_by_horizon": intervals,
                "data_version": data_version(cfg), "config_fingerprint": cfg.fingerprint,
                "reference_quantiles": reference_quantiles(samples.iloc[dev], ["np3", "age", "ledd"])}
    predictor = NextVisitPredictor(model, cfg.samples.max_seq_len, intervals, metadata, reference)
    return predictor, cover, by_horizon, metadata
