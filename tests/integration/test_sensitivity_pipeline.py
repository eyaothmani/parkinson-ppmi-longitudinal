"""Phase 3 de bout en bout (synthétique, mode rapide) : objectif 1 en CV, variantes des modèles, sensibilité
statistique et référence ajustée multi-horizon. Validation croisée seulement : rien ne vient du test."""
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient


def runs_with_stage(cfg, question: str, stage: str):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{cfg.tracking.experiment_prefix}/{question}")
    return client.search_runs([experiment.experiment_id], f"tags.stage = '{stage}'")


def test_state_model_is_compared_in_cross_validation(trained):
    from parkinson_mm.analysis import state_cv

    cfg, _ = trained
    result = state_cv(cfg)
    summary = pd.read_csv(cfg.path("reports") / "tables" / "etat_cv" / "cv_summary.csv")
    assert {"Persistance", "Persistance ajustée à l'état", result["modele"]} <= set(summary["model"])
    assert 0 <= result["cv_couverture_80"] <= 1 and len(runs_with_stage(cfg, "objectif2_score_futur", "etat_cv")) == 1


def test_variants_are_compared_to_their_model_and_to_persistence(trained, monkeypatch):
    from parkinson_mm.analysis import sensitivity

    cfg, _ = trained
    variant = {"retirer": ["target_off", "target_on", "target_untreated"], "modeles": ["LightGBM (M1)"],
               "face_persistance": True}
    monkeypatch.setitem(cfg, "sensibilite", {**cfg["sensibilite"], "variantes": {"sans état cible": variant}})
    result = sensitivity(cfg, fast=True)
    assert {c["comparaison"] for c in result["comparaisons"]} == {
        "LightGBM (M1) – sans état cible − LightGBM (M1)", "LightGBM (M1) – sans état cible − Persistance"}
    variants = pd.read_csv(cfg.path("reports") / "tables" / "sensibilite" / "cv_predictions.csv")
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    assert (samples.loc[variants["sample_id"], "split"] == "dev").all()
    assert len(runs_with_stage(cfg, "objectif2_score_futur", "sensibilite")) == 1


def test_stats_sensitivity_reports_time_shape_and_published_reference(prepared, analysed):
    from parkinson_mm.analysis import stats_sensitivity

    result = stats_sensitivity(prepared)
    labels = [a["analyse"] for a in result["analyses"]]
    assert any(label.startswith("De novo") for label in labels) and labels[-1].startswith("Holden")
    assert np.isfinite(result["forme_du_temps"]["LRT_p"])


def test_reference_horizons_after_the_horizon_stage(trained):
    cfg, _ = trained
    if not (cfg.path("reports") / "tables" / "horizons" / "cv_predictions.csv").exists():
        pytest.skip("étape horizons non exécutée dans cette session")
    from parkinson_mm.analysis import reference_horizons

    result = reference_horizons(cfg)
    assert {"cv_MAE_12m", "cv_MAE_24m", "cv_MAE_36m"} <= set(result)
