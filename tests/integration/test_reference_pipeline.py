"""Référence juste de l'objectif 2 de bout en bout (synthétique) : CV seulement, un run dans l'experiment des 4."""
import pandas as pd
from mlflow import MlflowClient


def test_reference_stage_compares_on_cross_validation_only(trained):
    from parkinson_mm.analysis import reference_state
    from parkinson_mm.analysis.reference import NAME

    cfg, _ = trained
    result = reference_state(cfg)
    folder = cfg.path("reports") / "tables" / "reference_etat"
    summary = pd.read_csv(folder / "cv_summary.csv")
    assert NAME in set(summary["model"]) and len(summary) == 5
    oof = pd.read_csv(cfg.path("reports") / "tables" / "models" / "cv_predictions.csv")
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    assert (samples.loc[oof["sample_id"], "split"] == "dev").all()                # rien ne vient du test
    by_state = pd.read_csv(folder / "mae_par_transition.csv")
    assert by_state["n"].sum() == (oof["model"] == "Persistance").sum()
    assert 0 <= result["part_changement_etat"] <= 1 and (cfg.path("metrics") / "reference_etat.json").exists()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{cfg.tracking.experiment_prefix}/{cfg.reference_etat.question}")
    runs = client.search_runs([experiment.experiment_id], "tags.stage = 'reference_etat'")
    assert len(runs) == 1 and runs[0].info.run_name == NAME and "cv_MAE" in runs[0].data.metrics
