"""Sprint 6 de bout en bout (synthétique) : tableaux, figures, métriques et run MLflow, sur la CV seulement."""
import pandas as pd
from mlflow import MlflowClient


def test_robustness_stage_writes_results_and_one_run(trained):
    from parkinson_mm.pipeline import robustness

    cfg, _ = trained
    result = robustness(cfg)
    errors = pd.read_csv(cfg.path("reports") / "tables" / "robustesse" / "mae_par_groupe.csv")
    assert {"critère", "groupe", "modèle", "MAE", "IC95_bas", "IC95_haut"} <= set(errors.columns)
    assert set(errors["modèle"]) == {cfg.robustesse.modele, cfg.robustesse.reference}
    assert 0 < result["couverture_globale"] <= 1 and (cfg.path("metrics") / "robustesse.json").exists()
    assert len(list((cfg.path("reports") / "figures" / "robustesse").glob("*.png"))) == 3
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    dev = set(samples.loc[samples["split"] == "dev", "PATNO"])
    assert errors["patients"].max() <= len(dev)                      # rien ne vient du test
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{cfg.tracking.experiment_prefix}/{cfg.robustesse.question}")
    assert len(client.search_runs([experiment.experiment_id])) == 1
