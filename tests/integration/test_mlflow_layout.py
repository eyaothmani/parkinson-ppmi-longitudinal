"""ADR-0022 sur le pipeline (données synthétiques) : runs TEST et runs de validation croisée ne partagent jamais un
experiment, et chaque experiment créé porte la description du catalogue docs/mlflow_experiments.yaml."""
import mlflow

from parkinson_mm.training.experiments import NOTE


def test_test_runs_live_apart_and_every_experiment_is_described(trained, analysed):
    cfg, _ = trained
    experiments = {e.experiment_id: e for e in mlflow.search_experiments()
                   if e.name.startswith(cfg.tracking.experiment_prefix + "/")}
    runs = mlflow.search_runs(list(experiments), output_format="list")
    apart = {experiments[r.info.experiment_id].name.endswith("_test") == r.info.run_name.startswith("TEST – ")
             for r in runs if "mlflow.parentRunId" not in r.data.tags}
    assert runs and apart == {True}
    assert all(NOTE in e.tags for e in experiments.values())
