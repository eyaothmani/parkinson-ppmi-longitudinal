"""ADR-0022 : chaque experiment écrit par le pipeline est décrit dans docs/mlflow_experiments.yaml."""
from pathlib import Path

from parkinson_mm.training.experiments import NOTE, catalog, experiment_tags, question_test
from parkinson_mm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_every_experiment_written_by_the_pipeline_is_in_the_catalog():
    cfg = load_config(ROOT / "params.yaml")
    trained = {e["question"] for e in cfg.experiments} | {cfg.horizons.question}
    written = trained | {question_test(q) for q in trained} | {"objectif1_score_actuel", "objectif3_facteurs"}
    written |= {cfg.robustesse.question, cfg.rag.question}
    assert written == set(catalog(ROOT))
    assert experiment_tags(ROOT, "objectif2_multi_horizon_test")[NOTE].startswith("Objectif 2")
    assert experiment_tags(ROOT, "inconnue") == {}
