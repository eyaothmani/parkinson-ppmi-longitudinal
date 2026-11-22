"""ADR-0019 : chaque run déclare ses datasets ; plis disjoints ; md5 = hash DVC ; aucune donnée copiée dans MLflow."""
import hashlib

import pandas as pd

from parkinson_mm.pipeline import run_datasets as rd
from parkinson_mm.training.datasets import dataset_inputs, file_md5


def samples():
    return pd.DataFrame({"PATNO": [1, 1, 2, 3, 4], "split": ["dev", "dev", "dev", "dev", "test"],
                         "fold": [0, 0, 1, 1, -1], "y_next": [20.0, 21.0, 30.0, 12.0, 8.0]})


def test_fold_runs_declare_disjoint_training_and_validation_sets():
    training, validation = rd.cross_validation(samples(), fold=1)
    assert (training.context, validation.context) == ("training", "validation")
    assert set(training.frame["PATNO"]).isdisjoint(validation.frame["PATNO"])
    assert "test" not in set(training.frame["split"]) | set(validation.frame["split"])


def test_test_runs_declare_dev_for_training_and_test_for_evaluation():
    dev, test = rd.test_evaluation(samples())
    assert (dev.name, test.name) == ("visite_suivante:dev", "visite_suivante:test")
    assert set(test.frame["split"]) == {"test"} and set(dev.frame["split"]) == {"dev"}


def test_inputs_carry_context_and_dvc_md5_but_no_rows(tmp_path):
    file = tmp_path / "data" / "processed" / "pd_next_visit_samples.csv"
    file.parent.mkdir(parents=True)
    samples().to_csv(file, index=False)
    (entry,) = dataset_inputs(tmp_path, rd.cross_validation(samples()))
    tags = {t.key: t.value for t in entry.tags}
    assert tags["mlflow.data.context"] == "training"
    assert tags["md5"] == hashlib.md5(file.read_bytes()).hexdigest() == file_md5(tmp_path, rd.NEXT)
    assert entry.dataset.name == "visite_suivante:dev" and entry.dataset.digest
    assert "20.0" not in entry.dataset.profile + entry.dataset.schema      # schéma et profil seulement
