"""De bout en bout sur données synthétiques : ingest -> validate -> split -> features."""
import pandas as pd

from parkinson_mm.data.split import assert_no_leakage
from parkinson_mm.utils.io import read_json


def test_processed_files_exist(prepared):
    for name in ("pd_longitudinal.csv", "pd_patients.csv", "splits.csv", "pd_next_visit_samples.csv"):
        assert (prepared.path("processed") / name).exists()


def test_quality_gates_passed(prepared):
    for stage in ("raw", "clean", "samples"):
        assert read_json(prepared.path("reports") / "validation" / f"{stage}.json")["passed"]


def test_no_patient_leakage(prepared):
    splits = pd.read_csv(prepared.path("processed") / "splits.csv")
    assert_no_leakage(splits)
    samples = pd.read_csv(prepared.path("processed") / "pd_next_visit_samples.csv")
    assert samples.groupby("PATNO")["split"].nunique().max() == 1


def test_only_parkinson_patients_are_kept(prepared):
    long = pd.read_csv(prepared.path("processed") / "pd_longitudinal.csv")
    assert long["PATNO"].max() < 3000 + 120             # les témoins synthétiques ont des PATNO >= 3120


def test_data_version_is_traced(prepared):
    assert len(read_json(prepared.path("metrics") / "data_manifest.json")["data_version"]) == 12
