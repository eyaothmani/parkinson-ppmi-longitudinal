"""Le découpage se fait par patient : aucune fuite possible entre dev et test."""
import pandas as pd
import pytest

from parkinson_mm.data.split import assert_no_leakage, attach_split, make_patient_split


@pytest.fixture
def long():
    return pd.DataFrame({"PATNO": [p for p in range(100) for _ in range(2 + p % 5)]})


def test_no_patient_in_both_sets(long):
    splits = make_patient_split(long, test_size=0.2, n_folds=5, seed=1)
    assert_no_leakage(splits)
    assert splits["split"].value_counts()["test"] == pytest.approx(20, abs=2)


def test_split_is_deterministic(long):
    pd.testing.assert_frame_equal(make_patient_split(long, seed=3), make_patient_split(long, seed=3))


def test_folds_only_on_dev(long):
    splits = make_patient_split(long, n_folds=5)
    assert (splits.loc[splits["split"] == "test", "fold"] == -1).all()
    assert set(splits.loc[splits["split"] == "dev", "fold"]) == set(range(5))


def test_leakage_is_detected():
    with pytest.raises(AssertionError):
        assert_no_leakage(pd.DataFrame({"PATNO": [1, 1], "split": ["dev", "test"], "fold": [0, -1]}))


def test_missing_split_is_an_error():
    with pytest.raises(ValueError):
        attach_split(pd.DataFrame({"PATNO": [1, 2]}), pd.DataFrame({"PATNO": [1], "split": ["dev"], "fold": [0]}))
