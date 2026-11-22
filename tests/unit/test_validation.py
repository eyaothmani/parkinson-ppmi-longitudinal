"""Les contrats de données arrêtent le pipeline quand une hypothèse est violée."""
import pandas as pd
import pytest

from parkinson_mm.data.validation import Check, DataValidationError, enforce, run_checks
from parkinson_mm.data.validation.clean import clean_checks


def test_failing_error_check_stops_the_pipeline():
    report = run_checks([Check("toujours faux", lambda: False)], stage="test")
    with pytest.raises(DataValidationError, match="toujours faux"):
        enforce(report)


def test_warning_does_not_stop():
    report = run_checks([Check("avertissement", lambda: False, "warning")], stage="test")
    assert enforce(report)["warnings"] == 1


def test_crashing_check_counts_as_failure():
    report = run_checks([Check("plante", lambda: 1 / 0)], stage="test")
    assert not report["passed"]


def test_duplicate_visit_is_detected(visits):
    long = pd.concat([visits, visits.iloc[[0]]]).assign(nhy=2.0, sbr_age_months=0, ledd=0.0)
    report = run_checks(clean_checks(), long, long[["PATNO"]].drop_duplicates(), stage="clean")
    assert not report["passed"]
