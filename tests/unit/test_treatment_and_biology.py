"""Dose de traitement (LEDD) et biomarqueurs : conventions de codage PPMI."""
import pandas as pd

from parkinson_mm.data.biology import censored_to_float
from parkinson_mm.data.treatment import add_ledd


def visit(date="2021-01-01"):
    return pd.DataFrame({"PATNO": [1], "visit_date": pd.to_datetime([date]), "on_treatment": [1.0]})


def test_comt_inhibitor_multiplies_active_levodopa():
    log = pd.DataFrame({"PATNO": [1, 1], "LEDTRT": ["Sinemet", "Entacapone"], "LEDD": ["300", "LD x 0.33"],
                        "STARTDT": ["01/2020", "01/2020"], "STOPDT": [None, None]})
    assert add_ledd(visit(), log)["ledd"].iloc[0] == 300 + 0.33 * 300


def test_stopped_treatment_is_not_counted():
    log = pd.DataFrame({"PATNO": [1], "LEDTRT": ["Sinemet"], "LEDD": ["300"], "STARTDT": ["01/2019"],
                        "STOPDT": ["06/2019"]})
    assert add_ledd(visit(), log)["ledd"].iloc[0] == 0


def test_implausible_dose_becomes_missing():
    log = pd.DataFrame({"PATNO": [1], "LEDTRT": ["X"], "LEDD": ["9000"], "STARTDT": ["01/2020"], "STOPDT": [None]})
    assert add_ledd(visit(), log, ledd_max=4000)["ledd"].isna().all()


def test_censored_values():
    assert censored_to_float(pd.Series(["<8", ">1997", "12.5"])).tolist() == [4.0, 1997.0, 12.5]
