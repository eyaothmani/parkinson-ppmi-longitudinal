"""Règles de nettoyage de la cible MDS-UPDRS III."""
import numpy as np
import pandas as pd

from parkinson_mm.data.journal import CleaningLog
from parkinson_mm.data.updrs3 import clean_updrs3
from parkinson_mm.data.updrs3_items import ALL_ITEMS
from parkinson_mm.data.visits import one_exam_per_visit


def exam(event="V04", total=10.0, state="OFF", item_value=0.0, n_bad=0, bad_code=101.0):
    items = dict.fromkeys(ALL_ITEMS, item_value)
    for name in ALL_ITEMS[:n_bad]:
        items[name] = bad_code
    return {"PATNO": 1, "EVENT_ID": event, "INFODT": "01/2020", "PDTRTMNT": 1.0, "PDSTATE": state,
            "NHY": 2.0, "NP3TOT": total, **items}


def test_code_101_is_missing_and_total_is_prorated():
    raw = pd.DataFrame([exam(total=np.nan, item_value=1.0, n_bad=2)])   # 31 items valides valant 1
    out = clean_updrs3(raw, pd.Series([1]), CleaningLog())
    assert out["np3"].iloc[0] == round(31 * 33 / 31)
    assert out["np3_imputed"].iloc[0] == 1


def test_too_many_missing_items_are_rejected():
    raw = pd.DataFrame([exam(total=np.nan, item_value=1.0, n_bad=5)])
    assert clean_updrs3(raw, pd.Series([1]), CleaningLog()).empty


def test_remote_visits_are_excluded():
    assert clean_updrs3(pd.DataFrame([exam(event="R06")]), pd.Series([1]), CleaningLog()).empty


def test_off_exam_is_preferred_over_on():
    raw = pd.DataFrame([exam(total=30.0, state="OFF"), exam(total=20.0, state="ON")])
    one = one_exam_per_visit(clean_updrs3(raw, pd.Series([1]), CleaningLog()), CleaningLog())
    assert len(one) == 1 and one["np3"].iloc[0] == 30.0
    assert one["np3_on"].iloc[0] == 20.0          # le score ON reste disponible pour la sensibilité
