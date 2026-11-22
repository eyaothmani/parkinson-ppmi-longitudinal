"""Phase 3 : neutralisation des variables, plan de comparaisons et persistance ajustée par horizon."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parkinson_mm.analysis.horizon_reference import adjusted_horizons, horizon_states
from parkinson_mm.analysis.sensitivity import CHANNELS, neutralize, planned_pairs, variant_specs
from parkinson_mm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_neutralized_columns_carry_no_information_and_originals_are_untouched():
    samples = pd.DataFrame({"target_on": [1.0, 0.0], "np3": [20.0, 30.0]})
    X = np.ones((2, 3, len(CHANNELS)), dtype=np.float32)
    s, Xn = neutralize(samples, X, ["target_on", "dt_prev"])
    assert s["target_on"].isna().all() and s["np3"].tolist() == [20.0, 30.0]
    assert np.isnan(Xn[..., CHANNELS.index("dt_prev")]).all() and not np.isnan(X).any()
    assert not np.isnan(Xn[..., CHANNELS.index("np3")]).any()


def test_variants_and_planned_comparisons_follow_params():
    cfg = load_config(ROOT / "params.yaml")
    specs = variant_specs(cfg, "sans état cible", cfg.sensibilite.variantes["sans état cible"]["modeles"])
    assert [s.algo for s in specs] == ["lgbm", "mlp", "gru"]
    assert all(s.name.endswith(" – sans état cible") for s in specs)
    pairs = planned_pairs(cfg)
    assert ("GRU – trajectoire (M1) – sans état cible", "Persistance") in pairs
    assert ("LightGBM (M1) – une seule visite", "LightGBM (M1)") in pairs and len(pairs) == 8


def test_horizon_states_and_shift_learned_per_horizon_on_other_folds():
    samples = pd.DataFrame({"med_state": ["OFF"] * 4, "off_12": 0, "on_12": 1, "untreated_12": 0,
                            "off_24": 1, "on_24": 0, "untreated_24": 0})
    samples.loc[3, "on_12"] = 0                                         # état inconnu à 12 mois
    assert horizon_states(samples, [12, 24])["target_state"].tolist() == ["ON"] * 3 + ["unknown"] + ["OFF"] * 4
    y = {(0, 12): 20, (1, 12): 20, (2, 12): 26, (3, 12): 26, (0, 24): 34, (1, 24): 34, (2, 24): 40, (3, 24): 40}
    oof = pd.DataFrame([{"PATNO": i, "sample_id": i, "horizon": h, "y": v, "y_last": 30.0, "pred": 30.0,
                         "model": "Persistance", "family": "baseline", "fold": int(i >= 2)} for (i, h), v in y.items()])
    out = adjusted_horizons(oof, samples, [12, 24]).set_index(["sample_id", "horizon"])["pred"]
    assert out[(0, 12)] == pytest.approx(26.0) and out[(2, 12)] == pytest.approx(20.0)   # OFF -> ON, autre pli
    assert out[(0, 24)] == pytest.approx(40.0) and out[(2, 24)] == pytest.approx(34.0)   # OFF -> OFF, autre pli
    assert out[(3, 12)] == pytest.approx(20.0)              # transition jamais vue à 12 mois : variation globale
