"""Métriques et intervalles de confiance."""
import numpy as np
import pandas as pd
import pytest

from parkinson_mm.evaluation.bootstrap import mae_ci, mae_difference, pairwise
from parkinson_mm.evaluation.metrics import regression_metrics


def test_persistence_has_zero_skill():
    y, last = np.array([10.0, 12.0, 15.0]), np.array([9.0, 12.0, 13.0])
    m = regression_metrics(y, last, last)
    assert m["skill_vs_persistance_%"] == pytest.approx(0.0)
    assert m["MAE"] == pytest.approx(1.0)


def test_perfect_prediction():
    y = np.array([10.0, 12.0, 15.0])
    assert regression_metrics(y, y - 1, y)["MAE"] == 0.0


def predictions():
    rng = np.random.default_rng(0)
    y = rng.normal(20, 5, 300)
    base = pd.DataFrame({"sample_id": range(300), "PATNO": np.arange(300) // 3, "y": y, "y_last": y})
    good = base.assign(model="bon", pred=y + rng.normal(0, 1, 300))
    bad = base.assign(model="mauvais", pred=y + rng.normal(0, 4, 300))
    return pd.concat([good, bad])


def test_better_model_has_negative_difference_with_ci_below_zero():
    diff = mae_difference(predictions(), "bon", "mauvais", n_boot=500)
    assert diff["IC95_haut"] < 0


def test_ci_contains_point_estimate():
    df = predictions()
    low, high = mae_ci(df, "bon", n_boot=500)
    mae = (df.query("model == 'bon'")["pred"] - df.query("model == 'bon'")["y"]).abs().mean()
    assert low <= mae <= high


def test_pairwise_only_flags_real_differences():
    """Deux modèles identiques ne sont jamais « distincts » ; l'IC est corrigé pour le nombre de couples."""
    df = predictions()
    twin = df[df["model"] == "bon"].assign(model="jumeau")
    table = pairwise(pd.concat([df, twin]), ["bon", "jumeau", "mauvais"], n_boot=300).set_index("comparaison")
    assert not bool(table.loc["bon − jumeau", "distincts"])
    assert bool(table.loc["bon − mauvais", "distincts"]) and bool(table.loc["jumeau − mauvais", "distincts"])
    assert table["niveau_IC"].iat[0].startswith("98.3 %")
