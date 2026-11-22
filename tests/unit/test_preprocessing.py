"""Le prétraitement apprend ses statistiques sur le train seul, et reste robuste aux valeurs extrêmes."""
import numpy as np
import pandas as pd

from parkinson_mm.features.preprocessing import CLIP, TabularPreprocessor


def test_statistics_come_from_train_only():
    train, test = pd.DataFrame({"x": [1.0, 2.0, 3.0]}), pd.DataFrame({"x": [1000.0]})
    pre = TabularPreprocessor(["x"]).fit(train)
    assert pre.mean_["x"] == 2.0
    assert pre.transform(test)[0, 0] == CLIP              # valeur extrême bornée, pas de saturation


def test_missing_values_get_an_indicator():
    pre = TabularPreprocessor(["x"]).fit(pd.DataFrame({"x": [1.0, np.nan, 3.0]}))
    out = pre.transform(pd.DataFrame({"x": [np.nan]}))
    assert out.shape == (1, 2) and out[0, 1] == 1.0       # z (médiane) + indicateur « manquant »


def test_absent_column_is_tolerated():
    pre = TabularPreprocessor(["x", "y"]).fit(pd.DataFrame({"x": [1.0, 2.0]}))
    assert np.isfinite(pre.transform(pd.DataFrame({"x": [1.5]}))).all()
