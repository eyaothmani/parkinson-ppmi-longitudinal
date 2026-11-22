"""Échantillons multi-horizon (ADR-0018) : la cible est la mesure FUTURE la plus proche de l'horizon, à ± tolérance."""
import numpy as np
import pandas as pd

from parkinson_mm.features.horizons import add_conditions, make_horizon_samples


def follow_up():
    """Patient 1 : visites à 0, 11, 13, 25 mois ; patient 2 : 0 et 30 mois (hors fenêtre de 24 mois)."""
    months = [0, 11, 13, 25, 0, 30]
    return pd.DataFrame({
        "PATNO": [1, 1, 1, 1, 2, 2], "months_from_bl": months, "time_years": np.array(months) / 12,
        "visit_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.array(months) * 30.44, unit="D"),
        "np3": [20.0, 21.0, 23.0, 30.0, 10.0, 18.0], "updrs2": [5.0, 6.0, 7.0, 9.0, 2.0, 4.0],
        "med_state": ["untreated", "OFF", "ON", "OFF", "untreated", "ON"]})


def build():
    long = follow_up()
    return make_horizon_samples(long, long[["PATNO"]].drop_duplicates(), [12, 24], 3, ["updrs2"])


def test_closest_measure_wins_and_ties_go_to_the_earliest():
    first = build().set_index(["PATNO", "months_from_bl"]).loc[(1, 0)]
    assert first["y_12"] == 21.0                    # 11 et 13 mois sont à 1 mois de 12 : la plus précoce
    assert first["y_24"] == 30.0 and first["updrs2_24"] == 9.0
    assert first["gap_24"] == 25 / 12 and first["off_24"] == 1.0 and first["on_24"] == 0.0


def test_targets_outside_the_window_are_missing():
    samples = build()
    assert 2 not in set(samples["PATNO"])           # 30 mois : ni 12 ± 3, ni 24 ± 3 -> aucune cible
    at_13 = samples.set_index(["PATNO", "months_from_bl"]).loc[(1, 13)]
    assert at_13["y_12"] == 30.0 and np.isnan(at_13["y_24"])


def test_targets_never_come_from_the_past():
    samples = build()
    for h in (12, 24):
        has = samples[f"y_{h}"].notna()
        assert ((samples.loc[has, f"gap_{h}"] * 12) > 0).all()


def test_unknown_conditions_use_the_nominal_horizon():
    s = add_conditions(pd.DataFrame(index=[0, 1]), 36, [np.nan, 30.0], [None, "OFF"])
    assert s["gap_36"].tolist() == [3.0, 2.5]
    assert s[["off_36", "on_36", "untreated_36"]].sum(axis=1).tolist() == [0.0, 1.0]
