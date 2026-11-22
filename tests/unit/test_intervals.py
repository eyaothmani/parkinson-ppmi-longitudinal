"""Intervalle de prédiction par horizon : plus large quand le délai est long, couverture tenue, repli sur le global."""
import numpy as np
import pytest

from parkinson_mm.serving.intervals import bounds, coverage_by_horizon, fit_intervals


def residuals(seed=0):
    rng = np.random.default_rng(seed)
    gaps = rng.choice([4.0, 9.0, 18.0], 3000)
    return rng.normal(0, gaps / 3), gaps          # l'erreur grandit avec le délai, comme dans PPMI


def test_intervals_widen_with_the_horizon():
    iv = fit_intervals(*residuals(), coverage=0.8)
    width = [iv[h][1] - iv[h][0] for h in ("0-6m", "6-12m", "12-24m")]
    assert width == sorted(width) and width[0] < width[-1]


def test_coverage_holds_on_new_data_in_every_horizon():
    iv = fit_intervals(*residuals(0), coverage=0.8)
    new, gaps = residuals(1)
    table = coverage_by_horizon(new, np.zeros_like(new), gaps, iv).set_index("horizon")
    assert table["couverture"].between(0.76, 0.84).all()


def test_missing_horizon_and_old_format_fall_back_to_global():
    low, high = bounds({"global": (-1.0, 2.0)}, [5.0, 30.0])
    assert list(low) == [-1.0, -1.0] and list(high) == [2.0, 2.0]
    low, high = bounds((-3.0, 4.0), [5.0])         # ancien prédicteur : un seul couple (basse, haute)
    assert (low[0], high[0]) == pytest.approx((-3.0, 4.0))
