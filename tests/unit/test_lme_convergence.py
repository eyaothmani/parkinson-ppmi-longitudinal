"""Objectif 3 : la convergence de chaque modèle mixte est tracée dans le tableau du criblage."""
import numpy as np
import pandas as pd

from parkinson_mm.stats.screening import screen_factors


def data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for patno in range(60):
        x, slope, age, duration = rng.normal(), rng.normal(2, 1), rng.normal(), rng.normal()
        for t in np.arange(0, 4, 0.5):
            med = "ON" if t > 2 else "untreated"
            rows.append({"PATNO": patno, "time_years": t, "med": med, "age_c": age, "male": patno % 2,
                         "dur_c": duration, "z_f": x,
                         "np3": 20 + (slope + 0.5 * x) * t - 4 * (med == "ON") + rng.normal(0, 2)})
    return pd.DataFrame(rows)


def test_screening_reports_convergence_and_recovers_the_effect():
    table = screen_factors(data(), {"Facteur simulé": "z_f"})
    assert table["converge"].dtype == bool and table["converge"].all()
    assert abs(table.loc[0, "effet_pente_pts_an"] - 0.5) < 0.4
