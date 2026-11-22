"""Hypothèses de l'ADR-0018, évaluées UNE fois sur le test. H1 est confirmatoire ; H2, H4 et H5 sont exploratoires.
Une concordance (H5) décrit un accord entre deux méthodes, jamais une cause."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .bootstrap import mae_difference


def differences(test: pd.DataFrame, dl: str, tabular: str, months, n_boot: int) -> list[dict]:
    """MAE(dl) − MAE(tabulaire) par horizon, IC 95 % bootstrap apparié par patient."""
    return [{"horizon": h, **mae_difference(test[test["horizon"] == h], dl, tabular, n_boot)} for h in months]


def speed_agreement(speeds: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """H5 : Spearman entre la vitesse du DL et la pente du modèle mixte, IC 95 % par bootstrap sur les patients."""
    x, y = speeds["vitesse_dl"].to_numpy(), speeds["pente_modele_mixte"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = [spearmanr(x[i], y[i]).statistic for i in rng.integers(0, len(x), (n_boot, len(x)))]
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    rho = float(spearmanr(x, y).statistic)
    return {"rho": rho, "IC95": [float(low), float(high)], "patients": len(x), "confirmee": bool(low > 0),
            "bon_accord": bool(rho >= 0.5)}


def evaluate_hypotheses(test: pd.DataFrame, dl: str, tabular: str, speeds: pd.DataFrame, h) -> dict:
    n_boot, months = 2000, list(h.months)
    diffs = differences(test, dl, tabular, months, n_boot)
    primary = next(d for d in diffs if d["horizon"] == h.primary)
    estimates = [d["diff_MAE"] for d in diffs]
    mine = test[test["model"] == dl]
    inside = (mine["y"] >= mine["q_low"]) & (mine["y"] <= mine["q_high"])
    low, high = h.coverage_range
    return {
        "H1": {"comparaison": primary["comparaison"], "horizon": h.primary, "diff_MAE": primary["diff_MAE"],
               "IC95": [primary["IC95_bas"], primary["IC95_haut"]], "confirmee": bool(primary["IC95_haut"] < 0),
               "role": "confirmatoire"},
        "H2": {"diff_MAE_par_horizon": {str(d["horizon"]): [d["diff_MAE"], d["IC95_bas"], d["IC95_haut"]]
                                        for d in diffs},
               "confirmee": bool(all(b < a for a, b in zip(estimates, estimates[1:], strict=False))),
               "role": "exploratoire"},
        "H4": {"couverture": float(inside.mean()), "intervalle_attendu": [low, high],
               "par_horizon": inside.groupby(mine["horizon"]).mean().round(4).to_dict(),
               "confirmee": bool(low <= inside.mean() <= high), "role": "exploratoire"},
        "H5": {**speed_agreement(speeds, n_boot), "role": "exploratoire"},
    }
