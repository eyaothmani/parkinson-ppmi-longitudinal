"""Importance par permutation, par GROUPE de variables (et donc par modalité).

On mélange les valeurs BRUTES d'un groupe entre échantillons et on mesure la hausse de MAE. La même méthode
s'applique au LightGBM et aux réseaux, ce qui rend les résultats comparables. C'est une mesure d'utilité
pour la prédiction, pas une preuve de causalité. SHAP écarté : coûteux et redondant ici (ADR-0007).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.groups import BIO, DAT, DEMO, GEN, MRI, STEP_CLIN

GROUPS = {
    "UPDRS III (historique)": ["np3_prev", "np3_slope_hist", "np3_mean_hist"],
    "Autres échelles cliniques": ["nhy", "updrs1", "updrs2", "updrs4", "schwab", "tremor_score", "pigd_score"],
    "Non moteur": ["moca", "ess", "gds", "scopa_aut", "stai"],
    "Traitement / état de mesure": ["on_treatment", "med_off", "med_on", "ledd", "target_off", "target_on",
                                    "target_untreated"],
    "Temps": ["gap_years", "time_years", "n_prior", "disease_dur_years"],
    "Démographie": [*DEMO, "age"], "Biologie": BIO, "DaTSCAN": DAT, "IRM": MRI, "Génétique": GEN,
}
SEQUENCE_CHANNELS = [*STEP_CLIN, "dt_prev"]


def permutation_importance(model, samples: pd.DataFrame, X, mask, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    rng, y = np.random.default_rng(seed), samples["y_next"].to_numpy()
    reference = np.abs(model.predict(samples, X, mask) - y).mean()
    rows = []
    for group, cols in GROUPS.items():
        tabular = [c for c in cols if c in samples]
        channels = [i for i, c in enumerate(SEQUENCE_CHANNELS) if c in cols]
        if not tabular and not channels:
            continue
        increases = []
        for _ in range(n_repeats):
            order = rng.permutation(len(samples))
            shuffled = samples.copy()
            shuffled[tabular] = samples[tabular].to_numpy()[order]
            X_shuffled = X.copy() if X is not None else None
            if channels and X is not None:
                X_shuffled[:, :, channels] = X[order][:, :, channels]
            increases.append(np.abs(model.predict(shuffled, X_shuffled, mask) - y).mean() - reference)
        rows.append({"groupe": group, "hausse_MAE": float(np.mean(increases)), "ecart_type": float(np.std(increases))})
    return pd.DataFrame(rows).sort_values("hausse_MAE", ascending=False).reset_index(drop=True)
