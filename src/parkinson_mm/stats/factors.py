"""Facteurs de baseline au niveau patient : phénotype, indicateurs binaires et versions standardisées (z)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TO_STANDARDIZE = ["updrs1_bl", "updrs2_bl", "sbr_putamen_bl", "sbr_put_asym_bl", "sbr_caudate_bl", "mri_putamen",
                  "mri_hippocampus", "mri_lat_ventricles", "mri_cortex", "prs", "EDUCYRS", "moca_bl", "ess_bl",
                  "gds_bl", "scopa_aut_bl", "stai_bl", "csf_abeta42", "csf_ab42_40", "csf_ptau181", "csf_asyn",
                  "plasma_nfl", "plasma_gfap"]


def build_patient_factors(long: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    first = long.sort_values("visit_date").drop_duplicates("PATNO").set_index("PATNO")
    p = patients.set_index("PATNO").copy()
    for col in ("updrs1", "updrs2", "tremor_score", "pigd_score"):
        p[f"{col}_bl"] = first[col]
    # Phénotype de Stebbins (2013) : ratio tremblement / PIGD
    ratio = p["tremor_score_bl"] / p["pigd_score_bl"].replace(0, np.nan)
    p["phenotype"] = np.select([(ratio >= 1.15).to_numpy(), (ratio <= 0.90).to_numpy()], ["TD", "PIGD"], "indéterminé")
    p.loc[(p["pigd_score_bl"] == 0) & (p["tremor_score_bl"] > 0), "phenotype"] = "TD"
    p["pigd_pheno"] = (p["phenotype"] == "PIGD").astype(float)
    p["hy2_bl"] = (p["nhy_bl"] >= 2).astype(float).where(p["nhy_bl"].notna())
    for col in [c for c in TO_STANDARDIZE if c in p]:   # z-score : effets comparables « par écart-type »
        p[f"z_{col}"] = (p[col] - p[col].mean()) / (p[col].std() or 1)
    return p.reset_index()
