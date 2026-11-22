"""IRM structurelle (FreeSurfer 7) : volumes à la baseline, normalisés par le volume intracrânien."""
from __future__ import annotations

import pandas as pd

REGIONS = {
    "putamen": ["Left_Putamen", "Right_Putamen"], "caudate": ["Left_Caudate", "Right_Caudate"],
    "pallidum": ["Left_Pallidum", "Right_Pallidum"], "thalamus": ["Left_Thalamus", "Right_Thalamus"],
    "hippocampus": ["Left_Hippocampus", "Right_Hippocampus"],
    "lat_ventricles": ["Left_Lateral_Ventricle", "Right_Lateral_Ventricle"],
    "brainstem": ["Brain_Stem"], "subcort_gray": ["SubCortGrayVol"], "cortex": ["CortexVol"],
}


def baseline_volumes(aseg: pd.DataFrame | None) -> pd.DataFrame | None:
    """Volumes en pour mille du volume intracrânien (retire l'effet de la taille de la tête)."""
    if aseg is None:
        return None
    a = aseg.drop_duplicates("PATNO")
    etiv = a["EstimatedTotalIntraCranialVol"].where(lambda s: s > 0)
    out = pd.DataFrame({"PATNO": a["PATNO"]})
    for name, cols in REGIONS.items():
        out[f"mri_{name}"] = a[cols].sum(axis=1, min_count=len(cols)) / etiv * 1000
    return out
