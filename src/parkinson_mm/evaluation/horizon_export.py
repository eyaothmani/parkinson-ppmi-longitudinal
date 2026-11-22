"""Export d'un tableau de métriques multi-horizon (modèle × horizon) vers MLflow et vers un JSON de métriques DVC."""
from __future__ import annotations

import numpy as np
import pandas as pd

MLFLOW_KEYS = ("MAE", "MAE_sd", "RMSE", "R2", "couverture_10_90", "part_mcid")
JSON_KEYS = ("MAE", "MAE_IC95", "MAE_sd", "MAE_sd_graines", "RMSE", "couverture_10_90", "part_mcid", "n")


def as_mlflow_metrics(table: pd.DataFrame, prefix: str = "") -> dict:
    """{cv_MAE_24m: …, cv_couverture_10_90_24m: …} : une métrique MLflow par horizon (NaN retirés ensuite)."""
    return {f"{prefix}{k}_{int(row['horizon'])}m": row[k] for _, row in table.iterrows()
            for k in MLFLOW_KEYS if k in row}


def _clean(value):
    return None if isinstance(value, float) and np.isnan(value) else value      # NaN -> null : JSON valide


def as_json(table: pd.DataFrame) -> dict:
    """{modèle: {"24_mois": {"MAE": …, …}}}."""
    return {model: {f"{int(r['horizon'])}_mois": {k: _clean(r[k]) for k in JSON_KEYS if k in r}
                    for _, r in group.iterrows()} for model, group in table.groupby("model")}
