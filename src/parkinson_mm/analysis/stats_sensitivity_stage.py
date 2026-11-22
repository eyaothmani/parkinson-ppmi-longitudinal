"""Étape « stats_sensibilite » : vérifications de l'objectif 3 demandées par l'audit (toute la cohorte)."""
from __future__ import annotations

import mlflow
import pandas as pd

from ..pipeline.common import data_version, logger, read_long, tables_dir
from ..pipeline.run_datasets import statistics
from ..stats.factors import build_patient_factors
from ..stats.lme import prepare
from ..training.tracking import log_artifact, model_run
from ..utils.io import write_json
from .stats_sensitivity import rows, time_shape


def stats_sensitivity(cfg) -> dict:
    long, patients = read_long(cfg)
    d = prepare(long, build_patient_factors(long, patients), cfg.stats.min_visits)
    shape, table = time_shape(d), pd.DataFrame(rows(d))
    folder = tables_dir(cfg, "stats_sensibilite")
    table.to_csv(folder / "sensibilite.csv", index=False)
    pd.DataFrame([shape]).to_csv(folder / "forme_du_temps.csv", index=False)
    result = {"forme_du_temps": shape, "analyses": table.round(4).to_dict("records")}
    write_json(result, cfg.path("metrics") / "stats_sensibilite.json")
    with model_run(cfg, "objectif3_facteurs", "Sensibilité : forme du temps, H&Y, GBA, de novo", {"donnees": "cohorte"},
                   {"stage": "stats_sensibilite"}, data_version(cfg), statistics(long)):
        mlflow.log_metrics({k: v for k, v in shape.items()})
        log_artifact(table, "sensibilite.csv")
    logger(cfg, "stats_sensibilite").info("temps quadratique : LRT p = %.3g", shape["LRT_p"])
    return result
