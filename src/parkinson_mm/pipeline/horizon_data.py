"""Étape « horizons » (ADR-0018) : échantillons multi-horizon sur le MÊME split patient que l'objectif 2."""
from __future__ import annotations

import pandas as pd

from ..data.split import attach_split
from ..data.validation import enforce, run_checks
from ..data.validation.horizons import horizon_checks
from ..features.groups import STEP_CLIN
from ..features.horizons import make_horizon_samples
from ..features.sequences import build_sequences
from ..utils.io import write_json
from .common import logger, read_long


def samples_path(cfg):
    return cfg.path("processed") / "pd_multi_horizon_samples.csv"


def pairs_table(samples: pd.DataFrame, months) -> dict:
    """Par horizon : paires (origine, cible), patients concernés, dont patients de test (tableau de l'ADR-0018)."""
    out = {}
    for h in months:
        with_target = samples[samples[f"y_{h}"].notna()]
        out[f"{h}_mois"] = {"paires": len(with_target), "patients": int(with_target["PATNO"].nunique()),
                            "patients_test": int(with_target.loc[with_target["split"] == "test", "PATNO"].nunique())}
    return out


def horizons(cfg) -> dict:
    long, patients = read_long(cfg)
    h = cfg.horizons
    samples = make_horizon_samples(long, patients, h.months, h.tolerance_months, h.auxiliary)
    samples = attach_split(samples, pd.read_csv(cfg.path("processed") / "splits.csv"))
    report = enforce(run_checks(horizon_checks(long, h.months, h.tolerance_months), samples, stage="horizons"))
    write_json(report, cfg.path("reports") / "validation" / "horizons.json")
    samples.to_csv(samples_path(cfg), index=False)
    result = {"visites_origine": len(samples), "par_horizon": pairs_table(samples, h.months)}
    write_json(result, cfg.path("metrics") / "horizons_data.json")
    logger(cfg, "horizons").info("%d visites d'origine : %s", len(samples), result["par_horizon"])
    return result


def load_horizon_data(cfg):
    """Échantillons + séquences de visites (valeurs manquantes conservées : le GRU-D les traite lui-même)."""
    long, _ = read_long(cfg)
    samples = pd.read_csv(samples_path(cfg))
    X, visit = build_sequences(long, samples, STEP_CLIN, max_len=cfg.samples.max_seq_len)
    return samples, X, visit
