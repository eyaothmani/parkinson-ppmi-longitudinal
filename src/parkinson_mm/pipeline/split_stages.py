"""Étapes « split » et « features » : le découpage patient est figé AVANT la construction des échantillons."""
from __future__ import annotations

import pandas as pd

from ..data.split import assert_no_leakage, attach_split, make_patient_split
from ..data.validation import enforce, run_checks
from ..data.validation.samples import samples_checks
from ..features.samples import make_training_samples
from ..utils.io import write_json
from .common import logger, read_long


def split(cfg) -> dict:
    long, _ = read_long(cfg)
    splits = make_patient_split(long, cfg.split.test_size, cfg.split.n_folds, cfg.split.seed)
    assert_no_leakage(splits)
    splits.to_csv(cfg.path("processed") / "splits.csv", index=False)
    result = {"by_split": splits["split"].value_counts().to_dict(), "seed": cfg.split.seed}
    write_json(result, cfg.path("metrics") / "split.json")
    logger(cfg, "split").info("patients par ensemble : %s", result["by_split"])
    return result


def features(cfg) -> dict:
    long, patients = read_long(cfg)
    samples = make_training_samples(long, patients, cfg.samples.min_gap_months, cfg.samples.max_gap_months)
    samples = attach_split(samples, pd.read_csv(cfg.path("processed") / "splits.csv"))
    report = enforce(run_checks(samples_checks(long), samples, stage="samples"))
    write_json(report, cfg.path("reports") / "validation" / "samples.json")
    samples.to_csv(cfg.path("processed") / "pd_next_visit_samples.csv", index=False)
    result = {"samples": len(samples), "by_split": samples["split"].value_counts().to_dict()}
    write_json(result, cfg.path("metrics") / "features.json")
    logger(cfg, "features").info("%d échantillons", result["samples"])
    return result
