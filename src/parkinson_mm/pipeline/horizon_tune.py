"""Réglages du multi-horizon (ADR-0018) : calculés à l'étape `train_horizons` sur les plis de réglage, figés dans
`reports/metrics/horizons_reglages.json` (versionné avant le tag `gel-multi-horizon`), relus tels quels par le test."""
from __future__ import annotations

from ..training.horizon_tuning import tune
from ..utils.io import read_json, write_json
from .common import tables_dir


def reglages_path(cfg):
    return cfg.path("metrics") / "horizons_reglages.json"


def tune_all(cfg, specs, samples, X, visit, settings, fast: bool = False) -> tuple[dict, dict]:
    """Un réglage par famille apprise présente dans `specs` (les ablations reprennent celui du GRU-D)."""
    h = cfg.horizons
    trials, folds = (2, list(cfg.fast_mode.folds)[:1]) if fast else (h.tuning.trials, list(h.tuning.folds))
    reglages, tables = {}, {}
    for spec in specs:
        if spec.algo in h.tuning.spaces and spec.algo not in reglages and not spec.options:
            reglages[spec.algo], tables[spec.algo] = tune(spec, samples, X, visit, settings, h, trials, folds,
                                                          cfg.project.seed)
            tables[spec.algo].to_csv(tables_dir(cfg, "horizons") / f"reglage_{spec.algo}.csv", index=False)
    write_json({"essais": trials, "plis": folds, "critere": "MAE moyenne des horizons", "reglages": reglages},
               reglages_path(cfg))
    return reglages, tables


def tuned(specs, reglages: dict) -> list:
    """Applique les réglages ; une ablation garde ses propres options (ex. masks = false) par-dessus."""
    return [s.with_options(**{**reglages.get(s.algo, {}), **dict(s.options)}) for s in specs]


def frozen_reglages(cfg) -> dict:
    return read_json(reglages_path(cfg))["reglages"]
