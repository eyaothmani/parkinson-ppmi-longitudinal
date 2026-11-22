"""Outils partagés par les étapes : journal, version des données, chargement des données de modélisation."""
from __future__ import annotations

import pandas as pd

from ..features.groups import STEP_CLIN
from ..features.sequences import build_sequences
from ..utils.io import read_json
from ..utils.log import get_logger


def logger(cfg, stage: str):
    return get_logger(f"parkinson_mm.{stage}", cfg.path("reports") / "logs" / f"{stage}.log")


def data_version(cfg) -> str | None:
    manifest = cfg.path("metrics") / "data_manifest.json"
    return read_json(manifest)["data_version"] if manifest.exists() else None


def tables_dir(cfg, name: str):
    folder = cfg.path("reports") / "tables" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_long(cfg, folder: str = "processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = cfg.path(folder)
    long = pd.read_csv(base / "pd_longitudinal.csv", parse_dates=["visit_date", "bl_date"])
    return long, pd.read_csv(base / "pd_patients.csv")


def load_modeling_data(cfg):
    """Échantillons + séquences de visites, reconstruits à l'identique depuis les fichiers figés."""
    long, _ = read_long(cfg)
    samples = pd.read_csv(cfg.path("processed") / "pd_next_visit_samples.csv")
    X, mask = build_sequences(long, samples, STEP_CLIN, max_len=cfg.samples.max_seq_len)
    return samples, X, mask


def specs(cfg, only=None):
    from ..training.spec import ModelSpec

    return [ModelSpec.from_params(e) for e in cfg.experiments if not only or e["name"] in only]


def device() -> str:
    from ..utils.env import hardware_report

    return hardware_report(verbose=False)["device"]
