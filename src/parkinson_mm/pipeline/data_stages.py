"""Étapes « données » : lecture + nettoyage (ingest), puis porte de qualité (validate)."""
from __future__ import annotations

import shutil

from ..data.build import build_longitudinal
from ..data.io import find_file, load_tables
from ..data.tables import ALL
from ..data.validation import enforce, run_checks
from ..data.validation.clean import clean_checks
from ..data.validation.raw import raw_checks
from ..utils.io import write_json
from ..utils.lineage import data_manifest, lineage
from .common import logger, read_long, tables_dir


def ingest(cfg) -> dict:
    """Empreinte des CSV lus, contrat « données brutes », nettoyage journalisé -> data/interim/."""
    used = [path for key in ALL if (path := find_file(cfg.path("raw"), key)) is not None]
    manifest = data_manifest(cfg.path("raw"), used)
    write_json(manifest, cfg.path("metrics") / "data_manifest.json")
    tables = load_tables(cfg.path("raw"))
    report = enforce(run_checks(raw_checks(tables), stage="raw"))
    write_json(report, cfg.path("reports") / "validation" / "raw.json")

    long, patients, journal = build_longitudinal(tables, cfg)
    long.to_csv(cfg.path("interim") / "pd_longitudinal.csv", index=False)
    patients.to_csv(cfg.path("interim") / "pd_patients.csv", index=False)
    journal.to_csv(tables_dir(cfg, "clean") / "journal_nettoyage.csv", index=False)

    result = {"patients": int(long["PATNO"].nunique()), "visits": len(long), "tables": sorted(tables),
              "lineage": lineage(cfg, manifest["data_version"])}
    write_json(result, cfg.path("metrics") / "ingest.json")
    logger(cfg, "ingest").info("données %s : %d patients, %d visites", manifest["data_version"],
                               result["patients"], result["visits"])
    return result


def validate(cfg) -> dict:
    """Seules des données conformes au contrat arrivent dans data/processed/."""
    long, patients = read_long(cfg, "interim")
    report = enforce(run_checks(clean_checks(), long, patients, stage="clean"))
    write_json(report, cfg.path("reports") / "validation" / "clean.json")
    for name in ("pd_longitudinal.csv", "pd_patients.csv"):
        shutil.copyfile(cfg.path("interim") / name, cfg.path("processed") / name)
    result = {"passed": True, "warnings": report["warnings"]}
    write_json(result, cfg.path("metrics") / "validate.json")
    return result
