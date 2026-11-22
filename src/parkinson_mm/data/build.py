"""Assemble le dataset longitudinal analytique : 1 ligne = 1 patient × 1 visite, + 1 table patient."""
from __future__ import annotations

import pandas as pd

from . import biology, datscan, mri
from .clinical import add_clinical_scales
from .cohort import select_cohort
from .journal import CleaningLog
from .nonmotor import add_nonmotor
from .patient import patient_table
from .treatment import add_disease_duration, add_ledd
from .updrs3 import clean_updrs3
from .visits import add_time_axis, one_exam_per_visit

VISIT_COLUMNS = ["PATNO", "EVENT_ID", "visit_date", "bl_date", "months_from_bl", "time_years", "n_visits", "np3",
                 "np3_on", "np3_off", "np3_imputed", "med_state", "on_treatment", "nhy", "tremor3", "pigd3",
                 "rigid3", "brady3", "asym3", "jump_flag"]


def build_longitudinal(tables: dict, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log, c = CleaningLog(), cfg.cleaning
    cohort = select_cohort(tables["status"], log, cfg.cohort.definition, cfg.cohort.enroll_status_keep)
    exams = clean_updrs3(tables["updrs3"], cohort["PATNO"], log, c.max_missing_items_prorate)
    v = add_time_axis(one_exam_per_visit(exams, log), log, c.jump_threshold_points, c.jump_window_months)
    v = add_clinical_scales(v[VISIT_COLUMNS], tables)
    v = add_disease_duration(v, tables.get("pd_dx_history"))
    v = add_nonmotor(add_ledd(v, tables.get("ledd"), c.ledd_max_mg), tables)
    scans = datscan.clean_scans(tables["datscan"], cohort["PATNO"], log)
    v = datscan.merge_last_scan(v, scans)
    patients = patient_table(tables, cohort[cohort["PATNO"].isin(v["PATNO"])])
    bl_dates = v.groupby("PATNO")["bl_date"].first()
    patients = patients.merge(datscan.baseline_scan(scans, bl_dates, c.datscan_bl_window_months), how="left")
    for extra in (mri.baseline_volumes(tables.get("mri_aseg")), biology.baseline_markers(tables.get("biospecimen"))):
        if extra is not None:
            patients = patients.merge(extra, on="PATNO", how="left")
    patients = add_baseline_values(patients, v)
    log.add("Dataset final", v, v, "Après fusion clinique, imagerie (as-of) et variables patient.")
    return v, patients, log.to_frame()


def add_baseline_values(p: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Valeurs à la première visite + résumé du suivi, utiles pour les analyses statistiques."""
    first = v.sort_values("visit_date").drop_duplicates("PATNO").set_index("PATNO")
    p["age_bl"] = p["PATNO"].map(first["age"]).fillna(p["ENROLL_AGE"])
    for col in ("np3", "nhy", "disease_dur_years", "moca", "ess", "gds", "scopa_aut", "stai", "ledd"):
        if col in first:
            p[f"{col}_bl"] = p["PATNO"].map(first[col])
    p = p.rename(columns={"disease_dur_years_bl": "disease_dur_bl"})
    p["followup_years"] = p["PATNO"].map(v.groupby("PATNO")["time_years"].max())
    p["n_visits"] = p["PATNO"].map(v.groupby("PATNO").size())
    return p
