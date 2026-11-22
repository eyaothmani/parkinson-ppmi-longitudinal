"""Contrat sur le dataset longitudinal nettoyé (porte de qualité avant data/processed/)."""
from __future__ import annotations

from .checks import Check


def clean_checks() -> list[Check]:
    return [
        Check("1 ligne par (patient, mois)", lambda v, p: not v.duplicated(["PATNO", "visit_date"]).any()),
        Check("1 ligne par patient (table patient)", lambda v, p: not p["PATNO"].duplicated().any()),
        Check("cible NP3 présente et dans [0, 132]", lambda v, p: v["np3"].between(0, 132).all()),
        Check("Hoehn & Yahr dans [0, 5]", lambda v, p: v["nhy"].dropna().between(0, 5).all()),
        Check("temps depuis BL >= 0", lambda v, p: (v["months_from_bl"] >= 0).all()),
        Check("DaTSCAN jamais postérieur à la visite", lambda v, p: (v["sbr_age_months"].dropna() >= 0).all()),
        Check("état médicamenteux connu",
              lambda v, p: v["med_state"].isin(["untreated", "OFF", "ON", "unknown"]).all()),
        Check("patients de la table visite présents dans la table patient",
              lambda v, p: v["PATNO"].isin(p["PATNO"]).all()),
        Check("LEDD positive", lambda v, p: (v["ledd"].dropna() >= 0).all()),
        Check(">= 1000 patients (taille attendue sur données réelles)",
              lambda v, p: v["PATNO"].nunique() >= 1000, "warning"),
    ]
