"""Population de l'étude : patients Parkinson réellement inclus dans PPMI."""
from __future__ import annotations

import pandas as pd

from .journal import CleaningLog


def select_cohort(status: pd.DataFrame, log: CleaningLog, definition: str, keep_status: list[str]) -> pd.DataFrame:
    pd_all = status[status["COHORT_DEFINITION"] == definition]
    log.add("Cohorte PD", status, pd_all, "Question limitée à la maladie de Parkinson (témoins/prodromaux exclus).")
    kept = pd_all[pd_all["ENROLL_STATUS"].isin(keep_status)]
    # Les patients retirés restent : les exclure ne garderait que ceux qui vont bien (biais de survie).
    log.add("Statut d'inclusion valide", pd_all, kept, "Screen-failed / refus : jamais suivis.")
    return kept
