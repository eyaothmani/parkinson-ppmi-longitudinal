"""Visites synthétiques : MDS-UPDRS I à IV, Schwab & England, âge, MoCA."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..updrs3_items import ALL_ITEMS
from .common import SCHEDULE, month_year, save


def updrs3_row(patno, event, date, total, treated, state, rng) -> dict:
    items = np.zeros(len(ALL_ITEMS))
    for _ in range(int(min(total, 120))):          # répartit le total sur les items (max 4 chacun)
        items[rng.choice(np.where(items < 4)[0])] += 1
    row = {"PATNO": patno, "EVENT_ID": event, "INFODT": month_year(date), "PDTRTMNT": float(treated),
           "PDSTATE": state, "NHY": float(min(4, 1 + total // 18)), "NP3TOT": float(total),
           **dict(zip(ALL_ITEMS, items, strict=True))}
    if event.startswith("R"):                       # visite à distance : pas d'examen
        row.update({"NP3TOT": np.nan, **{k: np.nan for k in ALL_ITEMS}})
    elif rng.random() < 0.05:                       # item non évaluable (code 101) -> total absent
        row.update({"NP3PSTBL": 101.0, "NP3TOT": np.nan})
    return row


def write_visit_tables(out, patients: pd.DataFrame, rng) -> None:
    rows = {k: [] for k in ("u3", "u1", "u1p", "u2", "u4", "se", "age", "moca")}
    for _, p in patients.iterrows():
        sick = p["COHORT_DEFINITION"] == "Parkinson's Disease"
        base, slope, therapy = (rng.normal(22, 9), rng.normal(2, 1.6), rng.uniform(.3, 2.5)) if sick else (2, .1, 99)
        for event, month in SCHEDULE[: rng.integers(3, len(SCHEDULE) + 1)]:
            t, date = max(month, 0) / 12, p["start"] + pd.DateOffset(months=month + 1)
            score, treated, key = max(0.0, base + slope * t + rng.normal(0, 4)), t >= therapy, (p["PATNO"], event)
            exams = [("OFF", score), ("ON", max(0.0, score - rng.normal(9.5, 4)))] if treated else [(None, score)]
            rows["u3"] += [updrs3_row(*key, date, round(s), treated, st, rng) for st, s in exams]
            rows["u1"].append({"PATNO": key[0], "EVENT_ID": event, "NP1RTOT": rng.integers(0, 8)})
            rows["u1p"].append({"PATNO": key[0], "EVENT_ID": event, "NP1PTOT": rng.integers(0, 10)})
            rows["u2"].append({"PATNO": key[0], "EVENT_ID": event, "NP2PTOT": int(score // 3), "NP2TRMR": 1,
                               "NP2WALK": int(score > 30), "NP2FREZ": 0})
            rows["u4"].append({"PATNO": key[0], "EVENT_ID": event, "NP4TOT": rng.integers(0, 6) if treated else 0})
            rows["se"].append({"PATNO": key[0], "EVENT_ID": event, "MSEADLG": int(np.clip(100 - score, 50, 100))})
            rows["age"].append({"PATNO": key[0], "EVENT_ID": event, "AGE_AT_VISIT": round(p["ENROLL_AGE"] + t, 1)})
            rows["moca"].append({"PATNO": key[0], "EVENT_ID": event, "MCATOT": int(np.clip(rng.normal(26, 3), 10, 30))})
    names = {"u3": "MDS-UPDRS_Part_III", "u1": "MDS-UPDRS_Part_I", "u1p": "MDS-UPDRS_Part_I_Patient_Questionnaire",
             "u2": "MDS_UPDRS_Part_II__Patient_Questionnaire", "u4": "MDS-UPDRS_Part_IV__Motor_Complications",
             "se": "Modified_Schwab___England_Activities_of_Daily_Living"}
    for key, name in names.items():
        save(pd.DataFrame(rows[key]), out, "Motor___MDS-UPDRS", name)
    save(pd.DataFrame(rows["age"]), out, "Subject_Demographics", "Age_at_visit")
    save(pd.DataFrame(rows["moca"]), out, "Non-motor_Assessments", "Montreal_Cognitive_Assessment__MoCA_")
