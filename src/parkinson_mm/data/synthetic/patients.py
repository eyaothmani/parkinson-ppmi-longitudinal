"""Tables patient synthétiques : statut, démographie, antécédents, génétique, histoire de la maladie."""
from __future__ import annotations

import pandas as pd

from .common import month_year, save


def make_patients(n_pd: int, n_hc: int, rng) -> pd.DataFrame:
    n = n_pd + n_hc
    return pd.DataFrame({
        "PATNO": range(3000, 3000 + n),
        "COHORT_DEFINITION": ["Parkinson's Disease"] * n_pd + ["Healthy Control"] * n_hc,
        "ENROLL_STATUS": rng.choice(["Enrolled", "Withdrew", "Screen failed"], n, p=[0.8, 0.12, 0.08]),
        "ENROLL_AGE": rng.normal(62, 9, n).clip(35, 85).round(1),
        "start": pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), unit="D"),
    })


def write_patient_tables(out, p: pd.DataFrame, rng) -> None:
    n, ids = len(p), p["PATNO"]
    save(p.drop(columns="start"), out, "", "Participant_Status")
    save(pd.DataFrame({"PATNO": ids, "SEX": rng.binomial(1, .6, n).astype(float)}), out, "Subject_Demographics",
         "Demographics")
    save(pd.DataFrame({"PATNO": ids, "EDUCYRS": rng.integers(8, 22, n)}), out, "Subject_Demographics",
         "Socio-Economics")
    save(pd.DataFrame({"PATNO": ids, "ANYFAMPD": rng.binomial(1, .3, n).astype(float)}), out, "", "Family_History")
    tested = ids[: int(n * 0.85)]   # tout le monde n'est pas génotypé
    save(pd.DataFrame({"PATNO": tested, "LRRK2": rng.choice(["0", "G2019S"], len(tested), p=[.88, .12]),
                       "GBA": rng.choice(["0", "N409S", None], len(tested), p=[.75, .1, .15]),
                       "SNCA": "0", "APOE": rng.choice(["E3/E3", "E3/E4"], len(tested)),
                       "PATHVAR_COUNT": rng.integers(0, 2, len(tested)).astype(float)}),
         out, "Genetic_Status", "iu_genetic_consensus")
    save(pd.DataFrame({"PATNO": tested[::2], "GP2_PGS": rng.normal(0.003, 0.002, len(tested[::2]))}),
         out, "Genetic_Status", "Polygenic_Risk_Scores")
    pd_rows = p[p["COHORT_DEFINITION"] == "Parkinson's Disease"]
    diagnosis = [s - pd.DateOffset(months=int(rng.integers(1, 24))) for s in pd_rows["start"]]
    save(pd.DataFrame({"PATNO": pd_rows["PATNO"], "PDDXDT": [month_year(d) for d in diagnosis],
                       "SXDT": [month_year(d - pd.DateOffset(months=12)) for d in diagnosis]}),
         out, "Medical", "PD_Diagnosis_History")
    therapy = [s + pd.DateOffset(months=int(rng.integers(4, 30))) for s in pd_rows["start"]]
    save(pd.DataFrame({"PATNO": pd_rows["PATNO"], "LEDTRT": "Carbidopa/Levodopa IR",
                       "LEDD": rng.choice([300.0, 450.0, 600.0], len(pd_rows)),
                       "STARTDT": [month_year(d) for d in therapy], "STOPDT": None}), out, "Medical",
         "LEDD_Concomitant_Medication_Log")
