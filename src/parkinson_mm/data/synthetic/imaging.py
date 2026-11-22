"""Imagerie et biologie synthétiques : DaTSCAN (SBR), IRM FreeSurfer, biomarqueurs à la baseline."""
from __future__ import annotations

import pandas as pd

from ..mri import REGIONS
from .common import month_year, save


def write_imaging(out, p: pd.DataFrame, rng) -> None:
    scans = []
    for i, row in p.iloc[: int(len(p) * 0.8)].iterrows():
        putamen = rng.normal(0.75 if row["COHORT_DEFINITION"] != "Healthy Control" else 1.6, 0.2)
        for k, (event, month) in enumerate([("SC", -1), ("V04", 12), ("V06", 24)]):
            sbr = max(0.05, putamen - 0.05 * k)    # la SBR baisse avec le temps
            scans.append({"PATNO": row["PATNO"], "EVENT_ID": event,
                          "DATSCAN_DATE": month_year(row["start"] + pd.DateOffset(months=month + 1)),
                          "DATSCAN_LIGAND": "123I-DaTscan" if i % 40 else "99mTc-TRODAT-1",   # quelques TRODAT
                          "DATSCAN_ANALYZED": "Yes" if (i + k) % 50 else "No",
                          "STRIATUM_REF_CWM": sbr * 1.2, "CAUDATE_L_REF_CWM": sbr * 1.3, "CAUDATE_R_REF_CWM": sbr * 1.2,
                          "PUTAMEN_L_REF_CWM": sbr, "PUTAMEN_R_REF_CWM": sbr * rng.uniform(.8, 1.1)})
    save(pd.DataFrame(scans), out, "DaTSCAN", "Xing_Core_Lab_-_Quant_SBR")
    half = p["PATNO"].iloc[::2]
    aseg = {"PATNO": half, "EVENT_ID": "BL", "EstimatedTotalIntraCranialVol": rng.normal(1.5e6, 1e5, len(half))}
    for col in {c for cols in REGIONS.values() for c in cols}:
        aseg[col] = rng.normal(4.5e5 if col == "CortexVol" else 5000, 300, len(half))
    save(pd.DataFrame(aseg), out, "MR_Analysis_Results", "FS7_ASEG_VOL")
    markers = [("ABeta42", 283, "pg/mL", 650), ("ABeta40", 283, "pg/mL", 7500), ("pTau181", 283, "pg/mL", 25),
               ("CSF Alpha-synuclein", 124, "pg/ml", 1500), ("NFL", 283, "Average (pg/ml)", 12),
               ("GFAP", 283, "Average (pg/ml)", 80)]
    bio = [{"PATNO": pid, "CLINICAL_EVENT": "BL", "TESTNAME": test, "PROJECTID": project, "UNITS": unit,
            "TESTVALUE": f"{rng.normal(mean, mean * .2):.2f}" if rng.random() > .02 else "<8"}
           for pid in half for test, project, unit, mean in markers]
    save(pd.DataFrame(bio), out, "Biospecimen_Analysis", "Current_Biospecimen_Analysis_Results")
