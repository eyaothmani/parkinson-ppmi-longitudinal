"""Échelles non motrices par visite, avec les règles de cotation standard de chaque questionnaire."""
from __future__ import annotations

import pandas as pd

GDS_REVERSED = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY"]
STAI_REVERSED = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30, 33, 34, 36, 39]


def _bounded(df: pd.DataFrame, cols: list[str], low: float, high: float) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.where((x >= low) & (x <= high))


def scores(tables: dict) -> dict[str, pd.DataFrame]:
    out = {}
    if "moca" in tables:
        out["moca"] = _bounded(tables["moca"], ["MCATOT"], 0, 30)["MCATOT"]
    if "ess" in tables:
        out["ess"] = _bounded(tables["ess"], [f"ESS{i}" for i in range(1, 9)], 0, 3).sum(axis=1, min_count=8)
    if "gds" in tables:
        items = [c for c in tables["gds"].columns if c.startswith("GDS")]
        x = _bounded(tables["gds"], items, 0, 1)
        x[GDS_REVERSED] = 1 - x[GDS_REVERSED]
        out["gds"] = x.sum(axis=1, min_count=len(items))
    if "scopa" in tables:   # items 1-21 ; code 9 (« utilise une sonde ») compte comme 3
        x = tables["scopa"][[f"SCAU{i}" for i in range(1, 22)]].apply(pd.to_numeric, errors="coerce").replace(9, 3)
        out["scopa_aut"] = x.where((x >= 0) & (x <= 3)).sum(axis=1, min_count=21)
    if "stai" in tables:
        x = _bounded(tables["stai"], [f"STAIAD{i}" for i in range(1, 41)], 1, 4)
        x[[f"STAIAD{i}" for i in STAI_REVERSED]] = 5 - x[[f"STAIAD{i}" for i in STAI_REVERSED]]
        out["stai"] = x.sum(axis=1, min_count=40)
    return out


def scores_table(tables: dict) -> pd.DataFrame | None:
    """Une ligne par (patient, visite) avec toutes les échelles non motrices disponibles."""
    key = {"moca": "moca", "ess": "ess", "gds": "gds", "scopa_aut": "scopa", "stai": "stai"}
    merged = None
    for name, values in scores(tables).items():
        t = tables[key[name]]
        # Certaines échelles (MoCA) sont faites au screening : SC est rattaché à BL.
        frame = pd.DataFrame({"PATNO": t["PATNO"], "EVENT_ID": t["EVENT_ID"].replace({"SC": "BL"}), name: values})
        frame = frame.groupby(["PATNO", "EVENT_ID"], as_index=False).mean()
        merged = frame if merged is None else merged.merge(frame, on=["PATNO", "EVENT_ID"], how="outer")
    return merged


def add_nonmotor(v: pd.DataFrame, tables: dict) -> pd.DataFrame:
    table = scores_table(tables)
    return v if table is None else v.merge(table, on=["PATNO", "EVENT_ID"], how="left")
