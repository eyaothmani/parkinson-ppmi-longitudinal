"""Journal de nettoyage : chaque règle laisse une trace (effectifs avant/après + justification)."""
from __future__ import annotations

import pandas as pd


class CleaningLog:
    def __init__(self):
        self.steps: list[dict] = []

    def add(self, step: str, before, after, why: str) -> None:
        def count(x):
            return (len(x), x["PATNO"].nunique()) if isinstance(x, pd.DataFrame) else (x, None)

        (rows_b, pat_b), (rows_a, pat_a) = count(before), count(after)
        self.steps.append({"etape": step, "lignes_avant": rows_b, "lignes_apres": rows_a,
                           "patients_avant": pat_b, "patients_apres": pat_a, "justification": why})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)
