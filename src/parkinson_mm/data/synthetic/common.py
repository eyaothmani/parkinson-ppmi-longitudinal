"""Petits utilitaires partagés par les générateurs synthétiques."""
import pandas as pd

SUFFIX = "01Jan2000"   # même convention de nommage que LONI (date en suffixe)
# (visite, mois depuis la baseline) ; R07 = visite à distance sans examen moteur
SCHEDULE = [("SC", -1), ("BL", 0), ("V01", 3), ("V02", 6), ("V03", 9), ("V04", 12), ("V05", 18), ("V06", 24),
            ("R07", 30), ("V08", 36), ("V10", 48), ("V12", 60)]


def month_year(date: pd.Timestamp) -> str:
    return f"{date.month:02d}/{date.year}"


def save(df: pd.DataFrame, out, folder: str, name: str) -> None:
    target = out / folder if folder else out
    df.to_csv(target / f"{name}_{SUFFIX}.csv", index=False)
