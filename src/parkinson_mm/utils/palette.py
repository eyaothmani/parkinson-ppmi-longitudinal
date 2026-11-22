"""Palette unique des figures (notebooks, rapport) : lisible par les daltoniens, ordre fixe jamais cyclé.
Sans dépendance graphique : importable sans matplotlib."""

# Palette catégorielle (ordre fixe, jamais cyclé) — 3 premiers créneaux validés « toutes paires »
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
SERIES = list(C.values())
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
COHORT_COLORS = {"Parkinson's Disease": C["blue"], "Healthy Control": C["orange"], "Prodromal": C["aqua"],
                 "SWEDD": MUTED}
MED_COLORS = {"untreated": C["blue"], "OFF": C["orange"], "ON": C["aqua"], "unknown": MUTED}
SEQ = "Blues"
DIV = "RdBu_r"
