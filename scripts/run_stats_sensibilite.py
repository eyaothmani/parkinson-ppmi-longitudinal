"""Objectif 3 : forme du temps, H&Y ≥ 2, GBA et vitesse des patients de novo (toute la cohorte, sans
prédiction). Une dizaine de modèles mixtes : quelques minutes."""
from _cli import main

from parkinson_mm.analysis import stats_sensitivity

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: stats_sensitivity(cfg)])
