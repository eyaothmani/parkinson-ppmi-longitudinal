"""Analyse statistique pré-spécifiée : modèles mixtes, criblage FDR, chaîne M1 -> M4, score d'évolution."""
from _cli import main

from parkinson_mm.pipeline import stats

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: stats(cfg)])
