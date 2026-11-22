"""Objectif 1 : score d'état actuel (niveau + vitesse) de chaque patient, validé sur les patients de test."""
from _cli import main

from parkinson_mm.pipeline import state

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: state(cfg)])
