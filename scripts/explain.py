"""Importance par permutation (groupes de variables / modalités) du modèle retenu."""
from _cli import main

from parkinson_mm.pipeline import explain

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: explain(cfg)])
