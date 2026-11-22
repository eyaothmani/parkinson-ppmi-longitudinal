"""Évaluation unique sur le test, IC bootstrap patient, enregistrement MLflow et promotion objective."""
from _cli import main

from parkinson_mm.pipeline import evaluate

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: evaluate(cfg, fast=args.fast)], [(("--fast",), {"action": "store_true"})])
