"""Validation croisée de toutes les expériences de params.yaml, tracée dans MLflow."""
from _cli import main

from parkinson_mm.pipeline import train

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: train(cfg, fast=args.fast, only=args.only)],
         [(("--fast",), {"action": "store_true", "help": "moins d'époques et de plis (CPU, CI)"}),
          (("--only",), {"nargs": "*", "help": "noms exacts d'expériences à exécuter"})])
