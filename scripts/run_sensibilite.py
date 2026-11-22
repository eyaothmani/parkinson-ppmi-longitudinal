"""Objectif 2 : mêmes modèles sans l'état de l'examen cible, et modèles « une seule visite » (RQ1).
Validation croisée seulement : le test n'est pas relu. Réseaux réentraînés pli par pli : pas sur un PC fragile."""
from _cli import main

from parkinson_mm.analysis import sensitivity

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: sensitivity(cfg, fast=args.fast)],
         [(("--fast",), {"action": "store_true", "help": "moins d'époques et de plis (CI)"})])
