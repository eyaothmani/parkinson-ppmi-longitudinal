"""Objectif 1 en validation croisée : le score d'état face aux persistances brute et ajustée (le test n'est pas
relu). Un modèle mixte par pli : quelques minutes."""
from _cli import main

from parkinson_mm.analysis import state_cv

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: state_cv(cfg)])
