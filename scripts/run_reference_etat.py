"""Objectif 2 : persistance ajustée à l'état de l'examen (ON/OFF), la référence juste face aux 4 modèles.
Validation croisée seulement : le test n'est pas relu. Calcul léger (quelques secondes)."""
from _cli import main

from parkinson_mm.analysis import reference_state

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: reference_state(cfg)])
