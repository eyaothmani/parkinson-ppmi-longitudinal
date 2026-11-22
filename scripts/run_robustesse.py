"""Sprint 6 : erreurs par sous-groupe et fiabilité de l'intervalle à 80 % du modèle retenu (objectif 2).
Validation croisée seulement : le test n'est pas relu. Calcul léger (quelques secondes)."""
from _cli import main

from parkinson_mm.pipeline import robustness

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: robustness(cfg)])
