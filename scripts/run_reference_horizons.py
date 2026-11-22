"""Sprint 5 relu : persistance ajustée à l'état à 12, 24 et 36 mois face aux modèles multi-horizon.
Validation croisée seulement : le test n'est pas relu. Calcul léger (quelques secondes)."""
from _cli import main

from parkinson_mm.analysis import reference_horizons

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: reference_horizons(cfg)])
