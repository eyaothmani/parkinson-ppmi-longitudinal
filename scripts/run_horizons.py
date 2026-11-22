"""Sprint 5 (ADR-0018) : échantillons multi-horizon -> validation croisée -> test unique (ou une étape avec --stage).
Calcul lourd : à lancer dans l'environnement distant ou sur Colab, jamais sur un PC portable fragile."""
from _cli import main

from parkinson_mm.pipeline import evaluate_horizons, horizons, train_horizons

STEPS = {"data": lambda cfg, args: horizons(cfg),
         "train": lambda cfg, args: train_horizons(cfg, fast=args.fast, only=args.only),
         "evaluate": lambda cfg, args: evaluate_horizons(cfg, fast=args.fast)}

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: [STEPS[s](cfg, args) for s in ([args.stage] if args.stage else STEPS)]],
         [(("--stage",), {"choices": list(STEPS), "help": "n'exécuter qu'une étape"}),
          (("--fast",), {"action": "store_true", "help": "moins d'époques, de plis et de graines (CI)"}),
          (("--only",), {"nargs": "*", "help": "noms exacts de modèles à valider (étape train)"})])
