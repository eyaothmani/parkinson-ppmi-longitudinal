"""Préparation des données : ingest -> validate -> split -> features (ou une seule étape avec --stage)."""
from _cli import main

from parkinson_mm.pipeline import features, ingest, split, validate

STEPS = {"ingest": ingest, "validate": validate, "split": split, "features": features}

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: [STEPS[s](cfg) for s in ([args.stage] if args.stage else STEPS)]],
         [(("--stage",), {"choices": list(STEPS), "help": "n'exécuter qu'une étape"})])
