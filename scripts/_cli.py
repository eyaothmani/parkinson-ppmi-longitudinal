"""Socle commun des scripts : lit params.yaml, exécute des étapes, renvoie un code de sortie explicite."""
from __future__ import annotations

import argparse
import sys

from parkinson_mm.data.validation import DataValidationError
from parkinson_mm.utils.config import load_config


def main(description: str, steps, options=()) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--params", default=None, help="chemin de params.yaml (défaut : racine du projet)")
    for flags, kwargs in options:
        parser.add_argument(*flags, **kwargs)
    args = parser.parse_args()
    cfg = load_config(args.params).make_dirs()
    try:
        for step in steps:
            step(cfg, args)
    except DataValidationError as exc:      # contrat de données violé : on s'arrête, sans résultat faux
        sys.exit(f"ARRÊT (contrat de données) : {exc}")
    except FileNotFoundError as exc:
        sys.exit(f"Fichier manquant : {exc} (l'étape précédente a-t-elle été exécutée ?)")
