"""Génère le jeu SYNTHÉTIQUE au schéma PPMI (CI et démonstration ; aucune donnée réelle)."""
import argparse

from parkinson_mm.data.synthetic import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--patients", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("écrit :", generate(args.out, n_pd=args.patients, seed=args.seed))
