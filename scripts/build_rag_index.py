"""Assistant documentaire : construit l'index (passages + embeddings locaux) à partir de rag/manifest.yaml.
Calcul léger (quelques centaines de passages, 2 threads) ; le modèle (≈ 470 Mo) est téléchargé une seule fois."""
from _cli import main

from parkinson_mm.rag import build_rag_index

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: build_rag_index(cfg)])
