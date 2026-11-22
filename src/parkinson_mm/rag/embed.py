"""Embeddings : vecteurs normalisés (produit scalaire = cosinus).

`E5Embedder` : modèle local multilingue (questions en français, articles en anglais). La fiche du modèle impose les
préfixes « query: » et « passage: ». Threads et taille de lot limités : le calcul tourne sur un PC portable.
`HashEmbedder` : sac de mots haché, sans modèle ni réseau ; pour les tests et la CI seulement."""
from __future__ import annotations

import hashlib
import re

import numpy as np


class E5Embedder:
    def __init__(self, model_name: str, threads: int = 2, batch_size: int = 8):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.name, self.batch_size = model_name, batch_size
        self.model = SentenceTransformer(model_name, device="cpu")

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


class HashEmbedder:
    name = "hash-bow-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        for word in re.findall(r"\w{3,}", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vector / (np.linalg.norm(vector) or 1.0)

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._one(text)
