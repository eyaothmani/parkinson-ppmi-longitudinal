"""Index FAISS exact (produit scalaire sur vecteurs normalisés) + passages en JSON lisible.
Pas de pickle : l'index se relit sans exécuter de code, contrairement à la sauvegarde FAISS de LangChain."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

FILES = ("index.faiss", "passages.jsonl", "info.json")


def build_index(chunks: list[dict], embedder, out_dir: Path, info: dict) -> dict:
    vectors = embedder.passages([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / FILES[0]))
    lines = (json.dumps(c, ensure_ascii=False) for c in chunks)
    (out_dir / FILES[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = {**info, "embedding_model": embedder.name, "dim": int(vectors.shape[1]), "n_passages": len(chunks),
            "n_documents": len({c["doc_id"] for c in chunks})}
    (out_dir / FILES[2]).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


class Index:
    def __init__(self, folder: Path, embedding_model: str | None = None):
        folder = Path(folder)
        self.info = json.loads((folder / FILES[2]).read_text(encoding="utf-8"))
        if embedding_model and self.info["embedding_model"] != embedding_model:
            raise ValueError(f"Index construit avec {self.info['embedding_model']}, params.yaml demande "
                             f"{embedding_model} : relancer `dvc repro rag_index`.")
        self.index = faiss.read_index(str(folder / FILES[0]))
        lines = (folder / FILES[1]).read_text(encoding="utf-8").splitlines()
        self.passages = [json.loads(line) for line in lines if line]

    def search(self, vector: np.ndarray, k: int) -> list[dict]:
        scores, ids = self.index.search(np.asarray(vector, dtype="float32")[None, :], k)
        pairs = zip(scores[0], ids[0], strict=True)
        return [{**self.passages[i], "score": round(float(s), 4)} for s, i in pairs if i >= 0]
