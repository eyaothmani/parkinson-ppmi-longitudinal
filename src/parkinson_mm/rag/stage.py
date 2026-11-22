"""Étape `rag_index` de l'assistant documentaire : construire l'index (l'évaluation est dans eval_stage.py).
Séparée du pipeline PPMI : elle ne lit que les sources du manifeste, jamais data/ ni mlflow.db."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..utils.log import get_logger
from .chunks import make_chunks
from .corpus import load_pages
from .embed import E5Embedder
from .index import build_index


def embedder_for(cfg):
    r = cfg.rag
    return E5Embedder(r.embedding_model, r.threads, r.batch_size)


def corpus_version(root: Path) -> str:
    """md5 enregistré par `dvc add rag/corpus` (12 caractères) : la version des articles indexés."""
    dvc_file = Path(root) / "rag" / "corpus.dvc"
    md5 = (yaml.safe_load(dvc_file.read_text()) or {}).get("outs", [{}])[0].get("md5", "") if dvc_file.exists() else ""
    return md5[:12] or "non-suivi"


def build_rag_index(cfg, embedder=None) -> dict:
    r = cfg.rag
    chunks = make_chunks(load_pages(cfg.root, r.manifest), r.chunk_size, r.chunk_overlap)
    info = build_index(chunks, embedder or embedder_for(cfg), cfg.root / r.index_dir,
                       {"corpus_version": corpus_version(cfg.root), "chunk_size": r.chunk_size,
                        "chunk_overlap": r.chunk_overlap})
    get_logger("parkinson_mm.rag_index").info("index : %d passages, %d documents", info["n_passages"],
                                              info["n_documents"])
    return info
