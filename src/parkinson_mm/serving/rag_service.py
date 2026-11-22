"""Assistant documentaire côté API : chargé à la PREMIÈRE question seulement (le modèle d'embeddings pèse ≈ 470 Mo),
puis gardé en mémoire. Si l'API du LLM échoue (réseau, limite de débit), la réponse retombe en mode extractif."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("parkinson_mm.api")
_lock = threading.Lock()


def assistant(state: dict) -> dict:
    with _lock:
        if state.get("rag") is None:
            from ..rag.index import Index
            from ..rag.llm import make_llm
            from ..rag.stage import embedder_for
            from ..utils.config import load_config

            cfg = load_config(Path(os.environ.get("PROJECT_ROOT", ".")) / "params.yaml")
            embedder = embedder_for(cfg)
            state["rag"] = {"index": Index(cfg.root / cfg.rag.index_dir, embedder.name), "embedder": embedder,
                            "llm": make_llm(cfg.rag.llm, cfg.root), "k": cfg.rag.k}
        return state["rag"]


def answer(question: str, rag: dict):
    from ..rag import ask

    try:
        return ask(question, rag["index"], rag["embedder"], rag["k"], rag["llm"]), rag["llm"] is not None
    except Exception as exc:                       # le LLM est un service externe : on ne casse pas la page
        log.warning("LLM indisponible (%s) : réponse extractive", type(exc).__name__)
        return ask(question, rag["index"], rag["embedder"], rag["k"], None), False
