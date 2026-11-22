"""Étape `rag_eval` : l'assistant face à un jeu de questions écrites avant la mesure, tracé dans MLflow.

Jeu par défaut : `rag.questions` (métriques dans reports/metrics/rag.json, ou rag_llm.json avec le LLM). Un autre jeu
(`--questions`, par exemple le jeu de contrôle) écrit rag[_llm]_<nom du jeu>.json : rien n'est écrasé."""
from __future__ import annotations

from pathlib import Path

import mlflow
import yaml

from ..training.tracking import log_artifact, model_run, numeric
from ..utils.io import write_json
from .evaluate import metrics, run_questions
from .index import Index
from .llm import make_llm
from .prompt import PROMPT_VERSION
from .stage import embedder_for


def eval_rag(cfg, use_llm: bool = False, embedder=None, llm=None, questions: str | None = None) -> dict:
    r, embedder = cfg.rag, embedder or embedder_for(cfg)
    llm = llm if llm is not None else (make_llm(r.llm, cfg.root) if use_llm else None)
    if use_llm and llm is None:
        raise SystemExit(f"--llm demandé, mais {r.llm.key_env} est absente (variable ou fichier .env).")
    path = questions or r.questions
    stem = Path(path).stem
    suffix = "" if path == r.questions else f"_{stem}"
    index = Index(cfg.root / r.index_dir, embedder.name)
    asked = yaml.safe_load((cfg.root / path).read_text(encoding="utf-8"))["questions"]
    details = run_questions(asked, index, embedder, r.k, llm, r.pause_s)
    result = metrics(details, r.k, llm is not None)
    write_json(result, cfg.path("metrics") / f"rag{'_llm' if llm is not None else ''}{suffix}.json")
    name = (f"Recherche + LLM ({r.llm.model}, consigne {PROMPT_VERSION})" if llm is not None else "Recherche seule")
    used = llm is not None
    params = {**{key: index.info[key] for key in ("embedding_model", "chunk_size", "chunk_overlap", "n_passages")},
              "k": r.k, "questions": stem, "n_questions": len(details), "llm": r.llm.model if used else "aucun",
              "consigne": PROMPT_VERSION if used else "aucune",
              "reasoning_effort": r.llm.get("reasoning_effort") if used else "aucun"}
    with model_run(cfg, r.question, f"{name}, {stem}" if suffix else name, params, {"stage": "rag_eval"},
                   index.info["corpus_version"]):
        mlflow.log_metrics(numeric(result))
        log_artifact(details, "questions.csv")
    return result
