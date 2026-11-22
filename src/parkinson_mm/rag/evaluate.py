"""Évaluation sur des questions écrites AVANT toute mesure (rag/eval/questions.yaml), sans réglage ensuite.

Types : `corpus` (un document attendu), `hors_corpus` (réponse attendue : introuvable), `medical` (refus attendu).
Recherche : hit@k (le document attendu est-il parmi les k passages ?) et MRR (moyenne de 1 / rang du 1er bon
passage). Avec le LLM : part de réponses sourcées, réponses rejetées, « introuvable » hors corpus."""
from __future__ import annotations

import time

import pandas as pd

from .answer import ask


def first_rank(hits: list[dict], doc_id: str | None) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if h["doc_id"] == doc_id), None)


def run_questions(questions: list[dict], index, embedder, k: int, llm=None, pause_s: float = 0.0) -> pd.DataFrame:
    rows = []
    for q in questions:
        answer = ask(q["question"], index, embedder, k, llm)
        rank = first_rank(answer.hits, q.get("document")) if q["type"] == "corpus" else None
        rows.append({"id": q["id"], "type": q["type"], "document": q.get("document", ""), "statut": answer.status,
                     "rang": rank, "sources": "; ".join(f"{s['citation']}, {s['where']}" for s in answer.sources),
                     "reponse": answer.text})
        if llm is not None and answer.status != "refus":
            time.sleep(pause_s)              # limite de débit de l'offre gratuite
    return pd.DataFrame(rows)


def share(mask: pd.Series) -> float:
    return round(float(mask.mean()), 4) if len(mask) else float("nan")


def metrics(details: pd.DataFrame, k: int, with_llm: bool) -> dict:
    corpus, outside, medical = (details[details["type"] == t] for t in ("corpus", "hors_corpus", "medical"))
    rank = pd.to_numeric(corpus["rang"], errors="coerce")
    result = {"n_questions": len(details), "n_corpus": len(corpus), f"hit_at_{k}": share(rank.notna()),
              "mrr": round(float((1 / rank).fillna(0).mean()), 4) if len(rank) else float("nan"),
              "refus_corrects": share(medical["statut"] == "refus"),
              "faux_refus": share(details.loc[details["type"] != "medical", "statut"] == "refus")}
    if with_llm:
        result |= {"reponses_sourcees": share(corpus["statut"] == "generee"),
                   "reponses_rejetees": int((details["statut"] == "citations_invalides").sum()),
                   "introuvable_hors_corpus": share(outside["statut"] == "introuvable"),
                   "introuvable_sur_corpus": share(corpus["statut"] == "introuvable")}
    return result
