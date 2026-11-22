"""Réponse sourcée : garde-fou → recherche des k passages → LLM (via API) → vérification des citations.

Sans clé d'API, ou si le LLM cite un extrait qui n'existe pas, l'assistant rend les passages eux-mêmes (mode
extractif) : il ne donne jamais de réponse sans source. Statuts : refus, introuvable, generee,
citations_invalides (réponse du LLM rejetée), extractif (pas de LLM)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .guard import REFUSAL, is_personal_medical
from .prompt import NOT_FOUND, PROMPT


@dataclass
class Answer:
    status: str
    text: str
    sources: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)


def context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['citation']}, {h['where']}) {h['text']}" for i, h in enumerate(hits, 1))


def numbered(hits: list[dict], numbers) -> list[dict]:
    return [{**hits[n - 1], "n": n} for n in sorted(numbers)]


def extractive(hits: list[dict], n: int = 3, note: str = "Passages les plus proches (à lire) :") -> str:
    # le texte du passage seulement, sur une ligne : les sources sont listées à part
    quotes = (f"[{i}] « {' '.join(h['text'][:300].split())}… »" for i, h in enumerate(hits[:n], 1))
    return "\n".join([note, *quotes])


def ask(question: str, index, embedder, k: int, llm=None) -> Answer:
    if is_personal_medical(question):
        return Answer("refus", REFUSAL)
    hits = index.search(embedder.query(question), k)
    top = numbered(hits, range(1, min(3, len(hits)) + 1))
    if llm is None:
        return Answer("extractif", extractive(hits), top, hits)
    text = llm.invoke(PROMPT.format_messages(context=context(hits), question=question)).content.strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    if not cited and NOT_FOUND.lower().rstrip(".") in text.lower():
        return Answer("introuvable", NOT_FOUND, [], hits)
    if cited and cited <= set(range(1, len(hits) + 1)):
        return Answer("generee", text, numbered(hits, cited), hits)
    note = "Réponse du LLM rejetée (citation absente ou inexistante). Passages les plus proches :"
    return Answer("citations_invalides", extractive(hits, note=note), top, hits)
