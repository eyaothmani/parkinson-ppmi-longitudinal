"""Extension : assistant documentaire (RAG), séparé du pipeline scientifique PPMI.

Il répond aux questions sur un petit corpus (articles publiés + README du projet) en citant chaque source, refuse
les questions médicales personnelles et répond « je ne trouve pas cela dans le corpus » plutôt que d'inventer.
Il ne lit aucune donnée patient. Chaîne : corpus → extract → chunks → embed → index → answer (guard, llm)."""
from .answer import NOT_FOUND, Answer, ask
from .eval_stage import eval_rag
from .stage import build_rag_index

__all__ = ["NOT_FOUND", "Answer", "ask", "build_rag_index", "eval_rag"]
