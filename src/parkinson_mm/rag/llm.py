"""LLM via une API compatible OpenAI (par défaut Groq, offre gratuite). Changer de fournisseur (Mistral, Gemini…) =
changer `base_url`, `key_env` et `model` dans params.yaml. La clé vient de la variable nommée par `key_env` ou du
fichier .env (jamais versionné). Ce qui part vers l'API : la question et des extraits d'articles publiés ou du
README. Jamais de donnée PPMI."""
from __future__ import annotations

import os
from pathlib import Path


def make_llm(settings, root: Path | None = None):
    """Modèle de chat LangChain, ou None sans clé : l'assistant passe alors en mode extractif."""
    from dotenv import load_dotenv

    load_dotenv(Path(root or ".") / ".env", override=False)
    key = os.environ.get(settings["key_env"])
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    effort = {"reasoning_effort": settings["reasoning_effort"]} if settings.get("reasoning_effort") else {}
    return ChatOpenAI(model=settings["model"], base_url=settings["base_url"], api_key=key,
                      temperature=settings["temperature"], max_tokens=settings["max_tokens"],
                      max_retries=6, timeout=90, **effort)   # 429 (limite de débit) : nouvelles tentatives espacées
