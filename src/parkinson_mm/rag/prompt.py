"""Consigne donnée au LLM. Versionnée : chaque run d'évaluation enregistre `PROMPT_VERSION` dans MLflow."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND = "Je ne trouve pas cela dans le corpus."
# v2 (21/11/2026), après relecture de la v1 : la v1 concluait au-delà d'un extrait (GRU-D « meilleur » que le MLP alors
# que l'extrait dit l'inverse) et répondait « introuvable » alors que le 1er extrait contenait la réponse.
PROMPT_VERSION = "v2"
SYSTEM = ("Tu es un assistant documentaire de recherche sur la maladie de Parkinson. Réponds en français, en 2 à 6 "
          "phrases, uniquement à partir des extraits numérotés. Termine chaque phrase par le numéro de l'extrait "
          "utilisé, par exemple [2]. Reprends les conclusions des extraits telles qu'elles sont écrites : "
          "si un extrait dit qu'une différence n'est pas significative ou qu'une hypothèse n'est pas confirmée, "
          "dis-le, sans conclure au-delà. Donne les chiffres avec leur intervalle quand l'extrait le donne. "
          "Lis tous les extraits : si l'un d'eux répond, même en partie, réponds. "
          f"Si aucun ne permet de répondre, écris seulement : {NOT_FOUND} "
          "Ne pose aucun diagnostic et ne recommande aucun traitement.")
HUMAN = "Extraits :\n{context}\n\nQuestion : {question}"
PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
