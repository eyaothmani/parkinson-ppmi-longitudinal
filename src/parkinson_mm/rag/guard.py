"""Garde-fou : l'assistant est un outil de recherche documentaire, pas un outil médical.

Une question qui demande un avis pour une personne (diagnostic, traitement, dose) est refusée AVANT toute recherche.
Règle simple et vérifiable : un mot médical ET une tournure personnelle. « Le traitement modifie-t-il le score
dans PPMI ? » reste permis ; « Quel traitement pour mon père ? » est refusé. La consigne du LLM est un 2e garde-fou."""
from __future__ import annotations

import re

MEDICAL = re.compile(r"traitement|médicament|medicament|\bdose|posologie|diagnosti|prescri|l[ée]vodopa|trembl|"
                     r"symptôme|soigner|guérir|treatment|medication|diagnos|prescription", re.I)
PERSONAL = re.compile(r"(?<!\w)(je|mon|ma|mes|ai-je|dois-je|puis-je|devrais-je|my|i|should i)(?!\w)|(?<!\w)j['’]", re.I)
REFUSAL = ("Je ne peux pas répondre à une question médicale personnelle (diagnostic, traitement, dose) : "
           "cet assistant résume des articles de recherche, il ne remplace pas un médecin. "
           "Parlez-en à un neurologue ou à votre médecin traitant.")


def is_personal_medical(question: str) -> bool:
    return bool(MEDICAL.search(question)) and bool(PERSONAL.search(question))
