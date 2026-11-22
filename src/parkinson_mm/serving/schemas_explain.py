"""Réponse de POST /explain : pourquoi le modèle a donné ce score (explication locale par groupe, ADR-0016)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GroupEffect(BaseModel):
    group: str
    effect_points: float = Field(..., description="Prédiction réelle − prédiction si ce groupe avait une valeur "
                                                  "typique des patients d'entraînement (points de MDS-UPDRS III)")


class ExplainResponse(BaseModel):
    current_score: float = Field(..., description="MDS-UPDRS III de la dernière visite : point de départ du modèle")
    prediction: float
    predicted_change: float = Field(..., description="prédiction − score actuel")
    effects: list[GroupEffect] = Field(..., description="du plus fort au plus faible ; non additifs")
    method: str
    model: str
    model_version: str
    disclaimer: str
