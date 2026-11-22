"""Objectif 1 (/state) et objectif 3 (/factors). Résultats agrégés : aucune donnée patient n'est stockée."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from .predictor import DISCLAIMER
from .schemas_state import FactorsResponse, StateRequest, StateResponse
from .security import require_key
from .state import state
from .state_scoring import score

router = APIRouter()


@router.post("/state", response_model=StateResponse)
def current_state(request: StateRequest, _: str = Depends(require_key)) -> StateResponse:
    """Score d'état actuel : niveau (sans traitement) et vitesse d'évolution estimés sur les visites passées."""
    if state["state_model"] is None:
        raise HTTPException(503, "Modèle de score d'état non chargé (exécuter `python scripts/run_state.py`).")
    started = time.time()
    result = score(state["state_model"], request)
    state["metrics"].record(time.time() - started, False, request.visits[-1].model_dump())
    return StateResponse(**result, model_version=state["info"]["version"], disclaimer=DISCLAIMER)


@router.get("/factors", response_model=FactorsResponse)
def factors(_: str = Depends(require_key)) -> FactorsResponse:
    """Facteurs associés à une évolution plus rapide ou plus lente (modèles mixtes, correction FDR)."""
    if state["factors"] is None:
        raise HTTPException(503, "Résultats statistiques absents (exécuter `python scripts/run_stats.py`).")
    return FactorsResponse(**state["factors"], disclaimer=DISCLAIMER)
