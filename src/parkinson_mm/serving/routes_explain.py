"""POST /explain (objectif 2) : pourquoi le modèle prédit CE score pour CE dossier (ADR-0016)."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..explainability.local import METHOD, explain_local
from .predictor import DISCLAIMER
from .routes import request_frame
from .schemas import PredictRequest
from .schemas_explain import ExplainResponse, GroupEffect
from .security import require_key
from .state import state

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest, _: str = Depends(require_key)) -> ExplainResponse:
    """Même entrée que /predict. Pour chaque groupe de variables : de combien la prédiction changerait si ce groupe
    prenait la valeur typique des patients d'entraînement. Explication du modèle, pas des causes."""
    predictor = state["predictor"]
    if predictor is None or not getattr(predictor, "reference", None):
        raise HTTPException(503, "Aucun modèle avec valeurs de référence chargé (exécuter scripts/evaluate.py).")
    samples, X, mask = predictor.prepare(request_frame(request))
    prediction = float(np.clip(predictor.model.predict(samples, X, mask)[0], 0, 132))
    effects = explain_local(predictor.model, samples, X, mask, predictor.reference)
    current = float(samples["np3"].iloc[0])
    return ExplainResponse(
        current_score=current, prediction=round(prediction, 2), predicted_change=round(prediction - current, 2),
        effects=[GroupEffect(group=row.groupe, effect_points=round(row.effet, 2)) for row in effects.itertuples()],
        method=METHOD, model=str(predictor.metadata.get("model", "?")), model_version=state["info"]["version"],
        disclaimer=DISCLAIMER)
