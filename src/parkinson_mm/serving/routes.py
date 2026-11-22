"""Points d'accès de l'API (toutes les routes sauf /health exigent la clé d'API)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from .predictor import DISCLAIMER
from .schemas import PredictRequest, PredictResponse
from .security import require_key
from .state import state

router = APIRouter()


@router.get("/health")
def health() -> dict:
    loaded = {"model_loaded": state["predictor"] is not None, "state_model_loaded": state["state_model"] is not None,
              "factors_loaded": state["factors"] is not None}
    return {"status": "ok" if all(loaded.values()) else "degraded", **loaded}


@router.get("/model-info")
def model_info(_: str = Depends(require_key)) -> dict:
    meta = state["predictor"].metadata if state["predictor"] else {}
    return {**state["info"], **{k: v for k, v in meta.items() if k != "reference_quantiles"}}


@router.get("/metrics")
def metrics(_: str = Depends(require_key)) -> dict:
    reference = state["predictor"].metadata.get("reference_quantiles") if state["predictor"] else None
    return state["metrics"].snapshot(reference)


def request_frame(request: PredictRequest) -> pd.DataFrame:
    """Requête -> une ligne par visite, avec les conditions de la visite à prédire (délai, état prévu)."""
    patient = request.patient.model_dump(exclude={"disease_dur_bl"})     # durée de maladie : réservée à /state
    return pd.DataFrame([{**v.model_dump(), **patient, "PATNO": 0, "gap_months": request.horizon_months,
                          "target_state": request.target_state} for v in request.visits])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, _: str = Depends(require_key)) -> PredictResponse:
    if state["predictor"] is None:
        raise HTTPException(503, "Aucun modèle chargé.")
    started = time.time()
    result = state["predictor"].predict_frame(request_frame(request)).iloc[0]
    ok = bool(np.isfinite(result["prediction"]))
    state["metrics"].record(time.time() - started, not ok, request.visits[-1].model_dump())
    if not ok:
        raise HTTPException(422, "Entrée non exploitable par le modèle.")
    return PredictResponse(prediction=round(float(result["prediction"]), 2),
                           interval_80=(round(float(result["lower"]), 2), round(float(result["upper"]), 2)),
                           model_version=state["info"]["version"], disclaimer=DISCLAIMER)
