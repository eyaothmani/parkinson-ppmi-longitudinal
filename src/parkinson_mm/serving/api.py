"""Application FastAPI de démonstration (outil de recherche, non diagnostique).

    uvicorn parkinson_mm.serving.api:app --port 8000   ->  tableau de bord : /app/  ·  Swagger : /docs
Les journaux ne contiennent jamais de données patient : identifiant de requête, latence, statut, version.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .loader import load_factors, load_served_model, load_state_model
from .predictor import DISCLAIMER
from .routes import router
from .routes_explain import router as explain_router
from .routes_more import router as more_router
from .routes_state import router as state_router
from .state import state
from .web import mount_web

log = logging.getLogger("parkinson_mm.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    loaders = {"champion": lambda: dict(zip(("predictor", "info"), load_served_model(), strict=True)),
               "state_model": lambda: {"state_model": load_state_model()},
               "factors": lambda: {"factors": load_factors()}}
    for name, loader in loaders.items():
        try:                   # l'API démarre quand même : /health signale ce qui manque
            state.update(loader())
        except Exception as exc:
            log.error("%s non chargé : %s", name, type(exc).__name__)
    yield


app = FastAPI(title="Parkinson multimodal — démonstrateur de recherche", description=DISCLAIMER, lifespan=lifespan)
app.include_router(router)
app.include_router(explain_router)
app.include_router(state_router)
app.include_router(more_router)
mount_web(app)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id, started = uuid.uuid4().hex[:8], time.time()
    response = await call_next(request)
    log.info("request_id=%s path=%s status=%s latency_ms=%.0f model=%s", request_id, request.url.path,
             response.status_code, 1000 * (time.time() - started), state["info"]["version"])
    response.headers["X-Request-ID"] = request_id
    return response
