"""Réponses des objectifs 1 (score d'état actuel) et 3 (facteurs associés à l'évolution)."""
from __future__ import annotations

from pydantic import BaseModel

from .schemas import History


class StateRequest(History):
    """Même entrée que /predict, sans horizon : le score d'état décrit la situation à la dernière visite."""


class ProjectionPoint(BaseModel):
    months: float
    level: float
    lower: float
    upper: float


class StateResponse(BaseModel):
    measured_last: float
    current_level: float
    current_level_ci95: tuple[float, float]
    speed_pts_per_year: float
    speed_ci95: tuple[float, float]
    n_visits: int
    projection: list[ProjectionPoint]
    model_version: str
    disclaimer: str


class FactorsResponse(BaseModel):
    mean_slope_pts_per_year: float
    sd_individual_slopes: float
    factors: list[dict]
    disclaimer: str
