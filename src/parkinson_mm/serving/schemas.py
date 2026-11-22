"""Schémas stricts de l'API : toute valeur hors des bornes cliniques est rejetée (HTTP 422)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MedState = Literal["untreated", "OFF", "ON"]
Binary = Literal[0, 1]


class Visit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_years: float = Field(..., ge=0, le=40, description="Années depuis la baseline")
    np3: float = Field(..., ge=0, le=132, description="MDS-UPDRS III total")
    med_state: MedState
    nhy: float | None = Field(None, ge=0, le=5)
    updrs2: float | None = Field(None, ge=0, le=52)
    schwab: float | None = Field(None, ge=0, le=100)
    ledd: float | None = Field(None, ge=0, le=4000)
    moca: float | None = Field(None, ge=0, le=30)
    age: float | None = Field(None, ge=18, le=110)


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    male: Binary | None = None
    age_bl: float | None = Field(None, ge=18, le=110)
    lrrk2: Binary | None = None
    gba: Binary | None = None
    disease_dur_bl: float | None = Field(None, ge=0, le=40, description="Années depuis le diagnostic, à la baseline")


class History(BaseModel):
    """Historique de visites d'un patient : entrée commune aux objectifs 1 (état) et 2 (visite suivante)."""
    model_config = ConfigDict(extra="forbid")
    visits: list[Visit] = Field(..., min_length=1, max_length=60)
    patient: Patient = Patient()

    @model_validator(mode="after")
    def visits_in_order(self):
        times = [v.time_years for v in self.visits]
        if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError("Les visites doivent être strictement croissantes dans le temps.")
        return self


class PredictRequest(History):
    horizon_months: float = Field(..., ge=3, le=24, description="Délai jusqu'à la visite à prédire")
    target_state: MedState = Field(..., description="État médicamenteux prévu à cette visite")


class PredictResponse(BaseModel):
    prediction: float
    interval_80: tuple[float, float]
    model_version: str
    disclaimer: str
