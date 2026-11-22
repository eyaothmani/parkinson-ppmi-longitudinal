"""Calcul du score d'état pour l'API : transforme la requête en historique, puis niveau, vitesse et projection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.state import REFERENCE, StateModel
from .schemas import History
from .schemas_state import ProjectionPoint

HORIZONS_MONTHS = (0, 6, 12, 18, 24)
NAN = float("nan")


def history_frame(request: History) -> pd.DataFrame:
    """Mêmes recodages que l'entraînement : âge centré sur 62 ans, durée de maladie centrée sur 1,5 an."""
    p = request.patient
    age_c = NAN if p.age_bl is None else p.age_bl - 62
    dur_c = NAN if p.disease_dur_bl is None else p.disease_dur_bl - 1.5
    return pd.DataFrame([{"time_years": v.time_years, "np3": v.np3, "med": v.med_state, "age_c": age_c,
                          "male": NAN if p.male is None else float(p.male), "dur_c": dur_c} for v in request.visits])


def score(model: StateModel, request: History) -> dict:
    history = history_frame(request)
    state = model.state(history)
    last = history.tail(1)
    future = pd.concat([last.assign(time_years=state["time_years"] + m / 12, med=REFERENCE)
                        for m in HORIZONS_MONTHS])
    mean, sd = model.predict(history, future, noise=False)
    clip = lambda x: float(np.clip(x, 0, 132))  # noqa: E731 - borne clinique du score
    projection = [ProjectionPoint(months=m, level=clip(mu), lower=clip(mu - 1.96 * s), upper=clip(mu + 1.96 * s))
                  for m, mu, s in zip(HORIZONS_MONTHS, mean, sd, strict=True)]
    half = lambda value, sd_: (value - 1.96 * sd_, value + 1.96 * sd_)  # noqa: E731
    return {"measured_last": state["np3_mesure"], "current_level": clip(state["niveau_actuel"]),
            "current_level_ci95": tuple(clip(x) for x in half(state["niveau_actuel"], state["niveau_sd"])),
            "speed_pts_per_year": state["vitesse"], "speed_ci95": half(state["vitesse"], state["vitesse_sd"]),
            "n_visits": len(history), "projection": projection}
