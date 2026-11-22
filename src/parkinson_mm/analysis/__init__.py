"""Analyses complémentaires (validation croisée seulement pour la prédiction ; le test n'est pas relu).

Dossier à part exprès : aucune étape d'entraînement ni de test ne dépend de lui dans dvc.yaml. Le modifier ne
relance donc ni l'entraînement, ni l'évaluation sur le test.
"""
from .horizon_stage import reference_horizons
from .reference_stage import reference_state
from .sensitivity_stage import sensitivity
from .state_cv_stage import state_cv
from .stats_sensitivity_stage import stats_sensitivity

__all__ = ["reference_horizons", "reference_state", "sensitivity", "state_cv", "stats_sensitivity"]
