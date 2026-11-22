"""Étapes du pipeline reproductible (appelées par scripts/ et déclarées dans dvc.yaml).

    ingest -> validate -> split -> features -> train -> evaluate -> explain   (objectif 2 : score futur)
                                     ├-> stats (objectif 3 : facteurs associés)
                                     ├-> state (objectif 1 : score actuel)
                                     └-> horizons -> train_horizons -> evaluate_horizons   (sprint 5 : 12/24/36 mois)
    train -> robustesse   (sprint 6 : erreurs par sous-groupe, intervalle conforme ; CV seulement)
Chaque étape lit params.yaml et les fichiers de l'étape précédente : aucun état caché en mémoire.
"""
from .data_stages import ingest, validate
from .evaluate_stage import evaluate
from .explain_stage import explain
from .horizon_data import horizons
from .horizon_evaluate import evaluate_horizons
from .horizon_train import train_horizons
from .model_stages import train
from .robustness_stage import robustness
from .split_stages import features, split
from .state_stage import state
from .stats_stage import stats

STAGES = {"ingest": ingest, "validate": validate, "split": split, "features": features, "stats": stats,
          "state": state, "train": train, "evaluate": evaluate, "explain": explain, "horizons": horizons,
          "train_horizons": train_horizons, "evaluate_horizons": evaluate_horizons, "robustesse": robustness}
