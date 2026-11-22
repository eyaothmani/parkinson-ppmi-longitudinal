"""Parkinson multimodal — analyse longitudinale de la cohorte PPMI.

Les couches ne dépendent que des couches situées en dessous :

    pipeline        orchestration des étapes (scripts/, dvc.yaml)
    analysis        analyses complémentaires de l'objectif 2 (CV seulement), hors entraînement
    serving         prédicteur empaqueté (modèle + prétraitement), wrapper MLflow
    training        validation croisée, entraînement, suivi MLflow
    models          baselines, réseaux, registry
    evaluation      métriques, intervalles bootstrap
    explainability  importance par permutation
    stats           cœur scientifique : modèles mixtes, FDR
    features        échantillons, séquences, prétraitement
    data            lecture PPMI, nettoyage, contrats, split
    utils           configuration, journalisation, traçabilité

À part : rag, l'assistant documentaire (extension), qui n'utilise que utils et le suivi MLflow.
"""

__version__ = "1.5.0"
