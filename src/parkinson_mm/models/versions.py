"""Versions du registry : retrouver une version déjà évaluée, lui donner son alias, relire sa décision (ADR-0017)."""
from __future__ import annotations

from mlflow import MlflowClient

# un alias par modèle ; `grud` : modèle multi-horizon du sprint 5 (ADR-0018)
ALIASES = {"persistence": "persistance", "lgbm": "lightgbm", "mlp": "mlp", "gru": "gru", "grud": "grud"}


def find_evaluated(experiment: str, registered_model: str, run_name: str, data_version, fingerprint: str):
    """Version déjà évaluée avec les mêmes données et la même configuration : on ne la ré-enregistre pas."""
    client = MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return None
    runs = client.search_runs([found.experiment_id], f"attributes.run_name = '{run_name}' and "
                              f"tags.data_version = '{data_version}' and tags.config_fingerprint = '{fingerprint}'")
    ids = {run.info.run_id for run in runs}
    return next((v for v in client.search_model_versions(f"name='{registered_model}'") if v.run_id in ids), None)


def label(registered_model: str, version, algo: str, tags: dict | None = None) -> None:
    """Alias propre au modèle (persistance, lightgbm, mlp, gru) et tag `modele` : chaque modèle se retrouve vite."""
    client, version = MlflowClient(), str(version)
    client.set_registered_model_alias(registered_model, ALIASES[algo], version)
    for key, value in {"modele": algo, **(tags or {})}.items():
        client.set_model_version_tag(registered_model, version, key, value)


def decision(registered_model: str, version) -> dict:
    """Décision de promotion déjà écrite en tags sur une version (même forme que `promote`)."""
    found = MlflowClient().get_model_version(registered_model, str(version))
    reasons = found.tags.get("promotion_reasons", "")
    return {"version": str(version), "champion": "champion" in found.aliases,
            "reasons": [] if reasons.startswith("tous les critères") else reasons.split(" | ")}
