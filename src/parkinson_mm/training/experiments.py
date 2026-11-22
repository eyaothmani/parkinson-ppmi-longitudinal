"""Organisation des experiments MLflow (ADR-0022) : un experiment = des runs qu'on compare entre eux.
Les runs TEST d'une question vivent à part (`<question>_test`) ; la description et les tags de chaque experiment
viennent du catalogue `docs/mlflow_experiments.yaml`, la seule source de ces textes."""
from __future__ import annotations

from pathlib import Path

import yaml

CATALOG = "docs/mlflow_experiments.yaml"
NOTE = "mlflow.note.content"            # tag affiché par l'interface en tête de l'experiment


def question_test(question: str) -> str:
    """Experiment des runs TEST d'une question : jamais mélangés à ses runs de validation croisée."""
    return f"{question}_test"


def catalog(root: Path) -> dict:
    path = Path(root) / CATALOG
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiments"] if path.exists() else {}


def experiment_tags(root: Path, question: str) -> dict:
    """Description + tags de filtrage d'une question ; {} si elle n'est pas au catalogue."""
    entry = catalog(root).get(question)
    if not entry:
        return {}
    return {NOTE: entry["description"].strip(), **{k: str(v) for k, v in entry.get("tags", {}).items()}}
