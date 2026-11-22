"""Enregistrement dans le Model Registry MLflow : pyfunc + signature + exemple d'entrée."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

from ..serving.pyfunc import NextVisitPyfunc, example_input


def register(predictor, registered_name: str, source_dir: Path) -> str:
    """Logue le modèle dans le run actif, l'enregistre et renvoie son numéro de version."""
    example = example_input()
    signature = infer_signature(example, predictor.predict_frame(example))
    with tempfile.TemporaryDirectory() as folder:
        artifact = Path(folder) / "predictor.pkl"
        predictor.save(artifact)
        info = mlflow.pyfunc.log_model(name="model", python_model=NextVisitPyfunc(),
                                       artifacts={"predictor": str(artifact)}, code_paths=[str(source_dir)],
                                       signature=signature, input_example=example,
                                       registered_model_name=registered_name)
    return str(info.registered_model_version)
