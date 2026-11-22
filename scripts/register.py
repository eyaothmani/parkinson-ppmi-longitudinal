"""Affiche l'état du Model Registry : versions, alias et raisons de chaque décision de promotion."""
import mlflow
from mlflow import MlflowClient

from parkinson_mm.training.tracking import tracking_uri
from parkinson_mm.utils.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    client, name = MlflowClient(), cfg.tracking.registered_model
    for v in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        aliases = ",".join(client.get_model_version(name, v.version).aliases) or "-"
        print(f"v{v.version:<3} alias={aliases:<20} MAE test={v.tags.get('test_MAE', '?'):<8} "
              f"{v.tags.get('promotion_decision', '?')} : {v.tags.get('promotion_reasons', '')}")
