"""Promotion objective par alias (ADR-0004). Toute version reçoit l'alias « candidate » ; elle devient
« champion » seulement si les critères sont remplis. Les raisons sont écrites en tags : aucune promotion sans trace.

Critère 3 : la comparaison au champion en place se fait sur la MAE de validation croisée (à données et plis
identiques), jamais sur la MAE de test : le test ne sert à aucune décision (correction du 28/10/2026)."""
from __future__ import annotations

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def promote(name: str, version: str, test_mae: float, ci_vs_persistence: tuple, cv_mae_sd: float,
            max_cv_mae_sd: float, cv_mae: float | None = None) -> dict:
    client, reasons = MlflowClient(), []
    client.set_registered_model_alias(name, "candidate", version)
    if ci_vs_persistence[1] >= 0:
        reasons.append(f"IC95 vs persistance [{ci_vs_persistence[0]:.2f} ; {ci_vs_persistence[1]:.2f}] contient 0")
    if cv_mae_sd > max_cv_mae_sd:
        reasons.append(f"instable en validation croisée (MAE_sd {cv_mae_sd:.2f} > {max_cv_mae_sd})")
    try:
        champion = client.get_model_version_by_alias(name, "champion")
        previous = champion.tags.get("cv_MAE")   # absent des versions antérieures : pas de comparaison possible
        if cv_mae is not None and previous is not None and cv_mae > float(previous):
            reasons.append(f"moins bon que le champion v{champion.version} en validation croisée")
    except MlflowException:
        pass   # premier modèle : pas encore de champion
    promoted = not reasons
    if promoted:
        client.set_registered_model_alias(name, "champion", version)
    tags = {"test_MAE": f"{test_mae:.4f}", "cv_MAE_sd": f"{cv_mae_sd:.4f}",
            "promotion_decision": "champion" if promoted else "candidate",
            "promotion_reasons": " | ".join(reasons) or "tous les critères sont remplis"}
    if cv_mae is not None:
        tags["cv_MAE"] = f"{cv_mae:.4f}"
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    return {"version": version, "champion": promoted, "reasons": reasons}
