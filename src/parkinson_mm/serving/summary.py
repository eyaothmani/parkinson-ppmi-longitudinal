"""GET /summary : les résultats AGRÉGÉS déjà calculés par le pipeline (reports/metrics), pour la page d'accueil.

Liste blanche de fichiers : aucun ne contient de donnée patient (comptes, erreurs moyennes, coefficients). Rien n'est
recalculé ici ; un fichier absent est simplement omis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FILES = ("ingest", "split", "state", "etat_cv", "cv", "test", "reference_etat", "sensibilite", "horizons_test",
         "reference_horizons", "stats", "stats_sensibilite", "robustesse", "rag", "rag_llm",
         "rag_llm_questions_controle")


def load_summary(root: Path | None = None) -> dict:
    folder = Path(root or os.environ.get("PROJECT_ROOT", ".")) / "reports" / "metrics"
    found = {name: folder / f"{name}.json" for name in FILES}
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in found.items() if path.exists()}
