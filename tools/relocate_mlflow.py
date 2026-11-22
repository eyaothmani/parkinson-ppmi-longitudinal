"""Rebase les chemins d'artefacts MLflow quand le projet change de dossier (ex. cloud -> D:\\parkinson-projet).

MLflow enregistre des chemins ABSOLUS (mlflow.db + fichiers MLmodel). Après une copie du projet, les
métriques restent lisibles mais les artefacts et le modèle `champion` deviennent introuvables.
Usage, depuis la nouvelle racine :  python tools/relocate_mlflow.py --old file:///D:/parkinson-projet
Une copie de sauvegarde mlflow.db.bak est créée avant toute modification.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

# (table, colonne) qui contiennent un chemin d'artefact dans le schéma MLflow 3
COLUMNS = [("experiments", "artifact_location"), ("runs", "artifact_uri"),
           ("logged_models", "artifact_location"), ("model_versions", "storage_location"),
           ("model_versions", "source")]


def rebase_database(db: Path, old: str, new: str) -> int:
    shutil.copy2(db, db.with_suffix(".db.bak"))
    changed = 0
    with sqlite3.connect(db) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in COLUMNS:
            if table in tables:
                cur = con.execute(f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?) "
                                  f"WHERE {column} LIKE ?", (old, new, f"%{old}%"))
                changed += cur.rowcount
    return changed


def rebase_mlmodel_files(mlruns: Path, old: str, new: str) -> int:
    changed = 0
    for path in mlruns.rglob("MLmodel"):
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", required=True, help="ancienne racine du projet (voir `mlflow ui`)")
    parser.add_argument("--root", default=".", help="nouvelle racine (défaut : dossier courant)")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    old, new = args.old.rstrip("/\\"), root.as_uri()          # URI file:/// valide sous Windows et Linux
    n_db = rebase_database(root / "mlflow.db", old, new)
    n_files = rebase_mlmodel_files(root / "mlruns", old, new)
    print(f"{n_db} chemins mis à jour dans mlflow.db, {n_files} fichiers MLmodel ; nouvelle racine : {new}")


if __name__ == "__main__":
    main()
