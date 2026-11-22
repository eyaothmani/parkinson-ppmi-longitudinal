"""Le rebasage des chemins MLflow doit toucher les chemins d'artefacts, et rien d'autre."""
import importlib.util
import sqlite3
from pathlib import Path

TOOL = Path(__file__).parents[2] / "tools" / "relocate_mlflow.py"
spec = importlib.util.spec_from_file_location("relocate_mlflow", TOOL)
relocate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relocate)


def test_rebase_database_and_mlmodel(tmp_path):
    db = tmp_path / "mlflow.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
        con.executemany("INSERT INTO runs VALUES (?, ?)",
                        [("a", "/old/root/mlruns/1/a/artifacts"), ("b", "s3://ailleurs/b")])
    model = tmp_path / "mlruns" / "5" / "MLmodel"
    model.parent.mkdir(parents=True)
    model.write_text("artifact_path: /old/root/mlruns/5\nflavors: {}\n")

    assert relocate.rebase_database(db, "/old/root", "file:///new") == 1
    assert relocate.rebase_mlmodel_files(tmp_path / "mlruns", "/old/root", "file:///new") == 1
    with sqlite3.connect(db) as con:
        uris = dict(con.execute("SELECT run_uuid, artifact_uri FROM runs"))
    assert uris == {"a": "file:///new/mlruns/1/a/artifacts", "b": "s3://ailleurs/b"}
    assert (tmp_path / "mlflow.db.bak").exists()
