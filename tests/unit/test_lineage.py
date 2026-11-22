"""data_version ne doit dépendre que des fichiers réellement lus, et changer si leur contenu change.
« +modifs » ne signale que du code non commité, pas les résultats que les étapes écrivent elles-mêmes."""
import subprocess

from parkinson_mm.utils.lineage import data_manifest, git_commit


def test_unused_file_does_not_change_version(tmp_path):
    used = tmp_path / "Motor" / "MDS-UPDRS_Part_III_01Jan2026.csv"
    used.parent.mkdir()
    used.write_text("PATNO,NP3TOT\n1,20\n")
    before = data_manifest(tmp_path, [used])["data_version"]
    (tmp_path / "Autre_fichier_01Jan2026.csv").write_text("x\n1\n")      # téléchargé mais non utilisé
    assert data_manifest(tmp_path, [used])["data_version"] == before
    assert data_manifest(tmp_path)["n_files"] == 2                          # sans liste : tous les CSV


def test_content_change_changes_version(tmp_path):
    used = tmp_path / "table.csv"
    used.write_text("a\n1\n")
    before = data_manifest(tmp_path, [used])["data_version"]
    used.write_text("a\n2\n")
    assert data_manifest(tmp_path, [used])["data_version"] != before


def test_results_written_by_the_stages_are_not_uncommitted_code(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.org")
    git("config", "user.name", "test")
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    for name in ("code.py", "dvc.lock", "reports/metrics/stats.json"):
        (tmp_path / name).write_text("0\n")
    git("add", ".")
    git("commit", "-qm", "init")
    (tmp_path / "dvc.lock").write_text("1\n")
    (tmp_path / "reports" / "metrics" / "stats.json").unlink()     # DVC efface la sortie avant de relancer
    assert not git_commit(tmp_path).endswith("+modifs")
    (tmp_path / "code.py").write_text("1\n")
    assert git_commit(tmp_path).endswith("+modifs")
