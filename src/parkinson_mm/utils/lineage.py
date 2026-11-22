"""Traçabilité : avec quelles données, quel code et quel environnement un résultat a été produit."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "pandas", "scikit-learn", "statsmodels", "lightgbm", "torch", "mlflow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest(raw_dir: Path, used: list[Path] | None = None) -> dict:
    """Empreinte des CSV réellement lus (`used`) ; `data_version` les résume en 12 caractères.
    Ajouter un fichier inutilisé dans data/raw ne change donc pas la version (sans `used` : tous les CSV)."""
    paths = sorted(used) if used is not None else [p for p in sorted(Path(raw_dir).rglob("*.csv"))
                                                   if "_to_delete" not in p.parts]
    files = [{"path": Path(p).relative_to(raw_dir).as_posix(), "bytes": Path(p).stat().st_size,
              "sha256": sha256(Path(p))} for p in paths]
    version = hashlib.sha256("".join(f["path"] + f["sha256"] for f in files).encode()).hexdigest()[:12]
    return {"data_version": version, "n_files": len(files), "files": files}


def git_commit(root: Path, results=("reports/metrics/", "dvc.lock")) -> str:
    try:   # « +modifs » : du CODE suivi a changé ; les résultats écrits par les étapes (results) ne comptent pas
        out = [subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout
               for a in (["rev-parse", "--short", "HEAD"], ["status", "--porcelain", "--untracked-files=no"])]
    except OSError:
        return "git-absent"
    code = [line for line in out[1].splitlines() if not line[3:].startswith(results)]
    return (out[0].strip() + ("+modifs" if code else "")) or "non-versionne"


def dvc_rev(root: Path) -> str:
    """md5 enregistré par `dvc add data/raw` ; 'non-suivi' tant que DVC n'est pas initialisé."""
    dvc_file = Path(root) / "data" / "raw.dvc"
    if not dvc_file.exists():
        return "non-suivi"
    import yaml
    return str((yaml.safe_load(dvc_file.read_text()) or {}).get("outs", [{}])[0].get("md5", "inconnu"))


def versions() -> dict:
    installed = {d.metadata["Name"].lower(): d.version for d in metadata.distributions()}
    return {"python": platform.python_version(), **{p: installed[p] for p in PACKAGES if p in installed}}


def lineage(cfg, data_version: str | None) -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(cfg.root), "dvc_rev": dvc_rev(cfg.root), "data_version": data_version,
            "config_fingerprint": cfg.fingerprint, "versions": versions()}
