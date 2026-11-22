"""Lecture de params.yaml, la SEULE source des paramètres : un dict lisible par attributs (cfg.training.lr).
Une section manquante ou inconnue lève une erreur tout de suite, plutôt qu'un résultat faux plus tard."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

PARAMS_FILE = "params.yaml"
SECTIONS = {"project", "paths", "cohort", "cleaning", "samples", "split", "stats", "evaluation", "tracking",
            "training", "fast_mode", "experiments", "explain", "promotion", "horizons", "robustesse",
            "reference_etat", "sensibilite", "rag"}


class Section(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(f"Paramètre absent de params.yaml : {key}")
        value = self[key]
        return Section(value) if isinstance(value, dict) else value


class Config(Section):
    def __init__(self, raw: dict, root: Path):
        missing, unknown = SECTIONS - set(raw), set(raw) - SECTIONS
        if missing or unknown:
            raise ValueError(f"params.yaml invalide — manquantes : {sorted(missing)}, inconnues : {sorted(unknown)}")
        super().__init__(raw)
        self.__dict__["root"] = root

    def path(self, key: str) -> Path:
        """Chemin absolu ; la variable PPMI_RAW_DIR redirige les données brutes (ex. jeu synthétique en CI)."""
        rel = os.environ.get("PPMI_RAW_DIR") if key == "raw" and os.environ.get("PPMI_RAW_DIR") else self.paths[key]
        return (Path(rel) if Path(rel).is_absolute() else self.root / rel).resolve()

    def make_dirs(self) -> Config:
        for key in ("interim", "processed", "reports", "metrics", "models"):
            self.path(key).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def fingerprint(self) -> str:
        """Empreinte courte de la configuration complète, enregistrée avec chaque résultat."""
        return hashlib.sha256(json.dumps(self, sort_keys=True, default=str).encode()).hexdigest()[:12]


def find_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / PARAMS_FILE).is_file():
            return folder
    raise FileNotFoundError(f"{PARAMS_FILE} introuvable depuis {here}")


def load_config(params: str | Path | None = None) -> Config:
    path = Path(params) if params else find_root() / PARAMS_FILE
    return Config(yaml.safe_load(path.read_text(encoding="utf-8")), path.resolve().parent)
