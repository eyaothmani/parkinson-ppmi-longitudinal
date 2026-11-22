"""Corpus de l'assistant : les sources du manifeste, et elles seules, découpées en pages ou en sections.
Garde-fous : jamais un fichier de données (data/, reports/tables/, mlflow.db…), jamais un tableau par patient."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .extract import clean, read_parts

FORBIDDEN = ("data/", "reports/tables/", "models/", "mlruns/", "mlflow.db")
PATIENT_TABLE = re.compile(r"\|\s*PATNO\s*\||^\s*PATNO\s*[,;\t]", re.M)   # en-tête d'un tableau ou d'un CSV


@dataclass(frozen=True)
class Page:
    doc_id: str
    citation: str
    where: str          # « p. 3 » (PDF) ou « section « Limites » » (Markdown)
    text: str


def load_manifest(root: Path, manifest: str) -> list[dict]:
    root = Path(root).resolve()
    docs = yaml.safe_load((root / manifest).read_text(encoding="utf-8"))["documents"]
    for doc in docs:
        path = (root / doc["fichier"]).resolve()
        if root not in path.parents or path.relative_to(root).as_posix().startswith(FORBIDDEN):
            raise ValueError(f"Source interdite pour l'assistant documentaire : {doc['fichier']}")
    return docs


def load_pages(root: Path, manifest: str) -> list[Page]:
    pages = []
    for doc in load_manifest(root, manifest):
        for where, text in read_parts(Path(root) / doc["fichier"]):
            if PATIENT_TABLE.search(text):
                raise ValueError(f"Tableau par patient détecté dans {doc['fichier']} ({where}) : source refusée")
            if clean(text):
                pages.append(Page(doc["id"], doc["citation"], where, clean(text)))
    return pages
