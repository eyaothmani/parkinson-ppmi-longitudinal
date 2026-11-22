"""Notebooks du dépôt (lus sur GitHub par l'encadrant) : des sorties seulement si elles sont déclarées
(`keep_output`), et jamais un tableau par patient (colonne ou index PATNO) : données PPMI sous DUA."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
PATIENT_TABLE = re.compile(r"<th>\s*PATNO\s*</th>|^\s*PATNO\s", re.MULTILINE)   # en-tête HTML ou texte


def outputs_text(notebook: dict) -> list[str]:
    texts = []
    for cell in notebook["cells"]:
        for out in cell.get("outputs", []) if cell["cell_type"] == "code" else []:
            data = out.get("data", {})
            for part in (out.get("text", ""), data.get("text/html", ""), data.get("text/plain", "")):
                texts.append("".join(part) if isinstance(part, list) else part)
    return texts


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_declared_and_never_per_patient(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    texts = [t for t in outputs_text(notebook) if t]
    if texts:
        assert notebook["metadata"].get("keep_output") is True, f"{path.name} : sorties non déclarées (keep_output)"
    assert not [t[:80] for t in texts if PATIENT_TABLE.search(t)], f"{path.name} : tableau par patient en sortie"
