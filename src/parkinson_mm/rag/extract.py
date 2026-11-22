"""Extraction du texte : PDF page par page (pypdf), Markdown section par section ; nettoyage léger."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

REFERENCES = re.compile(r"^\s*(references|bibliography|références)\s*$", re.I | re.M)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)              # ligatures des PDF : « ﬂuid » -> « fluid »
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # mot coupé en fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def cut_references(pages: list[str]) -> list[str]:
    """Retire la bibliographie (elle attire la recherche sans répondre) ; cherchée dans la 2e moitié seulement."""
    for i in range(len(pages) // 2, len(pages)):
        match = REFERENCES.search(pages[i])
        if match:
            return pages[:i] + [pages[i][: match.start()]]
    return pages


def read_parts(path: Path) -> list[tuple[str, str]]:
    """[(où, texte)] : « p. 3 » pour un PDF, « section « Limites » » pour un Markdown."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = cut_references([page.extract_text() or "" for page in PdfReader(path).pages])
        return [(f"p. {i}", text) for i, text in enumerate(pages, 1)]
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)
    return [(f"section « {s.splitlines()[0].strip()} »" if i else "introduction", s) for i, s in enumerate(sections)]
