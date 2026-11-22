"""Règle d'équipe : aucun fichier Python de src/, scripts/, tests/ ou tools/ ne dépasse 60 lignes.

Un fichier court = une responsabilité. Utilisé par pre-commit et par la CI.
"""
import sys
from pathlib import Path

LIMIT = 60
FOLDERS = ("src", "scripts", "tests", "tools")


def too_long(root: Path) -> list[tuple[str, int]]:
    counts = {p: len(p.read_text(encoding="utf-8").splitlines()) for f in FOLDERS for p in (root / f).rglob("*.py")}
    return [(str(p.relative_to(root)), n) for p, n in counts.items() if n > LIMIT]


if __name__ == "__main__":
    offenders = too_long(Path(__file__).resolve().parents[1])
    for path, n in offenders:
        print(f"{path} : {n} lignes (> {LIMIT})")
    sys.exit(1 if offenders else 0)
