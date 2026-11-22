"""Tableau de bord d'exploration (HTML, CSS, JavaScript sans framework) servi par l'API sur /app/.
Il n'appelle que les routes de l'API : aucune donnée n'est lue directement par le navigateur."""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).parent / "web"
for kind, suffix in (("text/javascript", ".js"), ("text/css", ".css"), ("image/svg+xml", ".svg")):
    mimetypes.add_type(kind, suffix)     # sous Windows, le registre peut annoncer .js en text/plain : modules refusés


def mount_web(app: FastAPI) -> None:
    app.mount("/app", StaticFiles(directory=WEB, html=True), name="app")
    app.add_api_route("/", lambda: RedirectResponse("/app/"), include_in_schema=False)
