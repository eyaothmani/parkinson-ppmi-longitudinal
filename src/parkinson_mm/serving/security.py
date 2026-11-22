"""Clé d'API (en-tête X-API-Key) et limitation de débit par clé, en mémoire (suffisant pour un démonstrateur)."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_calls: dict[str, deque] = defaultdict(deque)


def require_key(key: str | None = Depends(_header)) -> str:
    expected = os.environ.get("API_KEY")
    if expected and key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé d'API invalide ou absente (en-tête X-API-Key).")
    caller, now = key or "anonyme", time.time()
    recent = _calls[caller]
    while recent and now - recent[0] > 60:      # fenêtre glissante d'une minute
        recent.popleft()
    if len(recent) >= int(os.environ.get("RATE_LIMIT_PER_MIN", "60")):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Trop de requêtes : réessayer dans une minute.")
    recent.append(now)
    return caller
