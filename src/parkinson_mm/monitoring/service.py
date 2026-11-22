"""Compteurs de service en mémoire (thread-safe) : requêtes, erreurs, latences, entrées récentes."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from .drift import psi


class ServiceMetrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.requests = self.errors = 0
        self.latencies: deque = deque(maxlen=window)
        self.inputs: deque = deque(maxlen=window)
        self.started = time.time()

    def record(self, latency_s: float, error: bool, features: dict | None = None) -> None:
        with self._lock:
            self.requests += 1
            self.errors += int(error)
            self.latencies.append(latency_s)
            if features:
                self.inputs.append(features)

    def snapshot(self, reference: dict | None = None) -> dict:
        with self._lock:
            latencies = np.array(self.latencies) if self.latencies else np.array([np.nan])
            report = {"uptime_s": round(time.time() - self.started), "requests": self.requests,
                      "error_rate": self.errors / max(1, self.requests),
                      "latency_p50_ms": float(np.nanpercentile(latencies, 50) * 1000),
                      "latency_p95_ms": float(np.nanpercentile(latencies, 95) * 1000)}
            if reference and len(self.inputs) >= 30:   # le PSI n'a de sens qu'avec assez d'observations
                recent = pd.DataFrame(list(self.inputs))
                report["psi"] = {c: round(psi(recent[c], e), 4) for c, e in reference.items() if c in recent}
            return report
