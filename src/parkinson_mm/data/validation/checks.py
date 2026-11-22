"""Moteur minimal de contrôles qualité (fail fast).

Pandera ou Great Expectations auraient fonctionné, mais ~30 lignes suffisent pour nos besoins (ADR-0006) :
chaque contrôle est une fonction qui renvoie True/False, avec une gravité « error » (bloquant) ou « warning ».
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


class DataValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    test: Callable[..., bool]
    severity: str = "error"


def run_checks(checks: list[Check], *data, stage: str) -> dict:
    results = []
    for check in checks:
        try:
            ok = bool(check.test(*data))
        except Exception as exc:  # un contrôle qui plante est un échec, jamais un succès silencieux
            ok, check = False, Check(f"{check.name} (exception : {exc!r})", check.test, check.severity)
        results.append({"check": check.name, "ok": ok, "severity": check.severity})
    errors = [r for r in results if not r["ok"] and r["severity"] == "error"]
    warnings = [r for r in results if not r["ok"] and r["severity"] == "warning"]
    return {"stage": stage, "passed": not errors, "errors": len(errors), "warnings": len(warnings), "checks": results}


def enforce(report: dict) -> dict:
    """Lève une erreur explicite si un contrôle bloquant a échoué."""
    if not report["passed"]:
        failed = ", ".join(r["check"] for r in report["checks"] if not r["ok"] and r["severity"] == "error")
        raise DataValidationError(f"[{report['stage']}] contrat non respecté : {failed}")
    return report
