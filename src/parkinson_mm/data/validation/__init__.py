"""Contrats de données : le pipeline s'arrête si une hypothèse sur les données est violée."""
from .checks import Check, DataValidationError, enforce, run_checks  # noqa: F401
