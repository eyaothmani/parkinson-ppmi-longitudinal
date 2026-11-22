"""Indicateur de dérive des entrées (PSI) : stable sur la même loi, élevé après un décalage."""
import numpy as np

from parkinson_mm.monitoring.drift import psi


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    edges = np.quantile(rng.normal(0, 1, 5000), np.linspace(0, 1, 11)).tolist()
    assert psi(rng.normal(0, 1, 2000), edges) < 0.05
    assert psi(rng.normal(2, 1, 2000), edges) > 0.25
