"""Environnement d'exécution : graines aléatoires, GPU/CPU, Google Colab."""
from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les graines pour que deux exécutions donnent le même résultat."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def hardware_report(verbose: bool = True) -> dict:
    """Décrit la machine et choisit le device PyTorch ; conseille Colab s'il n'y a pas de GPU."""
    info = {"python": platform.python_version(), "colab": in_colab(), "cpu_count": os.cpu_count(),
            "torch": None, "device": "cpu", "gpu_name": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info.update(device="cuda", gpu_name=torch.cuda.get_device_name(0))
    except ImportError:
        info["torch"] = "non installé"
    if info["device"] == "cpu":
        info["recommendation"] = ("Pas de GPU : EDA/statistiques en local ; pour l'entraînement DL complet, "
                                  "utiliser Google Colab (GPU T4) ou l'option --fast.")
    else:
        info["recommendation"] = f"GPU disponible ({info['gpu_name']}) : entraînement local possible."
    if verbose:
        for key, value in info.items():
            print(f"{key:>15} : {value}")
    return info
