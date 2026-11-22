"""Figures du sprint 6 : MAE par sous-groupe, couverture de l'intervalle par sous-groupe, résidus."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import INK2, C, save


def _labels(d) -> list[str]:
    return [f"{c} · {g}" for c, g in zip(d["critère"], d["groupe"], strict=True)]


def _finish(fig, ax, title, folder, name):
    ax.set_title(title)
    path = save(fig, folder, name)
    plt.close(fig)
    return path


def mae_by_group(errors, model, reference, folder):
    d = errors[errors["modèle"] == model].reset_index(drop=True)
    ref = errors[errors["modèle"] == reference].set_index(["critère", "groupe"])["MAE"]
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(d) + 1.2))
    ax.errorbar(d["MAE"], y, xerr=[d["MAE"] - d["IC95_bas"], d["IC95_haut"] - d["MAE"]], fmt="o",
                color=C["blue"], label=f"{model} (IC 95 %)")
    ax.scatter([ref.get((c, g)) for c, g in zip(d["critère"], d["groupe"], strict=True)], y, marker="|", s=150,
               color=INK2, label=reference)
    ax.set_yticks(y, _labels(d))
    ax.set_xlabel("MAE (points MDS-UPDRS III)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    return _finish(fig, ax, "Erreur par sous-groupe (validation croisée)", folder, "mae_par_groupe")


def coverage_by_group(cover, level, folder):
    d = cover.reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(d) + 1.2))
    ax.barh(y, 100 * d["couverture"], color=C["blue"], height=0.6)
    ax.axvline(100 * level, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, _labels(d))
    ax.set_xlim(0, 100)
    ax.set_xlabel("% des scores réels dans l'intervalle")
    title = f"Couverture de l'intervalle à {100 * level:.0f} % (validation croisée)"
    return _finish(fig, ax, title, folder, "couverture_par_groupe")


def residuals(covered, folder):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.scatter(covered["pred"], covered["y"] - covered["pred"], s=4, alpha=0.25, color=C["blue"])
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("score prédit")
    ax.set_ylabel("résidu (réel − prédit)")
    return _finish(fig, ax, "Résidus du modèle retenu (validation croisée)", folder, "residus")
