"""Intervalles de confiance par bootstrap sur les PATIENTS (on rééchantillonne des patients, pas des visites,
car les visites d'un même patient ne sont pas indépendantes)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _per_patient(errors: pd.Series, patients: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    grouped = errors.groupby(patients.to_numpy()).agg(["sum", "size"])
    return grouped["sum"].to_numpy(), grouped["size"].to_numpy()


def _resample(sums, sizes, n_boot: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
    return sums[idx].sum(1) / sizes[idx].sum(1)


def mae_ci(df: pd.DataFrame, model: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    d = df[df["model"] == model]
    boots = _resample(*_per_patient((d["pred"] - d["y"]).abs(), d["PATNO"]), n_boot, seed)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mae_difference(df: pd.DataFrame, model_a: str, model_b: str, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """MAE(a) − MAE(b) avec IC (95 % par défaut). Un IC entièrement négatif = a significativement meilleur que b."""
    a = df[df["model"] == model_a].set_index("sample_id")
    b = df[df["model"] == model_b].set_index("sample_id").loc[a.index]
    diff = (a["pred"] - a["y"]).abs() - (b["pred"] - b["y"]).abs()
    sums, sizes = _per_patient(diff, a["PATNO"])
    boots, estimate = _resample(sums, sizes, n_boot, seed), sums.sum() / sizes.sum()
    low, high = np.percentile(boots, [50 * alpha, 100 - 50 * alpha])
    return {"comparaison": f"{model_a} − {model_b}", "diff_MAE": float(estimate),
            "IC95_bas": float(low), "IC95_haut": float(high)}


def pairwise(df: pd.DataFrame, models: list[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Différences appariées entre TOUS les couples de modèles, IC corrigé de Bonferroni (k couples) :
    deux modèles ne sont dits « distincts » que si leur IC corrigé exclut 0."""
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    alpha = 0.05 / max(1, len(pairs))
    out = pd.DataFrame([mae_difference(df, a, b, n_boot, seed, alpha) for a, b in pairs])
    out = out.rename(columns={"IC95_bas": "IC_bas", "IC95_haut": "IC_haut"})
    out["niveau_IC"] = f"{100 * (1 - alpha):.1f} % (Bonferroni, {len(pairs)} couples)"
    out["distincts"] = (out["IC_haut"] < 0) | (out["IC_bas"] > 0)
    return out
