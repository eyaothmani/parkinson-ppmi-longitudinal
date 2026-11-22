"""Étape « stats » : plan d'analyse pré-spécifié dans params.yaml (modèles mixtes, FDR, chaîne M1 -> M4)."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..stats.factors import build_patient_factors
from ..stats.lme import BASE_ADJUSTMENT, fit_lme, prepare, tidy
from ..stats.screening import screen_factors
from ..stats.slopes import individual_slopes
from ..training.tracking import log_artifact, model_run
from ..utils.io import write_json
from ..utils.lineage import lineage
from .common import data_version, logger, read_long, tables_dir
from .run_datasets import statistics


def modality_chain(d: pd.DataFrame, chain: dict) -> pd.DataFrame:
    """Modèles emboîtés M1 ⊂ M2 ⊂ M3 ⊂ M4 ajustés sur la MÊME sous-population (sinon l'AIC n'est pas comparable)."""
    needed = [v for variables in chain.values() for v in variables if v in d]
    common, terms, rows, previous = d.dropna(subset=needed), [], [], None
    for name, variables in chain.items():
        terms += [v for v in variables if v in common and common[v].nunique() > 1]
        formula = f"np3 ~ time_years * ({' + '.join(terms)}) + C(med, Treatment('untreated'))"
        fit = fit_lme(formula, common, reml=False)
        lrt = np.nan if previous is None else chi2.sf(2 * (fit.llf - previous.llf),
                                                      len(fit.params) - len(previous.params))
        rows.append({"modele": name, "patients": common["PATNO"].nunique(), "AIC": fit.aic, "BIC": fit.bic,
                     "LRT_p": lrt, "variance_pentes": float(fit.cov_re.iloc[1, 1]), "converge": bool(fit.converged)})
        previous = fit
    return pd.DataFrame(rows)


def stats(cfg) -> dict:
    long, patients = read_long(cfg)
    d = prepare(long, build_patient_factors(long, patients), cfg.stats.min_visits)
    base = fit_lme("np3 ~ time_years + " + BASE_ADJUSTMENT, d)
    tidy(base).to_csv(tables_dir(cfg, "stats") / "modele_base.csv")
    screening = screen_factors(d, dict(cfg.stats.factors), cfg.stats.fdr_alpha)
    screening.to_csv(tables_dir(cfg, "stats") / "criblage_facteurs_FDR.csv", index=False)
    chain = modality_chain(d, dict(cfg.stats.modality_chain))
    chain.to_csv(tables_dir(cfg, "stats") / "chaine_modalites.csv", index=False)
    individual_slopes(base, d).to_csv(cfg.path("processed") / "pd_score_evolution.csv", index=False)
    result = {"mean_slope_pts_per_year": float(base.params["time_years"]),
              "sd_individual_slopes": float(np.sqrt(base.cov_re.iloc[1, 1])),
              "factors_significant_fdr": screening.loc[screening["significatif_FDR"], "facteur"].tolist(),
              "lineage": lineage(cfg, data_version(cfg))}
    write_json(result, cfg.path("metrics") / "stats.json")
    plan = {"facteurs_testes": len(screening), "fdr_alpha": cfg.stats.fdr_alpha, "min_visits": cfg.stats.min_visits}
    with model_run(cfg, "objectif3_facteurs", "Criblage LME + FDR", plan, {"stage": "stats"}, data_version(cfg),
                   statistics(long)):
        mlflow.log_metrics({"pente_moyenne": result["mean_slope_pts_per_year"],
                            "ecart_type_pentes": result["sd_individual_slopes"],
                            "facteurs_significatifs": len(result["factors_significant_fdr"])})
        log_artifact(screening, "criblage_facteurs_FDR.csv")
        log_artifact(chain, "chaine_modalites.csv")
    logger(cfg, "stats").info("pente moyenne %.2f pts/an", result["mean_slope_pts_per_year"])
    return result
