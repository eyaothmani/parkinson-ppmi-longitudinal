"""Étape « evaluate » : évaluation UNIQUE sur le test (patients jamais vus) des modèles choisis en CV."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import mae_ci, mae_difference
from ..evaluation.metrics import regression_metrics
from ..explainability.local import reference_values
from ..training.fit import fit
from ..training.settings import from_params
from ..utils.env import set_seed
from ..utils.io import write_json
from .common import device, load_modeling_data, logger, specs, tables_dir
from .comparison import compare
from .release import release
from .release_compared import release_compared
from .selection import choose_models


def evaluate(cfg, fast: bool = False) -> dict:
    set_seed(cfg.project.seed)
    samples, X, mask = load_modeling_data(cfg)
    cv = pd.read_csv(tables_dir(cfg, "models") / "cv_summary.csv")
    chosen, selected = choose_models(cfg, cv)
    dev, test = np.where(samples["split"] == "dev")[0], np.where(samples["split"] == "test")[0]
    settings, by_name = from_params(cfg, device(), fast), {s.name: s for s in specs(cfg)}
    fitted, rows = {}, []
    for name in chosen:   # réentraîné sur tout le dev ; réseaux = ensemble de graines
        fitted[name] = fit(by_name[name], samples, X, mask, dev, settings, cfg.project.seed,
                           n_seeds=1 if fast else cfg.training.final_seeds)
        part = samples.iloc[test][["PATNO", "np3", "y_next", "gap_months"]].rename(columns={"np3": "y_last",
                                                                                         "y_next": "y"})
        rows.append(part.assign(model=name, sample_id=test, pred=fitted[name].predict(samples.iloc[test], X[test],
                                                                                      mask[test])))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(tables_dir(cfg, "models") / "test_predictions.csv", index=False)
    n_boot = cfg.evaluation.bootstrap_samples
    results = {n: {**regression_metrics(g["y"], g["y_last"], g["pred"]), "MAE_IC95": mae_ci(predictions, n, n_boot)}
               for n, g in predictions.groupby("model")}
    comparisons = [mae_difference(predictions, n, "Persistance", n_boot) for n in chosen if n != "Persistance"]
    pd.DataFrame(results).T.to_csv(tables_dir(cfg, "models") / "test_results.csv")
    pd.DataFrame(comparisons).to_csv(tables_dir(cfg, "models") / "test_bootstrap.csv", index=False)
    pairs = compare(cfg, fitted, predictions, samples, X, mask, test)   # les modèles appris se distinguent-ils ?
    reference = reference_values(samples.iloc[dev], X[dev], mask[dev])  # profil typique du dev, pour /explain
    decision = release(cfg, fitted[selected], selected, results[selected], comparisons, cv, predictions, samples, dev,
                       reference)
    others = release_compared(cfg, fitted, selected, results, comparisons, cv, predictions, samples, dev, reference)
    result = {"selected": selected, "test": results, "vs_persistence": comparisons, "comparaisons_appariees": pairs,
              "registry": decision, "registry_comparaison": others}
    write_json(result, cfg.path("metrics") / "test.json")
    logger(cfg, "evaluate").info("%s : MAE test %.3f -> %s", selected, results[selected]["MAE"], decision)
    return result
