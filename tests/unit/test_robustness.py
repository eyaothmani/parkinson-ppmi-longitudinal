"""Sprint 6 : groupes lisibles, erreurs par groupe, intervalle conforme calculé SANS le pli de l'exemple."""
import numpy as np
import pandas as pd

from parkinson_mm.evaluation.conformal import coverage, cross_conformal
from parkinson_mm.evaluation.robustness import subgroup_errors, with_groups
from parkinson_mm.evaluation.subgroups import assign_groups

SAMPLES = pd.DataFrame({"nhy": [1, 2, 3, np.nan], "target_state": ["OFF", "ON", "untreated", "OFF"],
                        "disease_dur_years": [1, 3, 8, 0.5], "lrrk2": [1, 0, np.nan, 0], "gba": [0, 1, np.nan, 0],
                        "male": [1, 0, 1, 0], "age": [55, 65, 75, 61],
                        "bl_date": ["2011-03-01", "2016-01-01", "2021-05-01", "2013-01-01"]})


def predictions(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y, patient = rng.normal(25, 8, n), np.arange(n) // 4
    good = pd.DataFrame({"PATNO": patient, "sample_id": np.arange(n) % 4, "fold": patient % 5, "y": y,
                         "pred": y + rng.normal(0, 3, n), "model": "A"})
    return pd.concat([good, good.assign(pred=y + rng.normal(0, 9, n), model="B")], ignore_index=True)


def test_groups_are_readable_and_unknown_stays_empty():
    g = assign_groups(SAMPLES)
    assert g["Stade (Hoehn & Yahr)"].astype(str).tolist()[:3] == ["H&Y 0–1", "H&Y 2", "H&Y ≥ 3"]
    assert g["Génétique"].tolist()[:2] == ["LRRK2", "GBA"] and pd.isna(g.loc[2, "Génétique"])
    assert g.loc[3, "Génétique"] == "non porteur" and pd.isna(g.loc[3, "Stade (Hoehn & Yahr)"])
    assert g["Période d'inclusion"].astype(str).tolist() == ["≤ 2014", "2015–2019", "≥ 2020", "≤ 2014"]


def test_errors_per_group_rank_the_models():
    data = with_groups(predictions(), assign_groups(SAMPLES))
    table = subgroup_errors(data, ["Sexe"], ["A", "B"], n_boot=200, seed=0)
    mae = table.set_index(["groupe", "modèle"])["MAE"]
    assert (mae.xs("A", level="modèle") < mae.xs("B", level="modèle")).all()
    assert (table["IC95_bas"] <= table["MAE"]).all() and (table["MAE"] <= table["IC95_haut"]).all()


def test_conformal_interval_covers_about_80_percent():
    out = cross_conformal(predictions().query("model == 'A'").reset_index(drop=True), 0.8)
    table = coverage(out.assign(Sexe="tous"), ["Sexe"])
    assert abs(table.iloc[0]["couverture"] - 0.8) < 0.04 and table.iloc[0]["exemples"] == 2000


def test_conformal_interval_never_uses_the_fold_of_the_example():
    oof = predictions().query("model == 'A'").reset_index(drop=True)
    oof.loc[oof["fold"] == 0, "pred"] += 30                     # le pli 0 se trompe énormément…
    fold0 = cross_conformal(oof, 0.8).query("fold == 0")
    assert fold0["demi_largeur"].iloc[0] < 10                   # … mais son intervalle ne voit pas ses erreurs
    assert fold0["couvert"].mean() < 0.2                        # donc il est mal couvert : la mesure reste honnête
