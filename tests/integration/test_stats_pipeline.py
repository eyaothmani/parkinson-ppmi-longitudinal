"""De bout en bout : plan d'analyse statistique (modèles mixtes, criblage FDR, score d'évolution)."""
import pandas as pd


def test_stats_stage_produces_tables_and_slopes(prepared, analysed):
    result = analysed["stats"]
    tables = prepared.path("reports") / "tables" / "stats"
    screening = pd.read_csv(tables / "criblage_facteurs_FDR.csv")
    assert {"effet_pente_pts_an", "p_FDR"} <= set(screening.columns)
    assert (screening["p_FDR"] >= screening["p"]).all()      # la correction ne peut que relever les p-values
    assert len(pd.read_csv(tables / "chaine_modalites.csv")) == 4
    slopes = pd.read_csv(prepared.path("processed") / "pd_score_evolution.csv")
    assert slopes["score_evolution"].notna().all()
    assert 0 < result["mean_slope_pts_per_year"] < 5      # données synthétiques : ~2 pts/an simulés


def test_state_stage_scores_every_patient_and_validates_on_test(prepared, analysed):
    result = analysed["state"]
    scores = pd.read_csv(prepared.path("processed") / "pd_score_etat.csv")
    assert scores[["niveau_actuel", "vitesse", "niveau_sd"]].notna().all().all()
    assert (scores["niveau_sd"] > 0).all()                   # toujours une incertitude affichée
    assert (prepared.path("models") / "state_model.pkl").exists()
    test = result["test"]
    assert test["n"] > 0 and 0 < test["couverture_80"] <= 1
    diff = test["vs_persistance"]
    assert diff["IC95_bas"] <= diff["diff_MAE"] <= diff["IC95_haut"]
