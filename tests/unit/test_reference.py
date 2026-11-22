"""Persistance ajustée à l'état : le décalage vient des AUTRES plis, jamais du pli évalué."""
import pandas as pd
import pytest

from parkinson_mm.analysis.reference import NAME, adjusted_persistence, changed_share, errors_by_transition, transitions


@pytest.fixture
def toy():
    samples = pd.DataFrame({"med_state": ["OFF", "OFF", "OFF", "OFF", "untreated", "ON"],
                            "target_state": ["ON", "ON", "ON", "ON", "untreated", "OFF"]})
    persistence = pd.DataFrame({"sample_id": range(6), "PATNO": [1, 1, 2, 2, 3, 3], "fold": [0, 0, 1, 1, 0, 1],
                                "y_last": [30.0] * 6, "y": [20.0, 20.0, 26.0, 26.0, 32.0, 40.0],
                                "model": "Persistance", "family": "baseline"})
    persistence["pred"] = persistence["y_last"]
    return samples, persistence


def test_shift_is_learned_on_the_other_folds_only(toy):
    samples, persistence = toy
    out = adjusted_persistence(persistence, samples)
    assert set(out["model"]) == {NAME} and len(out) == len(persistence)
    # OFF -> ON : pli 0 appris sur le pli 1 (Δ = -4), pli 1 appris sur le pli 0 (Δ = -10)
    assert out.loc[0, "pred"] == pytest.approx(26.0) and out.loc[2, "pred"] == pytest.approx(20.0)


def test_unseen_transition_falls_back_to_the_global_change(toy):
    samples, persistence = toy
    out = adjusted_persistence(persistence, samples)
    # « untreated -> untreated » n'existe que dans le pli 0 : pour le pli 0 il est appris sur le pli 1,
    # où il est absent -> variation médiane globale du pli 1 (Δ = -4, -4, +10 -> -4)
    assert out.loc[4, "pred"] == pytest.approx(26.0)


def test_errors_by_transition_counts_each_sample_once(toy):
    samples, persistence = toy
    oof = pd.concat([persistence, adjusted_persistence(persistence, samples)], ignore_index=True)
    oof["transition"] = transitions(samples).reindex(oof["sample_id"]).to_numpy()
    table = errors_by_transition(oof).set_index("transition")
    assert table.loc["OFF -> ON", "n"] == 4 and table["n"].sum() == len(samples)
    assert table.loc["OFF -> ON", NAME] < table.loc["OFF -> ON", "Persistance"]
    assert changed_share(samples, persistence["sample_id"]) == pytest.approx(5 / 6)
