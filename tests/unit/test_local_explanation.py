"""Explication locale : seul un groupe utilisé par le modèle a un effet, avec le bon signe et la bonne taille."""
import numpy as np
import pandas as pd
import pytest

from parkinson_mm.explainability.local import explain_local, reference_values
from parkinson_mm.explainability.permutation import SEQUENCE_CHANNELS


class OnlyHistory:
    """Modèle factice : score actuel + 0,5 × (score précédent − 20) ; il ignore tout le reste."""

    def predict(self, samples, X=None, mask=None):
        return samples["np3"].to_numpy(float) + 0.5 * (samples["np3_prev"].to_numpy(float) - 20)


def cohort(n=60):
    rng = np.random.default_rng(0)
    samples = pd.DataFrame({"np3": rng.normal(25, 5, n), "np3_prev": rng.normal(20, 5, n),
                            "nhy": rng.integers(1, 4, n).astype(float), "target_off": 1.0, "target_on": 0.0,
                            "target_untreated": 0.0})
    X = np.full((n, 3, len(SEQUENCE_CHANNELS)), np.nan, dtype=np.float32)
    return samples, X, np.ones((n, 3), dtype=np.float32)


def test_only_the_used_group_has_an_effect_with_the_right_sign():
    samples, X, mask = cohort()
    reference = reference_values(samples, X, mask)
    patient = samples.iloc[[0]].assign(np3_prev=reference["tabular"]["np3_prev"] + 10)   # 10 pts au-dessus du typique
    effects = explain_local(OnlyHistory(), patient, X[:1], mask[:1], reference).set_index("groupe")["effet"]
    assert effects.index[0] == "UPDRS III (historique)"
    assert effects.iloc[0] == pytest.approx(5.0)
    assert (effects.iloc[1:].abs() < 1e-9).all()


def test_states_are_replaced_by_the_most_frequent_combination():
    samples, X, mask = cohort()
    samples.loc[:9, ["target_off", "target_on"]] = [0.0, 1.0]
    ref = reference_values(samples, X, mask)["tabular"]
    assert (ref["target_off"], ref["target_on"], ref["target_untreated"]) == (1.0, 0.0, 0.0)


def test_a_missing_value_is_not_explained_away():
    """Un champ non saisi reste non saisi : son effet n'est pas compté (pas d'effet de l'indicateur « manquant »)."""
    samples, X, mask = cohort()
    reference = reference_values(samples, X, mask)
    patient = samples.iloc[[0]].assign(np3_prev=np.nan)
    effects = explain_local(OnlyHistory(), patient, X[:1], mask[:1], reference).set_index("groupe")["effet"]
    assert np.isnan(effects["UPDRS III (historique)"]) or effects["UPDRS III (historique)"] == pytest.approx(0.0)
