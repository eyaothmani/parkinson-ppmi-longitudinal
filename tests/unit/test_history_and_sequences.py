"""Les variables d'entrée n'utilisent jamais le futur, et les séquences ne mélangent jamais deux patients."""
import numpy as np

from parkinson_mm.features.history import with_history
from parkinson_mm.features.samples import make_training_samples
from parkinson_mm.features.sequences import build_sequences


def test_history_ignores_future_visits(visits):
    before = with_history(visits)
    changed = visits.copy()
    changed.loc[2, "np3"] = 99.0                    # on modifie la DERNIÈRE visite du patient 1
    after = with_history(changed)
    for col in ("np3_prev", "np3_slope_hist", "np3_mean_hist"):
        np.testing.assert_allclose(before[col].iloc[:2], after[col].iloc[:2])


def test_samples_target_is_next_visit(visits):
    samples = make_training_samples(visits, visits[["PATNO"]].drop_duplicates(), min_gap=3, max_gap=24)
    first = samples[samples["PATNO"] == 1].iloc[0]
    assert first["y_next"] == 22.0 and first["delta"] == 2.0 and first["gap_months"] == 6


def test_sequences_are_right_aligned_and_per_patient(visits):
    samples = make_training_samples(visits, visits[["PATNO"]].drop_duplicates())
    X, mask = build_sequences(visits, samples, ["np3"], max_len=4)
    patient2 = np.where(samples["PATNO"] == 2)[0][0]
    assert mask[patient2].tolist() == [0, 0, 0, 1]   # une seule visite passée, en dernière position
    assert X[patient2, -1, 0] == 30.0
