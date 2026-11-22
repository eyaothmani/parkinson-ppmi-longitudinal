"""GRU-D (ADR-0018) : temps écoulé par variable, remplissage ignoré, quantiles ordonnés, cibles absentes masquées."""
import numpy as np
import torch

from parkinson_mm.features.grud import GRUDScaler, elapsed, forward_fill
from parkinson_mm.models.multi_horizon import MultiHorizonNet
from parkinson_mm.training import horizon_loss


def sequence():
    """Alignée à droite : 1 visite de remplissage, puis t = 0, 1, 3 ans ; la variable 1 manque à t = 1 an."""
    X = np.array([[[np.nan, np.nan, np.nan], [0.0, 10.0, 1.0], [1.0, np.nan, 2.0], [3.0, 16.0, 3.0]]])
    return X, np.array([[0.0, 1.0, 1.0, 1.0]])


def test_elapsed_time_accumulates_while_a_variable_is_missing():
    X, visit = sequence()
    m = ~np.isnan(X) & (visit[..., None] > 0)
    delta = elapsed(X[..., 0], m, visit)
    assert delta[0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0]
    assert delta[0, :, 1].tolist() == [0.0, 0.0, 1.0, 3.0]    # dernière mesure à t = 0 : 3 ans plus tard


def test_forward_fill_keeps_the_last_observed_value():
    X, visit = sequence()
    m = ~np.isnan(X) & (visit[..., None] > 0)
    assert forward_fill(np.nan_to_num(X), m)[0, 2, 1] == 10.0


def batch(n=4):
    X, visit = sequence()
    arrays = GRUDScaler().fit(X, visit).transform(np.repeat(X, n, 0), np.repeat(visit, n, 0), time_index=0)
    return {k: torch.as_tensor(v) for k, v in arrays.items()} | {"static": torch.randn(n, 2)}


def test_quantiles_never_cross_and_padding_is_ignored():
    torch.manual_seed(0)
    net = MultiHorizonNet("grud", 3, 2, n_horizons=3, n_aux=2, d=8).eval()
    b = batch()
    with torch.no_grad():
        q, aux = net(b)
        changed = {**b, "x": b["x"].clone()}
        changed["x"][:, 0] = 99.0                    # visite de remplissage : ne doit rien changer
        assert torch.allclose(net(changed)[0], q)
    assert q.shape == (4, 3, 3) and aux.shape == (4, 3, 2)
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()


def test_missing_targets_do_not_contribute_to_the_loss():
    pred, quantiles = torch.zeros(2, 3, 3), torch.tensor([0.1, 0.5, 0.9])
    y = torch.tensor([[1.0, np.nan, np.nan], [1.0, np.nan, np.nan]])
    other = y.clone()
    other[:, 1:] = 0.0
    assert horizon_loss.pinball(pred, y, quantiles) > 0
    assert not torch.isnan(horizon_loss.pinball(pred, y, quantiles))
    assert horizon_loss.pinball(pred, y, quantiles) > horizon_loss.pinball(pred, other, quantiles)
