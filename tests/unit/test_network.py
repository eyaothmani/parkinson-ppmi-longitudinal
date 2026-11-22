"""Réseaux MLP (« dernière visite ») et GRU (« trajectoire ») : sorties finies, et ils lisent ce qu'ils disent lire."""
import pytest
import torch

from parkinson_mm.models.network import LongitudinalNet


def batch(n=8, steps=5, features=4):
    return {"seq": torch.randn(n, steps, features), "static": torch.randn(n, 3)}


@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_forward_shapes(kind):
    pred = LongitudinalNet(kind, d_step=4, d_static=3, d=16)(batch())
    assert pred.shape == (8,) and torch.isfinite(pred).all()


@pytest.mark.parametrize("kind, reads_past", [("mlp", False), ("gru", True)])
def test_only_the_gru_reads_past_visits(kind, reads_past):
    """Le MLP ne voit que la visite courante (dernière case) ; le GRU dépend aussi des visites passées."""
    torch.manual_seed(0)
    net = LongitudinalNet(kind, 4, 3, d=16).eval()
    b = batch()
    other = {**b, "seq": b["seq"].clone()}
    other["seq"][:, :-1] += 5.0
    with torch.no_grad():
        same = torch.allclose(net(b), net(other))
    assert same is not reads_past


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        LongitudinalNet("transformer", 4, 3)
