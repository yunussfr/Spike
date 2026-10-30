"""Faz 3: DecisionReadout testleri (K2, K3)."""

import copy
import inspect

import pytest
import torch

from src.decision import DecisionReadout
from tests.conftest import CFG

F, D = 20, 2


def make(rule: str) -> DecisionReadout:
    cfg = copy.deepcopy(CFG)
    cfg["decision"]["rule"] = rule
    return DecisionReadout.from_config(cfg, n_features=F, D_out=D)


def test_signature() -> None:
    """K3: güncelleme yalnızca δ ve η alır; e girişi yok."""
    assert list(inspect.signature(DecisionReadout.update).parameters) == ["self", "delta", "eta"]
    assert list(inspect.signature(DecisionReadout.forward).parameters) == ["self", "r", "beta"]


@pytest.mark.parametrize("rule", ["plain", "normalized"])
def test_forward_shape(rule: str) -> None:
    assert make(rule)(torch.randn(4, F)).shape == (4, D)


def test_plain_rule_matches_architecture() -> None:
    """plain: ΔW_a = η · δ · rᵀ (ARCHITECTURE.md §3.6)."""
    dec = make("plain")
    r = torch.randn(1, F)
    dec(r)
    delta, eta = torch.tensor([[0.5, -1.0]]), 0.1
    dec.update(delta, eta)
    assert torch.allclose(dec.W_a, eta * delta.T @ r)


def test_update_changes_only_w_a() -> None:
    dec = make("normalized")
    r = torch.randn(1, F)
    dec(r)
    mu, var = dec.norm.mu.clone(), dec.norm.var.clone()
    dec.update(torch.ones(1, D), eta=0.01)
    assert dec.W_a.abs().sum() > 0
    assert torch.equal(dec.norm.mu, mu) and torch.equal(dec.norm.var, var)   # update istatistiğe dokunmaz


def test_normalized_converges_with_large_features() -> None:
    """[f, v] ölçeğinde girdide ıraksamaz, doğrusal hedefte hatayı azaltır."""
    torch.manual_seed(0)
    dec = make("normalized")
    w_true = torch.randn(F, D)
    errs = []
    for _ in range(3000):
        r = torch.randn(1, F) * 300 + 50
        y = (r - 50) / 300 @ w_true + 0.5
        delta = y - dec(r)
        dec.update(delta, eta=CFG["regulators"]["defaults"]["eta"])
        errs.append(delta.pow(2).mean().item())
    assert all(torch.isfinite(torch.tensor(errs)))
    assert sum(errs[-500:]) < 0.5 * sum(errs[:500])


def test_reset_keeps_weights() -> None:
    dec = make("normalized")
    dec(torch.randn(1, F))
    dec.update(torch.ones(1, D), eta=0.01)
    W = dec.W_a.clone()
    dec.reset_state()
    assert torch.equal(dec.W_a, W)
    with pytest.raises(RuntimeError):
        dec.update(torch.ones(1, D), eta=0.01)
