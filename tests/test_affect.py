"""Faz 4: AffectSpace ve AffectTargets testleri (K2, K3)."""

import copy
import inspect

import pytest
import torch

from src.affect import AffectSpace, AffectTargets
from tests.conftest import CFG

F = 30


def make() -> AffectSpace:
    return AffectSpace.from_config(copy.deepcopy(CFG), n_features=F, seed=0)


def test_signatures() -> None:
    """K3: update yalnızca q, u, s alır; δ ve e'yi doğrudan görmez."""
    assert list(inspect.signature(AffectSpace.update).parameters) == ["self", "q", "u", "s"]
    assert list(inspect.signature(AffectSpace.forward).parameters) == ["self", "r"]


def test_shapes_and_range() -> None:
    aff = make()
    z, heads = aff(torch.randn(4, F) * 100)
    assert z.shape == (4, CFG["affect"]["k"])
    assert (z.abs() < 1).all()
    assert set(heads) == {"q", "u", "s"} and all(h.shape == (4,) for h in heads.values())


def test_targets_read_only() -> None:
    """Hedefler δ ve e'yi değiştirmez; q = −‖δ‖², u = ‖e‖², ilk adımda s = 0."""
    tg = AffectTargets(0.1, 0.01)
    delta, e = torch.randn(2, 2), torch.randn(2, 5)
    d0, e0 = delta.clone(), e.clone()
    out = tg(delta, e)
    assert torch.equal(delta, d0) and torch.equal(e, e0)
    assert torch.allclose(out["q"], -(d0 ** 2).sum(-1)) and torch.allclose(out["u"], (e0 ** 2).sum(-1))
    assert torch.equal(out["s"], torch.zeros(2))
    out = tg(delta, 2 * e)                                                   # belirsizlik arttı → eğilim > 0
    assert (out["s"] > 0).all()


def test_update_changes_only_own_weights() -> None:
    aff = make()
    Wz, Wh = aff.W_z.clone(), aff.W_h.clone()
    phi_norm = aff.norm.state()
    for t in range(3):                                                       # ilk adımda standart hedef = 0
        aff(torch.randn(1, F))
        aff.update(torch.tensor([-1.0 - t]), torch.tensor([2.0 * t]), torch.tensor([0.5 - t]))
    assert not torch.equal(aff.W_h, Wh) and not torch.equal(aff.W_z, Wz)
    assert phi_norm[2] == 0 and aff.norm.n_seen == 3                         # istatistik yalnızca forward'da


def test_heads_learn_known_target() -> None:
    """r'nin doğrusal (z'den okunabilir) fonksiyonu olan hedeflerde baş hatası düşer.

    Mekanizma testi: öğrenme hızları config varsayılanından bağımsız sabitlenir (lr 0,1, L2 yok).
    """
    torch.manual_seed(0)
    cfg = copy.deepcopy(CFG)
    cfg["affect"].update(lr_z=0.1, lr_head=0.1, preact_l2=0.0)
    aff = AffectSpace.from_config(cfg, n_features=F, seed=0)
    w = torch.randn(F)
    errs = []
    for _ in range(6000):
        r = torch.randn(1, F) * 50 + 10
        v = ((r - 10) / 50 @ w) / F ** 0.5                                   # [1]
        aff(r)
        err = aff.update(-v ** 2, v.abs(), v)
        errs.append(err.pow(2).mean().item())
    assert all(torch.isfinite(torch.tensor(errs)))
    assert sum(errs[-1000:]) < 0.5 * sum(errs[10:1010])


def test_frozen_does_not_learn() -> None:
    aff = make()
    aff(torch.randn(1, F))
    aff.update(torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([0.0]))
    aff.frozen = True
    Wz, Wh, mu = aff.W_z.clone(), aff.W_h.clone(), aff.t_mu.clone()
    aff(torch.randn(1, F))
    aff.update(torch.tensor([-5.0]), torch.tensor([5.0]), torch.tensor([1.0]))
    assert torch.equal(aff.W_z, Wz) and torch.equal(aff.W_h, Wh) and torch.equal(aff.t_mu, mu)


def test_own_generator_does_not_touch_global_rng() -> None:
    torch.manual_seed(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    make()
    assert torch.equal(torch.rand(3), a)


def test_update_requires_forward() -> None:
    with pytest.raises(RuntimeError):
        make().update(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_preact_l2_shrinks_saturation() -> None:
    """λ > 0: hedefsiz (sıfır hata) güncellemede ön-aktivasyon küçülür; λ = 0'da W_z değişmez."""
    for lam, should_change in ((0.0, False), (1.0, True)):
        cfg = copy.deepcopy(CFG)
        cfg["affect"]["preact_l2"] = lam
        aff = AffectSpace.from_config(cfg, n_features=F, seed=0)
        r = torch.randn(1, F)
        aff(r)
        aff.heads = torch.zeros(1, 3)
        aff.t_seen, aff.t_mu, aff.t_var = 10, torch.zeros(3), torch.ones(3)
        aff.frozen = False
        aff.t_rate = 0.0
        a0 = aff.a.abs().sum()
        Wz = aff.W_z.clone()
        aff.update(torch.zeros(1), torch.zeros(1), torch.zeros(1))          # hata = 0
        assert torch.equal(aff.W_z, Wz) != should_change
        if should_change:
            aff(r)
            assert aff.a.abs().sum() < a0
