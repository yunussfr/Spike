"""Faz 5: Regulators testleri (K4 sınırlar, K5 kapalı mod, perturbasyon, elle eşleme)."""

import copy
import math

import pytest
import torch

from src.affect import PARAMS, Regulators
from tests.conftest import CFG

K = CFG["affect"]["k"]


def make(enabled: bool = True, **over: object) -> Regulators:
    cfg = copy.deepcopy(CFG)
    cfg["regulators"]["enabled"] = enabled
    for key, val in over.items():
        if key in ("M", "noise_std"):
            cfg["regulators"]["perturbation"][key] = val
        else:
            cfg["regulators"][key] = val
    return Regulators.from_config(cfg, seed=0)


def test_starts_at_defaults() -> None:
    """W_r = 0 → çıktı tam olarak varsayılanlar (b_r doğru ters çevrilmiş)."""
    reg = make()
    out = reg(torch.randn(4, K))
    for p in PARAMS:
        assert out[p].shape == (4,)
        assert torch.allclose(out[p], torch.full((4,), CFG["regulators"]["defaults"][p]), rtol=1e-5)


@pytest.mark.parametrize("scale", [0.1, 10.0, 1.0e3])
def test_k4_outputs_within_bounds(scale: float) -> None:
    """K4: W_r ne olursa olsun (aşırı büyük dahil) dört değer sınırların içinde."""
    reg = make()
    gen = torch.Generator().manual_seed(1)
    reg.W_r = torch.randn(len(PARAMS), K, generator=gen) * scale
    out = reg(torch.rand(256, K, generator=gen) * 2 - 1)                       # z ∈ (−1, 1)
    for p in PARAMS:
        lo, hi = CFG["regulators"]["bounds"][p]
        assert (out[p] >= lo * (1 - 1e-5)).all() and (out[p] <= hi * (1 + 1e-5)).all(), p
        assert torch.isfinite(out[p]).all()


def test_k4_tau_log_scaled() -> None:
    """K4: ham_τ = 0,5 → τ = √(τ_alt·τ_üst) (log-ölçeğin ortası), doğrusal ortası değil. (simetrik eşleme)"""
    reg = make(tau_ceiling=None)
    reg.b_r.zero_()                                                         # σ(0) = 0,5
    out = reg(torch.zeros(1, K))
    lo, hi = CFG["regulators"]["bounds"]["tau"]
    assert math.isclose(float(out["tau"]), math.sqrt(lo * hi), rel_tol=1e-5)
    glo, ghi = CFG["regulators"]["bounds"]["g"]
    assert math.isclose(float(out["g"]), (glo + ghi) / 2, rel_tol=1e-5)


def test_k5_disabled_is_constant() -> None:
    """K5: enabled false → W_r ne olursa olsun varsayılanlar; update hiçbir şey değiştirmez."""
    reg = make(enabled=False, M=2)
    reg.W_r = torch.ones(len(PARAMS), K) * 5
    W = reg.W_r.clone()
    for _ in range(10):
        assert reg.update(torch.randn(3)) is None
        out = reg(torch.randn(3, K))
        for p in PARAMS:
            assert torch.equal(out[p], torch.full((3,), float(CFG["regulators"]["defaults"][p])))
    assert torch.equal(reg.W_r, W)


def test_first_step_defaults() -> None:
    """İlk adımda z yok → varsayılanlar (ARCHITECTURE.md §6 adım 2)."""
    out = make()(None, B=2)
    assert all(torch.equal(out[p], torch.full((2,), float(CFG["regulators"]["defaults"][p]))) for p in PARAMS)


def test_perturbation_keeps_improvement_and_reverts_worse() -> None:
    """Referans → deneme döngüsü: deneme penceresinde q iyileşirse W_r tutulur, kötüleşirse geri alınır."""
    M = 3
    reg = make(M=M, noise_std=0.5)
    W0 = reg.W_r.clone()
    for _ in range(M):                                                      # referans penceresi: q = −1
        assert reg.update(torch.tensor([-1.0])) is None
    W_try = reg.W_r.clone()
    assert not torch.equal(W_try, W0) and reg.phase == "trial"
    res = None
    for _ in range(M):                                                      # deneme: q = −2 (kötü)
        res = reg.update(torch.tensor([-2.0]))
    assert res == {"q_ref": -1.0, "q_trial": -2.0, "accepted": False}
    assert torch.equal(reg.W_r, W0)
    for _ in range(M):
        reg.update(torch.tensor([-1.0]))
    W_try = reg.W_r.clone()
    for _ in range(M):                                                      # deneme: q = −0,5 (iyi)
        res = reg.update(torch.tensor([-0.5]))
    assert res["accepted"] and torch.equal(reg.W_r, W_try)
    assert (reg.n_trials, reg.n_accepted) == (2, 1)


def test_frozen_and_abort() -> None:
    """Donukken update etkisiz; abort_trial yarım denemeyi geri alır."""
    reg = make(M=2, noise_std=0.5)
    W0 = reg.W_r.clone()
    reg.update(torch.tensor([-1.0]))
    reg.update(torch.tensor([-1.0]))                                         # deneme başladı
    reg.abort_trial()
    assert torch.equal(reg.W_r, W0) and reg.phase == "ref"
    reg.frozen = True
    for _ in range(10):
        assert reg.update(torch.tensor([-1.0])) is None
    assert torch.equal(reg.W_r, W0)


def test_noise_reproducible_and_global_rng_untouched() -> None:
    """Aynı seed → aynı perturbasyon; global RNG durumu değişmez."""
    state = torch.random.get_rng_state()
    regs = [make(M=1, noise_std=0.5) for _ in range(2)]
    for reg in regs:
        reg.update(torch.tensor([-1.0]))
    assert torch.equal(regs[0].W_r, regs[1].W_r)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_hand_mapping_direction() -> None:
    """Elle eşleme: û yüksek (w_u·z > 0) → η varsayılandan büyük, τ küçük; g ve β varsayılanda."""
    reg = make(mode="hand", hand_gain=2.0)
    w_u = torch.tensor([1.0, -0.5, 0.2])[:K]
    reg.set_hand_mapping(w_u)
    hi_u, lo_u = w_u.unsqueeze(0), -w_u.unsqueeze(0)                        # [1, k]
    d = CFG["regulators"]["defaults"]
    out_hi, out_lo = reg(hi_u), reg(lo_u)
    assert out_hi["eta"] > d["eta"] > out_lo["eta"]
    assert out_hi["tau"] < d["tau"] < out_lo["tau"]
    for p in ("g", "beta"):
        assert math.isclose(float(out_hi[p]), d[p], rel_tol=1e-5)
    assert reg.update(torch.tensor([-1.0])) is None                          # hand modunda perturbasyon yok


def test_bad_mode() -> None:
    with pytest.raises(ValueError):
        make(mode="foo")


# --- Asimetrik τ eşlemesi (§3.5, KR-030) ------------------------------------------------

def _tau_at(reg: Regulators, a: float) -> float:
    """W_r'nin τ satırı ilk z boyutunda 1 → τ ön-aktivasyonu a = z_0."""
    reg.W_r.zero_()
    reg.W_r[PARAMS.index("tau"), 0] = 1.0
    z = torch.zeros(1, K)
    z[0, 0] = a
    return float(reg(z)["tau"])


def test_tau_asymmetric_ceiling_and_floor() -> None:
    """a = 0 → varsayılan; a → +∞ → c·τ_0 tavanı; a → −∞ → τ_alt; tavan aşılmaz, K4 içinde."""
    c = 1.2
    reg = make(tau_ceiling=c)
    t0 = CFG["regulators"]["defaults"]["tau"]
    lo, _ = CFG["regulators"]["bounds"]["tau"]
    assert math.isclose(_tau_at(reg, 0.0), t0, rel_tol=1e-5)
    assert math.isclose(_tau_at(reg, 1.0e3), c * t0, rel_tol=1e-4)
    assert math.isclose(_tau_at(reg, -1.0e3), lo, rel_tol=1e-4)
    for a in torch.linspace(-20, 20, 81).tolist():
        assert lo * (1 - 1e-5) <= _tau_at(reg, a) <= c * t0 * (1 + 1e-5)


def test_tau_asymmetric_is_monotone_and_only_tau() -> None:
    """τ, a'da monoton artar; g, η, β eşlemesi simetrik sürümle aynı kalır."""
    asym, sym = make(tau_ceiling=1.2), make(tau_ceiling=None)
    gen = torch.Generator().manual_seed(3)
    W = torch.randn(len(PARAMS), K, generator=gen)
    asym.W_r, sym.W_r = W.clone(), W.clone()
    z = torch.rand(64, K, generator=gen) * 2 - 1
    oa, os_ = asym(z), sym(z)
    for p in ("g", "eta", "beta"):
        assert torch.allclose(oa[p], os_[p])
    taus = [_tau_at(asym, a) for a in torch.linspace(-3, 3, 31).tolist()]
    assert all(t1 <= t2 + 1e-9 for t1, t2 in zip(taus, taus[1:]))


def test_tau_ceiling_must_exceed_one() -> None:
    with pytest.raises(ValueError):
        make(tau_ceiling=1.0)
