"""ARCHITECTURE.md değişmez kurallarının otomatik kontrolü. Her commit öncesi geçmeli.

Yeni modüller eklendikçe (Faz 2+) ilgili kural kontrolleri buraya eklenir.
"""

import math

import pytest
import torch

from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from tests.conftest import CFG, D_IN

ALLOWED_TAUS = [20.0, 100.0, 500.0, 2000.0]


def test_k1_reservoir_has_no_trainable_state() -> None:
    """K1: rezervuarda eğitilebilir parametre ve update metodu yok."""
    res = LIFReservoir.from_config(CFG, D_in=D_IN, seed=0, calibrate=False)
    assert list(res.parameters()) == []
    assert not res.W_in.requires_grad and not res.W_rec.requires_grad
    assert not hasattr(res, "update")
    assert {"W_in", "W_rec"} <= set(res.state_dict())         # checkpoint'e girer


def test_k3_k6_cerebellum_interfaces() -> None:
    """K3: beyincik yalnızca kendi tahmin hatası e ile güncellenir (δ girişi yok).
    K6: forward yalnızca r(t) ve a(t)'yi alır, gelecekteki girdiye erişimi yok."""
    import inspect

    from src.cerebellum import Cerebellum
    assert list(inspect.signature(Cerebellum.update).parameters) == ["self", "e", "eta"]
    assert list(inspect.signature(Cerebellum.forward).parameters) == ["self", "r", "a"]


def test_k3_decision_interface() -> None:
    """K3: karar bloğu yalnızca sonuç hatası δ ile güncellenir (e girişi yok)."""
    import inspect

    from src.decision import DecisionReadout
    assert list(inspect.signature(DecisionReadout.update).parameters) == ["self", "delta", "eta"]


def test_k3_affect_interface() -> None:
    """K3: duygu uzayı δ ve e'yi doğrudan almaz; yalnızca onlardan türetilen q, u, s ile güncellenir."""
    import inspect

    from src.affect import AffectSpace
    assert list(inspect.signature(AffectSpace.update).parameters) == ["self", "q", "u", "s"]


def test_k4_tau_grid_covers_allowed_range() -> None:
    """θ(τ) tablosu regülatörün izin verdiği τ aralığının tamamını kapsar."""
    lo, hi = CFG["regulators"]["bounds"]["tau"]
    grid = CFG["reservoir"]["calibration"]["tau_grid"]
    assert min(grid) <= lo and max(grid) >= hi


@pytest.mark.parametrize("tau", ALLOWED_TAUS)
def test_k7_firing_rate_within_bounds(calibrated_reservoir: LIFReservoir, tau: float) -> None:
    """K7: kalibrasyonda görülmemiş rastgele girdide her τ için oran ∈ [ρ/2, 2ρ]."""
    rho = CFG["reservoir"]["rho"]
    res = calibrated_reservoir
    res.reset_state()
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    gen = torch.Generator().manual_seed(12345)
    washout = math.ceil(3 * tau / (CFG["time"]["T"] * CFG["time"]["dt"]))
    rates = []
    for t in range(washout + 100):
        r = res(enc(torch.rand(1, D_IN, generator=gen)), g=CFG["regulators"]["defaults"]["g"], tau=tau)
        if t >= washout:
            rates.append(r[:, : res.N].mean().item())
    rate = sum(rates) / len(rates)
    assert rho / 2 <= rate <= 2 * rho


def test_k4_regulators_within_bounds() -> None:
    """K4: regülatör çıktıları, W_r ne olursa olsun, sınırların içinde; yalnızca dört parametre."""
    import copy
    from src.affect import PARAMS, Regulators
    cfg = copy.deepcopy(CFG)
    cfg["regulators"]["enabled"] = True
    reg = Regulators.from_config(cfg, seed=0)
    gen = torch.Generator().manual_seed(0)
    reg.W_r = torch.randn(reg.W_r.shape, generator=gen) * 100
    out = reg(torch.rand(512, cfg["affect"]["k"], generator=gen) * 2 - 1)
    assert set(out) == set(PARAMS) == {"g", "tau", "eta", "beta"}
    for p, v in out.items():
        lo, hi = cfg["regulators"]["bounds"][p]
        assert (v >= lo * (1 - 1e-5)).all() and (v <= hi * (1 + 1e-5)).all(), p


def test_k5_regulators_disabled_constant() -> None:
    """K5: regulators.enabled: false → dört değer varsayılanda sabit."""
    import copy
    from src.affect import Regulators
    cfg = copy.deepcopy(CFG)
    cfg["regulators"]["enabled"] = False
    reg = Regulators.from_config(cfg, seed=0)
    reg.W_r.fill_(10.0)
    out = reg(torch.randn(8, cfg["affect"]["k"]))
    for p, v in out.items():
        assert torch.equal(v, torch.full((8,), float(cfg["regulators"]["defaults"][p])))
