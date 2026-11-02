"""Faz 5: AffectDynamics (E/M/R) testleri (ARCHITECTURE.md §3.5)."""

import copy

import pytest
import torch

from src.affect import AffectDynamics
from tests.conftest import CFG

K = CFG["affect"]["k"]


def make(**over: float) -> AffectDynamics:
    cfg = copy.deepcopy(CFG)
    cfg["affect"]["dynamics"].update(over)
    return AffectDynamics.from_config(cfg)


def test_shapes_and_bounds() -> None:
    dyn = make()
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        out = dyn(torch.randn(4, K, generator=gen) * 50)                     # aşırı büyük girdi
        for v in out.values():
            assert v.shape == (4, K) and torch.isfinite(v).all()
        assert (out["E"].abs() <= 1).all() and (out["R"].abs() <= 1).all()


def test_constant_input_stays_at_zero() -> None:
    """Sabit z → merkezlenmiş girdi 0 → durum sıfırda kalır (merkezleme doygunluğu engeller)."""
    dyn = make()
    z = torch.full((2, K), 0.8)
    for _ in range(500):
        out = dyn(z)
    for v in out.values():
        assert torch.allclose(v, torch.zeros_like(v))


def test_offset_input_does_not_saturate() -> None:
    """Ortalaması −0,7 olan gürültülü z: merkezlemeyle E büyük ölçüde doygun değil."""
    dyn = make()
    gen = torch.Generator().manual_seed(1)
    E = []
    for _ in range(5000):
        E.append(dyn(-0.7 + 0.2 * torch.randn(1, K, generator=gen))["E"])
    E = torch.cat(E[1000:])
    assert (E.abs() > 0.95).float().mean() < 0.05


def test_smoother_than_input() -> None:
    """Adım başı |ΔE| < |Δz| (titreme azalır)."""
    dyn = make()
    gen = torch.Generator().manual_seed(2)
    z = torch.randn(3000, 1, K, generator=gen) * 0.3
    E = torch.cat([dyn(z[t])["E"] for t in range(len(z))])
    assert (E[1:] - E[:-1]).abs().mean() < 0.5 * (z[1:] - z[:-1]).abs().mean()


def test_threshold_gates_regulator() -> None:
    """|E − M| < θ_R → drive 0, R yalnızca sönümlenir; λ_eff formülü."""
    dyn = make()
    d = CFG["affect"]["dynamics"]
    z = torch.zeros(1, K)
    dyn(z)
    dyn.E = torch.full((1, K), 0.10)
    dyn.M = torch.zeros(1, K)                                                # |E − M| = 0,10 < 0,15
    dyn.R = torch.full((1, K), 0.4)
    R0, M0, E0 = dyn.R.clone(), dyn.M.clone(), dyn.E.clone()
    out = dyn(z)
    assert torch.allclose(out["R"], (1 - d["gamma_decay"]) * R0)
    lam = d["lambda_max"] * (1 - d["kappa"] * M0.abs())
    E_exp = E0 - d["alpha_down"] * (E0 - M0) - lam * out["R"]                # z_in = 0
    assert torch.allclose(out["E"], E_exp)
    assert torch.allclose(out["M"], (1 - d["mood_rate"]) * M0 + d["mood_rate"] * E_exp)


def test_regulator_opposes_deviation() -> None:
    """|E − M| > θ_R → R sapma yönünde büyür ve E'yi geri iter."""
    dyn = make()
    dyn(torch.zeros(1, K))
    dyn.E = torch.full((1, K), 0.6)
    out = dyn(torch.zeros(1, K))
    assert (out["R"] > 0).all()


def test_order_constraints() -> None:
    with pytest.raises(ValueError):
        make(alpha_up=0.05)                                                  # alpha_up < alpha_down
    with pytest.raises(ValueError):
        make(gamma_up=0.5)                                                   # gamma_up > alpha_up


def test_reset_state() -> None:
    dyn = make()
    dyn(torch.randn(1, K))
    dyn.reset_state()
    assert dyn.E is None and dyn.M is None and dyn.R is None and dyn.mu is None
