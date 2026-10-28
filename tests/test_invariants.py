"""ARCHITECTURE.md değişmez kurallarının otomatik kontrolü. Her commit öncesi geçmeli.

Yeni modüller eklendikçe (Faz 2+) ilgili kural kontrolleri buraya eklenir.
"""

import math
import re
from pathlib import Path

import pytest
import torch

from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from tests.conftest import CFG, D_IN

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TAUS = [20.0, 100.0, 500.0, 2000.0]


def test_k1_reservoir_has_no_trainable_state() -> None:
    """K1: rezervuarda eğitilebilir parametre ve update metodu yok."""
    res = LIFReservoir.from_config(CFG, D_in=D_IN, seed=0, calibrate=False)
    assert list(res.parameters()) == []
    assert not res.W_in.requires_grad and not res.W_rec.requires_grad
    assert not hasattr(res, "update")
    assert {"W_in", "W_rec"} <= set(res.state_dict())         # checkpoint'e girer


def test_k2_no_autograd_in_src() -> None:
    """K2: kaynak kodda backward / optimizer / autograd kullanımı yok."""
    forbidden = re.compile(r"\.backward\(|torch\.optim|torch\.autograd|requires_grad_\(True\)|requires_grad\s*=\s*True")
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in (ROOT / "src").rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if forbidden.search(line)
    ]
    assert offenders == []


def test_k3_k6_cerebellum_interfaces() -> None:
    """K3: beyincik yalnızca kendi tahmin hatası e ile güncellenir (δ girişi yok).
    K6: forward yalnızca r(t) ve a(t)'yi alır, gelecekteki girdiye erişimi yok."""
    import inspect

    from src.cerebellum import Cerebellum
    assert list(inspect.signature(Cerebellum.update).parameters) == ["self", "e", "eta"]
    assert list(inspect.signature(Cerebellum.forward).parameters) == ["self", "r", "a"]


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
