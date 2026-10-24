"""Faz 1: SpikeEncoder ve LIFReservoir testleri."""

import math

import pytest
import torch

from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir, kwta
from tests.conftest import CFG, D_IN

N_IN = SpikeEncoder.n_channels(CFG, D_IN)
N = CFG["reservoir"]["N"]
T = CFG["time"]["T"]
TAU_DEFAULT = CFG["regulators"]["defaults"]["tau"]
ALLOWED_TAUS = [20.0, 100.0, 500.0, 2000.0]                   # PHASES.md Faz 1


def make_reservoir(seed: int = 0) -> LIFReservoir:
    return LIFReservoir.from_config(CFG, D_in=D_IN, seed=seed, calibrate=False)


# --- SpikeEncoder -----------------------------------------------------------

L = CFG["encoder"]["level"]["n_levels"]


def test_encoder_shape_and_delta_channels() -> None:
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    x0 = torch.zeros(2, D_IN)
    x1 = torch.tensor([[1.0, -1.0, 0.0, 0.05, 0.0]] * 2)
    assert enc(x0)[:, :, : 2 * D_IN].sum() == 0               # ilk adımda delta spike'ı yok
    u = enc(x1)                                                # [B, T, (2+L)·D_in]
    assert u.shape == (2, T, (2 + L) * D_IN) and u.shape[-1] == N_IN
    expected = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=u.dtype)
    assert torch.equal(u[0, 0, : 2 * D_IN], expected)         # 0.05 < θ_enc → spike yok
    assert torch.equal(u[:, 0:1].expand_as(u), u)              # T boyunca sabit


def test_encoder_level_channels_are_thermometer() -> None:
    """Seviye kanalları: x arttıkça yanan kanal sayısı artar; aralık dışında hepsi 0 / hepsi 1."""
    enc = SpikeEncoder.from_config(CFG, D_in=1)
    lo, hi = CFG["encoder"]["level"]["range"]
    xs = torch.linspace(lo - 0.1, hi + 0.1, 50).unsqueeze(1)  # [50, 1]
    counts = []
    for x in xs:
        enc.reset_state()
        level = enc(x.unsqueeze(0))[0, 0, 2:]                  # [L]
        n_on = int(level.sum())
        assert torch.equal(level, (torch.arange(L) < n_on).to(level.dtype))   # soldan dolar
        counts.append(n_on)
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == L
    enc.reset_state()
    mid = enc(torch.tensor([[(lo + hi) / 2]]))[0, 0, 2:]
    assert mid.sum() == L // 2                                 # eşikler eşit aralıklı


def test_encoder_level_disabled() -> None:
    cfg = {**CFG, "encoder": {**CFG["encoder"], "level": {**CFG["encoder"]["level"], "enabled": False}}}
    enc = SpikeEncoder.from_config(cfg, D_in=D_IN)
    assert enc(torch.rand(1, D_IN)).shape[-1] == 2 * D_IN == SpikeEncoder.n_channels(cfg, D_IN)


def test_encoder_reset_state() -> None:
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    enc(torch.zeros(1, D_IN))
    enc.reset_state()
    assert enc(torch.ones(1, D_IN))[:, :, : 2 * D_IN].sum() == 0


# --- k-WTA ------------------------------------------------------------------

def test_kwta_exact_k_when_enough_candidates() -> None:
    cand = torch.randn(4, 1000) + 3.0                          # hepsi eşiğin üstünde
    s = kwta(cand, 50)
    assert torch.all(s.sum(dim=-1) == 50)
    top = cand.topk(50, dim=-1).indices
    assert torch.all(s.gather(-1, top) == 1)


def test_kwta_only_positive_candidates_fire() -> None:
    cand = -torch.ones(1, 100)
    cand[0, :3] = 0.5
    s = kwta(cand, 10)
    assert s.sum() == 3 and torch.all(s[0, :3] == 1)


# --- Başlatma ve K1 -----------------------------------------------------------

def test_same_seed_same_matrices() -> None:
    a, b, c = make_reservoir(0), make_reservoir(0), make_reservoir(1)
    assert torch.equal(a.W_in, b.W_in) and torch.equal(a.W_rec, b.W_rec)
    assert not torch.equal(a.W_in, c.W_in)


def test_init_structure() -> None:
    res = make_reservoir()
    r = CFG["reservoir"]
    assert torch.all((res.W_in != 0).sum(dim=1) == r["c_in"])
    radius = torch.linalg.eigvals(res.W_rec.double()).abs().max().item()
    assert radius == pytest.approx(r["rho_spec"], rel=1e-4)
    assert torch.all(res.W_rec.diagonal() == 0)
    density = (res.W_rec != 0).float().mean().item()
    assert density == pytest.approx(r["p_rec"], rel=0.1)


def test_tau_multipliers() -> None:
    """m_i: geometrik ortalama 1, yayılım ~[1/S, S], seed'e bağlı ve deterministik."""
    S = CFG["reservoir"]["tau_spread"]
    a, b = make_reservoir(0), make_reservoir(0)
    m = a.tau_mult
    assert torch.equal(m, b.tau_mult)
    assert m.log().mean().item() == pytest.approx(0.0, abs=1e-5)
    assert m.min().item() > 1 / S / 1.2 and m.max().item() < S * 1.2
    assert m.max() / m.min() > S                               # gerçekten yayılmış


def test_homogeneous_mode_has_unit_multipliers() -> None:
    cfg = {**CFG, "reservoir": {**CFG["reservoir"], "tau_spread": 1.0}}
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    assert torch.allclose(res.tau_mult, torch.ones(N))


def test_soft_reset_subtracts_theta() -> None:
    """Yumuşak sıfırlama: ateşleyen nöronun V'sinden θ çıkar, fazlası kalır."""
    res = LIFReservoir(n_in=2, N=2, dt=1.0, T=1, rho=0.5, theta_init=1.0, tau_grid=[20.0, 2000.0],
                       c_in=1, p_rec=0.0, rho_spec=0.9, rec_gain=1.0, tau_spread=1.0,
                       reset="subtract", seed=0)
    res.W_in.copy_(torch.tensor([[5.0, 0.0], [0.0, 0.0]]))
    u = torch.tensor([[[1.0, 0.0]]])                           # [B=1, T=1, n_in=2]
    res(u, g=1.0, tau=20.0)                                    # V = 5 → ateşler → 5 − θ = 4
    assert res.last_spike_count.item() == 1
    assert res.V[0, 0].item() == pytest.approx(4.0)
    res.reset = "zero"
    res.reset_state()
    res(u, g=1.0, tau=20.0)
    assert res.V[0, 0].item() == 0.0


def test_matrices_unchanged_after_run(calibrated_reservoir: LIFReservoir) -> None:
    """K1: ağırlıklar bir koşu döngüsünden sonra değişmemiş."""
    res = calibrated_reservoir
    res.reset_state()
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    W_in0, W_rec0 = res.W_in.clone(), res.W_rec.clone()
    for _ in range(30):
        res(enc(torch.rand(2, D_IN)), g=2.0, tau=TAU_DEFAULT)
    assert torch.equal(res.W_in, W_in0) and torch.equal(res.W_rec, W_rec0)


# --- θ(τ) tablosu -------------------------------------------------------------

def test_theta_interpolation() -> None:
    res = make_reservoir()
    res.theta_table.copy_(torch.arange(len(res.theta_tau), dtype=torch.float32))
    grid = res.theta_tau
    assert torch.allclose(res.theta(grid), res.theta_table)    # ızgara noktalarında tam
    mid = (grid[0] * grid[1]).sqrt()                           # log-ölçekte orta nokta
    assert res.theta(mid).item() == pytest.approx(0.5, abs=1e-5)
    assert res.theta(grid[0] / 10).item() == pytest.approx(0.0)       # ızgara altı → uç değer
    assert res.theta(grid[-1] * 10).item() == pytest.approx(len(grid) - 1.0)


def test_calibration_is_deterministic_and_saved(calibrated_reservoir: LIFReservoir) -> None:
    assert bool(calibrated_reservoir.calibrated)
    assert torch.all(calibrated_reservoir.theta_table > 0)
    state = calibrated_reservoir.state_dict()
    assert {"theta_tau", "theta_table", "calibrated"} <= set(state)   # checkpoint'e girer


# --- Dinamik ------------------------------------------------------------------

def test_readout_is_f_and_v(calibrated_reservoir: LIFReservoir) -> None:
    """r = [f, v]: [B, 2N]; f ∈ [0, 1], v son iç adımdaki zar potansiyeli."""
    res = calibrated_reservoir
    res.reset_state()
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    for _ in range(3):
        r = res(enc(torch.rand(2, D_IN)), g=1.0, tau=TAU_DEFAULT)
    assert r.shape == (2, 2 * N)
    f, v = r[:, :N], r[:, N:]
    assert torch.all((f >= 0) & (f <= 1))
    assert torch.allclose(f * T, (f * T).round())             # f = spike sayısı / T
    assert torch.equal(v, res.V)


def test_exactly_k_fire_under_strong_drive(calibrated_reservoir: LIFReservoir) -> None:
    """Güçlü girdide her iç adımda k nöron ateşler; hiçbir τ'da k aşılmaz."""
    res = calibrated_reservoir
    u = torch.ones(1, T, N_IN)
    for tau in ALLOWED_TAUS:
        res.reset_state()
        for _ in range(5):
            res(u, g=1.0, tau=tau)
            assert torch.all(res.last_spike_count <= res.k)
        assert torch.all(res.last_spike_count == res.k)


def test_never_more_than_k_fire(calibrated_reservoir: LIFReservoir) -> None:
    res = calibrated_reservoir
    enc = SpikeEncoder.from_config(CFG, D_in=D_IN)
    for tau in ALLOWED_TAUS:
        res.reset_state()
        enc.reset_state()
        for _ in range(20):
            res(enc(torch.rand(3, D_IN) * 3), g=2.0, tau=tau)
            assert torch.all(res.last_spike_count <= res.k)


@pytest.mark.parametrize("tau", ALLOWED_TAUS)
def test_activity_dies_out_with_zero_input(calibrated_reservoir: LIFReservoir, tau: float) -> None:
    """Kararlılık: girdi kesilince aktivite birkaç ortam adımında söner.

    Yumuşak sıfırlamada birikmiş yük ateşlemeyle boşalır; büyük τ'da bu birkaç adım sürer.
    Sınır: en geç bir zaman sabiti (τ / (T·dt) ortam adımı, en az 10). Kendi kendini
    sürdüren aktivite bu sınırı aşar.
    """
    res = calibrated_reservoir
    res.reset_state()
    u = torch.ones(1, T, N_IN)
    for _ in range(5):
        res(u, g=2.0, tau=tau)
    assert res.last_spike_count.sum() > 0
    zero = torch.zeros_like(u)
    limit = max(10, math.ceil(tau / (T * CFG["time"]["dt"])))
    for _ in range(limit):
        res(zero, g=2.0, tau=tau)
        if res.last_spike_count.sum() == 0:
            break
    assert res.last_spike_count.sum() == 0


def test_state_persists_across_steps_until_reset(calibrated_reservoir: LIFReservoir) -> None:
    """K6: durum dizi içinde sıfırlanmaz; yalnızca reset_state() sıfırlar."""
    res = calibrated_reservoir
    res.reset_state()
    u = torch.ones(1, T, N_IN)
    r_first = res(u, g=1.0, tau=TAU_DEFAULT)
    r_second = res(u, g=1.0, tau=TAU_DEFAULT)
    assert not torch.equal(r_first, r_second)                  # birikmiş V etkili
    res.reset_state()
    assert torch.equal(res(u, g=1.0, tau=TAU_DEFAULT), r_first)


def test_per_sample_regulator_values(calibrated_reservoir: LIFReservoir) -> None:
    """g ve τ batch içinde örnek başına verilebilir."""
    res = calibrated_reservoir
    res.reset_state()
    u = torch.ones(2, T, N_IN)
    r = res(u, g=torch.tensor([1.0, 1.0]), tau=torch.tensor([20.0, 2000.0]))
    assert r.shape == (2, 2 * N)
    assert not torch.equal(r[0], r[1])


def test_wrong_T_raises() -> None:
    res = make_reservoir()
    with pytest.raises(ValueError):
        res(torch.zeros(1, T + 1, N_IN), g=1.0, tau=TAU_DEFAULT)


def test_k_rounding() -> None:
    assert make_reservoir().k == math.ceil(CFG["reservoir"]["rho"] * N)
