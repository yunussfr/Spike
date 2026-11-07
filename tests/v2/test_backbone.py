"""Faz V2: TemporalBackbone, okuma ve akış değerlendirmesi (ARCHITECTURE.md §12.4, §12.7, KV1, KV6, KV7)."""

import copy
from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.v2.backbone import MODES, TemporalBackbone, homeostatic_loss, spike_fn
from src.v2.decision import MultiHeadReadout
from src.v2.model import V2Model, stream_predict

BASE = load_config(Path(__file__).resolve().parents[2] / "configs" / "v2_base.yaml")
C = 6


def small_cfg(**over: object) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["backbone"]["N"] = 20
    cfg["readout"]["taus_ms"] = [16.0, 64.0]
    for k, v in over.items():
        cfg["backbone"][k] = v
    return cfg


def model(mode: str, seed: int = 0, **over: object) -> V2Model:
    torch.manual_seed(seed)
    m = V2Model(small_cfg(**over), C, 2, mode)
    return m


def x_seq(B: int = 3, L: int = 40, seed: int = 1) -> torch.Tensor:
    return torch.randn(B, L, C, generator=torch.Generator().manual_seed(seed)) * 2.0


def test_tau_bounds_per_subpopulation() -> None:
    m = model("plif")
    bb = m.backbone
    with torch.no_grad():
        bb.logit_tau_mem.copy_(torch.linspace(-50, 50, bb.N))
        bb.logit_tau_syn.copy_(torch.linspace(-50, 50, bb.N))
    t, ts = bb.tau_mem(), bb.tau_syn()
    lo_f, hi_f = BASE["backbone"]["tau_mem_fast_ms"]
    lo_s, hi_s = BASE["backbone"]["tau_mem_slow_ms"]
    assert ((t[: bb.half] >= lo_f) & (t[: bb.half] <= hi_f)).all()
    assert ((t[bb.half:] >= lo_s) & (t[bb.half:] <= hi_s)).all()
    lo, hi = BASE["backbone"]["tau_syn_ms"]
    assert ((ts >= lo) & (ts <= hi)).all()


def test_odd_N_rejected() -> None:
    with pytest.raises(ValueError):
        model("plif", N=21)


def test_surrogate_gradient() -> None:
    v = torch.tensor([-1.0, -0.01, 0.0, 0.5], requires_grad=True)
    s = spike_fn(v, 10.0)
    assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    s.sum().backward()
    assert torch.allclose(v.grad, 1.0 / (1.0 + 10.0 * v.detach().abs()) ** 2)


@pytest.mark.parametrize("mode", MODES)
def test_modes_output_and_gradients(mode: str) -> None:
    m = model(mode)
    m.train()
    m.reset_state(3)
    y, rate = m(x_seq())
    assert y.shape == (3, 40, 2) and rate.shape == (20,)
    s = m.backbone.s
    if mode == "tau_gru":
        assert not m.backbone.spiking
    else:
        assert set(s.unique().tolist()) <= {0.0, 1.0}
    (y.pow(2).mean() + homeostatic_loss(rate, 0.02, 0.08)).backward()
    params = dict(m.named_parameters())
    first_in = "backbone.w_in.weight" if mode == "plif" else "backbone.w_h.weight"
    assert params[first_in].grad is not None and params[first_in].grad.abs().sum() > 0
    if mode == "plif":
        assert params["backbone.logit_tau_mem"].grad.abs().sum() > 0


@pytest.mark.parametrize("mode", MODES)
def test_state_not_reset_within_sequence(mode: str) -> None:
    """KV6: iki parça ardışık verilince tek parçayla aynı sonuç (durum taşınır)."""
    m = model(mode)
    m.eval()
    x = x_seq()
    m.reset_state(3)
    y_full, _ = m(x)
    m.reset_state(3)
    y1, _ = m(x[:, :17])
    y2, _ = m(x[:, 17:])
    assert torch.allclose(torch.cat([y1, y2], 1), y_full, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_causal(mode: str) -> None:
    m = model(mode)
    m.eval()
    x = x_seq()
    x2 = x.clone()
    x2[:, 25:] = 5.0
    m.reset_state(3)
    a, _ = m(x)
    m.reset_state(3)
    b, _ = m(x2)
    assert torch.equal(a[:, :25], b[:, :25])


@pytest.mark.parametrize("mode", MODES)
def test_eval_stream_changes_no_parameters(mode: str) -> None:
    """KV1: çevrimiçi akış ve homeostaz hiçbir öğrenilen parametreye dokunmaz; ϑ sınırlar içinde."""
    m = model(mode)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x = x_seq(B=1, L=300)[0]
    stream_predict(m, x, 100, 300, n_streams=2, burnin=50)
    m.reset_state(1)
    with torch.no_grad():
        for t in range(len(x)):
            m.backbone.homeostasis_update(m.backbone.step(x[t: t + 1] * 10))
    after = m.state_dict()
    for k, v in before.items():
        if k in ("backbone.theta", "backbone.rate_ema"):
            continue
        assert torch.equal(v, after[k]), k
    th = after["backbone.theta"]
    h = BASE["backbone"]["homeostasis"]
    assert ((th >= h["theta_min"]) & (th <= h["theta_max"])).all()
    if mode == "tau_gru":
        assert torch.equal(th, before["backbone.theta"])


def test_homeostasis_direction() -> None:
    m = model("plif")
    bb = m.backbone
    th0 = bb.theta.clone()
    for _ in range(50):
        bb.homeostasis_update(torch.ones(4, bb.N))                            # hep ateşliyor → eşik yükselir
    assert (bb.theta > th0).all()
    for _ in range(5000):
        bb.homeostasis_update(torch.zeros(4, bb.N))                           # hiç ateşlemiyor → eşik düşer
    assert (bb.theta < th0).all()


def test_homeostatic_loss_band() -> None:
    assert homeostatic_loss(torch.full((5,), 0.05), 0.02, 0.08) == 0
    assert homeostatic_loss(torch.full((5,), 0.0), 0.02, 0.08) > 0
    assert homeostatic_loss(torch.full((5,), 0.5), 0.02, 0.08) > 0


def test_chrono_init() -> None:
    m = model("spiking_gru")
    bb = m.backbone
    z0 = torch.sigmoid(bb.w_z.bias)
    expected = BASE["time"]["dt_ms"] / bb.tau_mem().detach()
    assert torch.allclose(z0, expected.clamp(max=1 / (1 + BASE["backbone"]["chrono_min"])), atol=1e-5)


def test_readout_heads_and_gate() -> None:
    r = MultiHeadReadout(4, 2, 4.0, [16.0, 64.0])
    r.reset_state(1)
    s = torch.ones(1, 4)
    y, rbar = r.step(s)
    expected = torch.stack([h(r.traces[k]) for k, h in enumerate(r.heads)]).mean(0)
    assert torch.allclose(y, expected)
    r.reset_state(1)
    _, rbar_g = r.step(s, g_fast=2.0, g_slow=0.5)
    assert torch.allclose(rbar_g[0, :2], 2 * rbar[0, :2]) and torch.allclose(rbar_g[0, 2:], 0.5 * rbar[0, 2:])


@pytest.mark.parametrize("mode", MODES)
def test_stream_predict_matches_direct(mode: str) -> None:
    """Her akış, kendi başlangıcından sıfır durumla kesintisiz akıtılan doğrudan tahminle aynı."""
    m = model(mode)
    m.eval()
    x = x_seq(B=1, L=200)[0]                                                 # [n, C]
    start, stop, burnin = 60, 200, 30
    pred, rate = stream_predict(m, x, start, stop, n_streams=3, burnin=burnin)
    assert pred.shape == (stop - start, 2)
    seg = -(-(stop - start) // 3)
    for i in range(3):
        s0 = start + i * seg
        b = max(0, s0 - burnin)
        e = min(s0 + seg, stop)
        m.reset_state(1)
        y, _ = m(x[None, b:e])
        assert torch.allclose(pred[s0 - start: e - start], y[0, s0 - b:], atol=1e-5)
    assert rate.shape == (20,)
