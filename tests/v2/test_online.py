"""Faz V4: çevrimiçi çapalı NLMS akışı (KV1, KV2, KV6, §12.7)."""

import copy
from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.v2.model import V2Model
from src.v2.online import stream_adapt

BASE = load_config(Path(__file__).resolve().parents[2] / "configs" / "v2_base.yaml")
C, D = 5, 2


def make_model(seed: int = 0) -> V2Model:
    cfg = copy.deepcopy(BASE)
    cfg["backbone"]["N"] = 20
    cfg["readout"]["taus_ms"] = [16.0, 64.0]
    torch.manual_seed(seed)
    m = V2Model(cfg, C, D, "plif")
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
    m.readout.save_anchor()
    m.eval()
    return m


def data(n: int = 200, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, C, generator=g) * 2.0, torch.randn(n, D, generator=g)


def run(m: V2Model, x: torch.Tensor, y: torch.Tensor, etas: list[float], alpha: float = 1e-4,
        homeo: bool = False, start: int = 80, stop: int = 200, burnin: int = 30, delay: int = 0):
    return stream_adapt(m, x, y, start, stop, burnin, etas, alpha, 1e-4, homeo, label_delay=delay)


def test_requires_anchor() -> None:
    m = make_model()
    m.readout.anchored.fill_(False)
    x, y = data()
    with pytest.raises(RuntimeError):
        run(m, x, y, [0.0])


def test_eta_zero_equals_frozen_stream() -> None:
    m = make_model()
    x, y = data()
    res = run(m, x, y, [0.0], alpha=0.0)
    m.reset_state(1)
    direct, _ = m(x[None, 50:200])                                           # begin = 80 − 30
    assert torch.allclose(res.pred[0], direct[0, 30:], atol=1e-5)
    assert float(res.dw_rel[0]) == 0.0


@pytest.mark.parametrize("delay", [0, 1, 7])
def test_zero_leakage(delay: int) -> None:
    """KV6 + etiket gecikmesi d: y_t değişince v̂_{t+d} ve öncesi aynı; ilk değişen v̂_{t+d+1}."""
    m = make_model()
    x, y = data()
    a = run(m, x, y, [0.05], delay=delay)
    y2 = y.clone()
    y2[120] += 10.0
    b = run(m, x, y2, [0.05], delay=delay)
    i = 120 - 80 + delay
    assert torch.equal(a.pred[0, : i + 1], b.pred[0, : i + 1])
    assert not torch.allclose(a.pred[0, i + 1], b.pred[0, i + 1])


def test_delayed_labels_from_burnin_unused() -> None:
    """Burn-in adımlarının etiketleri d adım sonra da kullanılmaz (maske tamponda taşınır)."""
    m = make_model()
    x, y = data()
    a = run(m, x, y, [0.05], delay=10)
    y2 = y.clone()
    y2[50:80] += 50.0                                                        # burn-in bölgesi (begin = 50)
    b = run(m, x, y2, [0.05], delay=10)
    assert torch.equal(a.pred, b.pred)


def test_batched_etas_match_separate() -> None:
    m = make_model()
    x, y = data()
    both = run(m, x, y, [0.0, 0.02, 0.1])
    for j, eta in enumerate([0.0, 0.02, 0.1]):
        one = run(m, x, y, [eta])
        assert torch.allclose(both.pred[j], one.pred[0], atol=1e-5)


def test_model_unchanged_and_bounded() -> None:
    """KV1/KV2: akış omurgaya ve kayıtlı başlıklara dokunmaz (homeostaz açık olsa da); büyük hatada W sonlu."""
    m = make_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x, y = data()
    res = run(m, x * 20, y * 1e4, [0.5], alpha=0.01, homeo=True)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert torch.isfinite(res.pred).all() and torch.isfinite(res.dw_rel).all()
    assert not torch.equal(res.theta, before["backbone.theta"])              # homeostaz akış içinde çalıştı


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
@pytest.mark.parametrize("homeo", [False, True])
@pytest.mark.parametrize("delay", [0, 16])
def test_graph_chunks_match_eager(homeo: bool, delay: int) -> None:
    dev = torch.device("cuda")
    x, y = data(n=300)
    x, y = x.to(dev) * 3, y.to(dev)
    m = make_model().to(dev)
    eager = stream_adapt(m, x, y, 100, 300, 40, [0.0, 0.05], 1e-3, 1e-4, homeo, chunk=0, label_delay=delay)
    graph = stream_adapt(m, x, y, 100, 300, 40, [0.0, 0.05], 1e-3, 1e-4, homeo, chunk=64, label_delay=delay)   # 240 adım: eksik son parça
    assert torch.allclose(eager.pred, graph.pred, atol=1e-4)
    assert torch.allclose(eager.dw_rel, graph.dw_rel, atol=1e-5)
    assert torch.allclose(eager.rate, graph.rate, atol=1e-6)
    assert torch.allclose(eager.theta, graph.theta, atol=1e-6)


def test_adaptation_reduces_error_on_shift() -> None:
    """Hedefe sabit bir kayma eklenince uyum açık koşulun hatası donuk koşuldan düşük."""
    m = make_model()
    x, _ = data(n=600)
    m.reset_state(1)
    base, _ = m(x[None])
    y = base[0].detach() * -1.0                                              # okuma tersine dönmüş: güçlü kayma
    res = run(m, x, y, [0.0, 0.1], start=100, stop=600)
    err = ((res.pred - y[100:600].unsqueeze(0)) ** 2)[:, 250:].mean((1, 2))
    assert err[1] < 0.5 * err[0]
