"""Faz V2: eğitim döngüsü. CUDA Graph adımı eager adımla aynı; eğitim ve val seçimi test dilimine dokunmuyor."""

import copy
from pathlib import Path

import pytest
import torch

from src.v2.metrics import r2_columns
from src.data.primate import Session
from src.utils.config import load_config
from src.v2.model import V2Model
from src.v2.train import TrainStep, pool, prepare_session, train_model

BASE = load_config(Path(__file__).resolve().parents[2] / "configs" / "v2_base.yaml")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")


def small_cfg(dropout: float = 0.0) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["backbone"]["N"] = 20
    cfg["backbone"]["dropout"] = dropout
    cfg["windows"] = {"length_ms": 80.0, "stride_ms": 40.0}                 # 20 / 10 adım
    cfg["train"].update(batch=4, epochs=2, patience=5, eval_streams=2, eval_burnin_ms=40.0)
    return cfg


def metric(p: torch.Tensor, y: torch.Tensor) -> float:
    return float(sum(r2_columns(p, y)) / y.shape[1])


def fake_session(n: int = 600, C: int = 5) -> Session:
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(n, C, generator=g) < 0.2).float()
    y = torch.cumsum(x[:, :2] - 0.2, 0) * 0.1 + torch.randn(n, 2, generator=g) * 0.01
    return Session("fake_20200101_01", x, y, 400, 500)


def run_steps(cfg: dict, graph: bool, device: torch.device, n: int = 6) -> tuple[list[float], dict]:
    cfg = copy.deepcopy(cfg)
    cfg["train"]["cuda_graph"] = graph
    torch.manual_seed(0)
    m = V2Model(cfg, 5, 2, "plif").to(device)
    m.train()
    step = TrainStep(m, cfg, 10, 4, 20, 5, 2, device)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(n):
        xb = torch.randn(4, 20, 5, generator=g).to(device)
        yb = torch.randn(4, 20, 2, generator=g).to(device)
        losses.append(step(xb, yb))
    return losses, {k: v.detach().cpu() for k, v in m.state_dict().items()}


@cuda
def test_graph_step_matches_eager() -> None:
    dev = torch.device("cuda")
    cfg = small_cfg(dropout=0.0)
    le, se = run_steps(cfg, False, dev)
    lg, sg = run_steps(cfg, True, dev)
    assert torch.allclose(torch.tensor(le), torch.tensor(lg), atol=1e-4)
    for k in se:
        assert torch.allclose(se[k], sg[k], atol=1e-4), k


@cuda
def test_graph_dropout_resamples() -> None:
    dev = torch.device("cuda")
    cfg = small_cfg(dropout=0.3)
    cfg["train"]["cuda_graph"] = True
    cfg["train"]["lr"] = 0.0                                                 # ağırlık değişmesin; fark yalnızca dropout'tan
    torch.manual_seed(0)
    m = V2Model(cfg, 5, 2, "plif").to(dev)
    m.train()
    step = TrainStep(m, cfg, 10, 4, 20, 5, 2, dev)
    xb, yb = torch.randn(4, 20, 5, device=dev), torch.randn(4, 20, 2, device=dev)
    losses = [step(xb, yb) for _ in range(cfg["train"]["graph_warmup"] + 3)]
    replays = losses[-2:]
    assert replays[0] != replays[1]


def test_train_never_reads_test_split() -> None:
    """Test dilimi NaN olsa da eğitim ve val seçimi sonlu (test'e bakılmıyor)."""
    cfg = small_cfg()
    cfg["train"]["cuda_graph"] = False
    s = fake_session()
    s.x[s.test_start:] = float("nan")
    s.y[s.test_start:] = float("nan")
    p = prepare_session(cfg, s, torch.device("cpu"))
    assert torch.isfinite(p.y_mu).all() and int(p.starts[-1]) + 20 <= s.val_start
    torch.manual_seed(0)
    m = V2Model(cfg, 5, 2, "plif")
    res = train_model(m, p, cfg, 0, metric)
    assert res.epochs_run == 2 and torch.isfinite(torch.tensor(res.best_val))
    assert all(torch.isfinite(v).all() for v in res.state.values() if v.is_floating_point())


def test_eval_every_and_multi_session_pool() -> None:
    """Val yalnızca her eval_every epoch'ta ve son epoch'ta; çok oturumda pencereler oturum sınırını aşmaz."""
    cfg = small_cfg()
    cfg["train"].update(cuda_graph=False, epochs=7, eval_every=3, patience=100)
    p1 = prepare_session(cfg, fake_session(), torch.device("cpu"))
    p2 = prepare_session(cfg, fake_session(n=500), torch.device("cpu"))
    x, y, starts = pool([p1, p2])
    assert len(x) == 1100 and len(starts) == len(p1.starts) + len(p2.starts)
    s2 = starts[len(p1.starts):]
    assert int(s2.min()) >= 600 and int(s2.max()) + 20 <= 600 + 400       # ikinci oturumun train dilimi
    assert int(starts[: len(p1.starts)].max()) + 20 <= 400
    m = V2Model(cfg, 5, 2, "plif")
    res = train_model(m, [p1, p2], cfg, 0, metric)
    evald = [r["epoch"] for r in res.history if "val_r2" in r]
    assert evald == [2, 5, 6]
    assert "val_r2_fake_20200101_01" in res.history[2]


def test_save_anchor_and_stats_from() -> None:
    cfg = small_cfg()
    m = V2Model(cfg, 5, 2, "plif")
    assert not bool(m.readout.anchored)
    m.readout.save_anchor()
    with torch.no_grad():
        m.readout.weight[0].add_(1.0)                                  # başlık değişir, çapa değişmez
    assert bool(m.readout.anchored)
    assert torch.allclose(m.readout.w_anchor[0] + 1.0, m.readout.weight[0])
    a, b = fake_session(), fake_session(n=500)
    b.x[: b.val_start] *= 0.0                                                 # farklı istatistik
    pb_own = prepare_session(cfg, b, torch.device("cpu"))
    pb_a = prepare_session(cfg, b, torch.device("cpu"), stats_from=a)
    pa = prepare_session(cfg, a, torch.device("cpu"))
    assert torch.equal(pb_a.y_mu, pa.y_mu) and not torch.equal(pb_own.y_mu, pa.y_mu)
