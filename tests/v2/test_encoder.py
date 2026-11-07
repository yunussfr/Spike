"""Faz V1: SpikeEncoderV2 (ARCHITECTURE.md §12.3)."""

from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.v2.encoder import SpikeEncoderV2

CFG = load_config(Path(__file__).resolve().parents[2] / "configs" / "v2_base.yaml")


def make(C: int = 4) -> SpikeEncoderV2:
    return SpikeEncoderV2.from_config(CFG, C)


def data(n: int = 2000, C: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rates = torch.tensor([0.02, 0.1, 0.5, 2.0])[:C]
    return torch.poisson(rates.expand(n, C), generator=g)                    # [n, C] sayım


def test_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        make().normalize(torch.zeros(3, 4))


def test_stats_only_from_train() -> None:
    x = data()
    tr, rest = x[:1500], x[1500:]
    e1, e2 = make(), make()
    e1.fit(tr)
    e2.fit(tr)
    rest_changed = torch.cat([tr, rest * 50.0])                               # val/test'i boz
    assert torch.equal(e1.normalize(rest_changed[:1500]), e2.normalize(tr))
    assert torch.equal(e1.mean, e2.mean) and torch.equal(e1.std, e2.std)


def test_train_standardized_and_clipped() -> None:
    x = data()
    e = make()
    e.fit(x)
    z = e.normalize(x)
    assert z.abs().max() < CFG["encoder"]["clip"]
    assert z.mean(0).abs().max() < 0.3                                       # tanh kırpması ortalamayı biraz kaydırır
    huge = e.normalize(torch.full((1, 4), 1e6))
    assert huge.abs().max() <= CFG["encoder"]["clip"]


def test_constant_channel_is_finite() -> None:
    x = data()
    x[:, 0] = 0.0                                                            # hiç ateşlemeyen kanal
    e = make()
    e.fit(x)
    assert torch.isfinite(e.normalize(x)).all()


def test_smooth_causal_and_step_equals_sequence() -> None:
    x = data(n=300)
    e = make()
    e.fit(x)
    xn = e.normalize(x)                                                      # [n, C]
    seq = e.smooth_sequence(xn)
    e.reset_state(1)
    steps = torch.stack([e(x[t: t + 1])[1][0] for t in range(len(x))])
    assert torch.allclose(seq, steps, atol=1e-6)
    xn2 = xn.clone()
    xn2[151:] = 9.0                                                          # gelecek değişir
    assert torch.equal(e.smooth_sequence(xn2)[:151], seq[:151])


def test_batched_forward_matches_single() -> None:
    x = data(n=50)
    e = make()
    e.fit(x)
    xb = torch.stack([x, x.flip(0)], 1)                                      # [n, 2, C]
    e.reset_state(2)
    outs = [e(xb[t])[1] for t in range(len(x))]
    e.reset_state(1)
    single = [e(x[t: t + 1])[1] for t in range(len(x))]
    assert torch.allclose(outs[-1][0], single[-1][0], atol=1e-6)
