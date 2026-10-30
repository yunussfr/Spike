"""Batch'li okuma bloklarının tekil Agent ile eşdeğerliği (src/batched.py)."""

import copy

import pytest
import torch

from src.agent import Agent
from src.batched import BatchedAgent
from tests.conftest import CFG

N, D_IN, D_OUT, M = 50, 4, 2, 3
SEEDS = [0, 1, 2]


class _Res:
    """Rezervuar yer tutucusu: ajan yalnızca N okur (r dışarıdan verilir)."""
    N = N


def agents(config: str, affect: bool) -> list[Agent]:
    cfg = copy.deepcopy(CFG)
    cfg["agent"]["config"] = config
    cfg["affect"]["enabled"] = affect
    return [Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=s, reservoir=_Res()) for s in SEEDS]


@pytest.mark.parametrize("config,affect", [("A", False), ("B", False), ("B", True)])
def test_batched_matches_single(config: str, affect: bool) -> None:
    g = torch.Generator().manual_seed(0)
    T = 120
    r = torch.randn(M, T, 2 * N, generator=g) * 20 + 3                       # her ajanın kendi r'si
    x = torch.rand(T, D_IN, generator=g)
    y = torch.randn(T, D_OUT, generator=g)
    singles = agents(config, affect)
    batched = BatchedAgent.from_agents(agents(config, affect), torch.device("cpu"))
    for phase, steps in (("train", range(0, 90)), ("eval", range(90, T - 1))):
        if phase == "eval":
            for a in singles:
                a.set_eval(True)
            batched.set_eval(True)
        for t in steps:
            yb = batched.act(r[:, t])
            ys = torch.cat([a.act(r[m, t: t + 1]) for m, a in enumerate(singles)])
            assert torch.allclose(yb, ys, rtol=1e-4, atol=1e-5), (phase, t)
            fn = "learn" if phase == "train" else "errors"
            ob = getattr(batched, fn)(y[t].expand(M, -1), x[t + 1].expand(M, -1))
            os_ = [getattr(a, fn)(y[t: t + 1], x[t + 1: t + 2]) for a in singles]
            if config == "B":
                assert torch.allclose(batched.x_hat, torch.cat([a.x_hat for a in singles]), rtol=1e-4, atol=1e-5)
            if affect:
                assert torch.allclose(ob["z"], torch.cat([o["z"] for o in os_]), rtol=1e-4, atol=1e-5)
                assert torch.allclose(ob["head_err"], torch.cat([o["head_err"] for o in os_]), rtol=1e-3, atol=1e-4)
    assert torch.allclose(batched.W_a, torch.stack([a.decision.W_a for a in singles]), rtol=1e-4, atol=1e-6)
