"""Faz 3: Agent (ARCHITECTURE.md §6 adım 1–8) testleri: K1, K3, K5, K6."""

import copy

import pytest
import torch

from src.agent import Agent
from tests.conftest import CFG

D_IN, D_OUT, N_STEPS = 3, 2, 30


def cfg_for(config: str, rule: str = "normalized") -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["reservoir"]["N"] = 100
    cfg["agent"]["config"] = config
    cfg["decision"]["rule"] = rule
    cfg["cerebellum"]["rule"] = rule
    return cfg


def make(config: str, rule: str = "normalized") -> Agent:
    from src.reservoir import LIFReservoir
    cfg = cfg_for(config, rule)
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    return Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0, reservoir=res)


def data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(N_STEPS, 1, D_IN, generator=g), torch.randn(N_STEPS, 1, D_OUT, generator=g)


def run(agent: Agent, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    preds = []
    for t in range(len(x)):
        r = agent.perceive(x[t])
        preds.append(agent.act(r).clone())
        agent.learn(y[t], x[t + 1] if t + 1 < len(x) else None)
    return preds


@pytest.mark.parametrize("config", ["A", "B"])
def test_shapes(config: str) -> None:
    agent = make(config)
    x, _ = data()
    r = agent.perceive(x[0])
    assert r.shape == (1, 2 * agent.reservoir.N)
    assert agent.act(r).shape == (1, D_OUT)
    assert (agent.cerebellum is None) == (config == "A")
    if config == "B":
        assert agent.x_hat.shape == (1, D_IN)


@pytest.mark.parametrize("config", ["A", "B"])
def test_k1_reservoir_untouched(config: str) -> None:
    agent = make(config)
    before = {k: v.clone() for k, v in agent.reservoir.state_dict().items()}
    run(agent, *data())
    for k, v in agent.reservoir.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_k3_cerebellum_does_not_change_decision() -> None:
    """B'deki beyincik karar çıktısını etkilemez: A ve B aynı seed'de birebir aynı ŷ üretir."""
    pa = run(make("A"), *data())
    pb = run(make("B"), *data())
    assert all(torch.equal(a, b) for a, b in zip(pa, pb))


def test_cerebellum_receives_decision_output() -> None:
    """Faz 3: beyincik a(t) = ŷ(t) alır."""
    agent = make("B", rule="plain")
    x, _ = data()
    y_hat = agent.act(agent.perceive(x[0]))
    agent.decision.W_a.normal_()
    y_hat = agent.act(agent.perceive(x[1]))
    assert torch.equal(agent.cerebellum.phi[:, -D_OUT:], y_hat)


def test_learn_updates_only_readouts() -> None:
    agent = make("B")
    x, y = data()
    agent.act(agent.perceive(x[0]))
    Wa, Wc = agent.decision.W_a.clone(), agent.cerebellum.W_c.clone()
    out = agent.learn(y[0], x[1])
    assert not torch.equal(agent.decision.W_a, Wa) and not torch.equal(agent.cerebellum.W_c, Wc)
    assert set(out) == {"delta", "e"}


def test_k6_no_future_leak() -> None:
    """t anındaki ŷ_t ve x̂_{t+1}, x(t+1..) değiştirilince aynı kalır."""
    x1, y = data()
    x2 = x1.clone()
    t_cut = 20
    x2[t_cut + 1:] = 1.0 - x2[t_cut + 1:]
    outs = []
    for x in (x1, x2):
        agent = make("B")
        for t in range(t_cut + 1):
            agent.act(agent.perceive(x[t]))
            if t < t_cut:
                agent.learn(y[t], x[t + 1])
        outs.append((agent.y_hat.clone(), agent.x_hat.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_k5_regulators_must_be_disabled() -> None:
    cfg = cfg_for("A")
    cfg["regulators"]["enabled"] = True
    with pytest.raises(NotImplementedError):
        Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0)


# --- Faz 4: adım 9 (duygu uzayı) -------------------------------------------------

def make_affect(enabled: bool, config: str = "B") -> Agent:
    from src.reservoir import LIFReservoir
    cfg = cfg_for(config)
    cfg["affect"]["enabled"] = enabled
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    return Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0, reservoir=res)


def test_affect_requires_cerebellum() -> None:
    with pytest.raises(ValueError):
        make_affect(True, config="A")


def test_affect_no_leak_into_behaviour() -> None:
    """Faz 4 çıkış kriteri 3: duygu uzayı açıkken ŷ, x̂ ve karar/beyincik ağırlıkları birebir aynı."""
    off, on = make_affect(False), make_affect(True)
    x, y = data()
    for t in range(len(x) - 1):
        for ag in (off, on):
            ag.act(ag.perceive(x[t]))
        assert torch.equal(off.y_hat, on.y_hat) and torch.equal(off.x_hat, on.x_hat)
        out_off, out_on = off.learn(y[t], x[t + 1]), on.learn(y[t], x[t + 1])
        assert torch.equal(out_off["delta"], out_on["delta"]) and torch.equal(out_off["e"], out_on["e"])
    assert torch.equal(off.decision.W_a, on.decision.W_a)
    assert torch.equal(off.cerebellum.W_c, on.cerebellum.W_c)
    assert {"z", "heads", "targets", "head_err"} <= set(out_on)
    assert on.affect.W_h.abs().sum() > 0                                       # duygu uzayı gerçekten öğrendi


def test_affect_frozen_in_eval() -> None:
    ag = make_affect(True)
    x, y = data()
    for t in range(5):
        ag.act(ag.perceive(x[t]))
        ag.learn(y[t], x[t + 1])
    ag.set_eval(True)
    Wz, Wh = ag.affect.W_z.clone(), ag.affect.W_h.clone()
    ag.act(ag.perceive(x[5]))
    out = ag.errors(y[5], x[6])
    assert "z" in out and torch.equal(ag.affect.W_z, Wz) and torch.equal(ag.affect.W_h, Wh)
