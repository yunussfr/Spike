"""Agent (ARCHITECTURE.md §6) testleri: adım 1–8 (Faz 3), 9 (Faz 4), 2 ve 10 (Faz 5); K1, K3–K6."""

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


def test_regulators_only_in_c() -> None:
    """Regülatörler yalnızca C'de; C duygu uzayı ister."""
    cfg = cfg_for("B")
    cfg["regulators"]["enabled"] = True
    with pytest.raises(ValueError):
        Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0)
    with pytest.raises(ValueError):
        Agent(cfg_for("C"), D_in=D_IN, D_out=D_OUT, seed=0)                  # affect.enabled: false


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


# --- Faz 5: adım 2 ve 10 (regülatörler, konfigürasyon C) ----------------------------

def make_c(enabled: bool, mode: str = "perturb", M: int = 4, noise_std: float = 1.0,
           dynamics: bool = False, paired: bool = True) -> Agent:
    from src.reservoir import LIFReservoir
    cfg = cfg_for("C")
    cfg["affect"]["enabled"] = True
    cfg["affect"]["dynamics"]["enabled"] = dynamics
    cfg["regulators"].update(enabled=enabled, mode=mode, hand_gain=5.0)
    cfg["regulators"]["perturbation"].update(M=M, noise_std=noise_std, paired=paired)
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    return Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0, reservoir=res)


@pytest.mark.parametrize("dynamics", [False, True])
def test_k5_c_disabled_equals_b_with_affect(dynamics: bool) -> None:
    """K5: C + regulators.enabled: false, B + duygu uzayı ile birebir aynı (E/M/R açık olsa da)."""
    b, c = make_affect(True), make_c(False, dynamics=dynamics)
    x, y = data()
    for t in range(len(x) - 1):
        for ag in (b, c):
            ag.act(ag.perceive(x[t]))
            ag.learn(y[t], x[t + 1])
        assert torch.equal(b.y_hat, c.y_hat) and torch.equal(b.x_hat, c.x_hat)
    d = CFG["regulators"]["defaults"]
    assert (c.g, c.tau, c.eta, c.beta) == (d["g"], d["tau"], d["eta"], d["beta"])


@pytest.mark.parametrize("mode,dynamics", [("perturb", False), ("hand", False), ("perturb", True)])
def test_c_regulators_change_and_stay_in_bounds(mode: str, dynamics: bool) -> None:
    """C açıkken g/τ/η değişir (en azından biri), değerler hep sınırlarda (K4), W_in/W_rec sabit (K1)."""
    ag = make_c(True, mode=mode, dynamics=dynamics)
    W_in, W_rec = ag.reservoir.W_in.clone(), ag.reservoir.W_rec.clone()
    x, y = data()
    seen = {p: [] for p in ("g", "tau", "eta", "beta")}
    trials = 0
    for t in range(len(x) - 1):
        ag.act(ag.perceive(x[t]))
        for p in seen:
            seen[p].append(torch.as_tensor(getattr(ag, p)).flatten())
        trials += "trial" in ag.learn(y[t], x[t + 1])
    for p, vals in seen.items():
        v = torch.cat(vals)
        lo, hi = CFG["regulators"]["bounds"][p]
        assert (v >= lo * (1 - 1e-5)).all() and (v <= hi * (1 + 1e-5)).all(), p
    assert torch.cat(seen["tau"]).unique().numel() > 1                       # τ sabit değil
    assert torch.equal(ag.reservoir.W_in, W_in) and torch.equal(ag.reservoir.W_rec, W_rec)
    if mode == "perturb":
        assert trials > 0 and ag.regulators.n_trials == trials


def test_c_first_step_uses_defaults_and_reset() -> None:
    """İlk adımda ve reset_state sonrası z_{t−1} yok → varsayılanlar (§6 adım 2, K6)."""
    ag = make_c(True, noise_std=5.0)
    ag.regulators.W_r.fill_(3.0)                                             # z gelince varsayılandan uzaklaşsın
    x, y = data()
    d = CFG["regulators"]["defaults"]
    ag.act(ag.perceive(x[0]))
    assert float(ag.tau) == pytest.approx(d["tau"], rel=1e-5)
    ag.learn(y[0], x[1])
    ag.act(ag.perceive(x[1]))
    assert float(ag.tau) != pytest.approx(d["tau"], rel=1e-3)
    ag.reset_state()
    ag.act(ag.perceive(x[2]))
    assert float(ag.tau) == pytest.approx(d["tau"], rel=1e-5)


def test_c_eval_freezes_regulators() -> None:
    ag = make_c(True, M=2)
    x, y = data()
    for t in range(3):                                                       # deneme yarıda kalır
        ag.act(ag.perceive(x[t]))
        ag.learn(y[t], x[t + 1])
    assert ag.regulators.phase == "trial"
    ag.set_eval(True)
    W = ag.regulators.W_r.clone()
    assert ag.regulators.phase == "ref"
    for t in range(3, 10):
        ag.act(ag.perceive(x[t]))
        ag.errors(y[t], x[t + 1])
    assert torch.equal(ag.regulators.W_r, W)


def test_batched_rejects_regulators() -> None:
    from src.batched import BatchedAgent
    with pytest.raises(NotImplementedError):
        BatchedAgent([make_c(True)], device="cpu")


def test_c_dynamics_feeds_regulators() -> None:
    """E/M/R açıkken regülatör girdisi E'dir; dönüşte E, M, R vardır; reset durumu siler."""
    ag = make_c(True, dynamics=True)
    x, y = data()
    for t in range(5):
        ag.act(ag.perceive(x[t]))
        out = ag.learn(y[t], x[t + 1])
    assert {"E", "M", "R"} <= set(out)
    assert torch.equal(ag.z_prev, out["E"]) and not torch.equal(out["E"], out["z"])
    ag.reset_state()
    assert ag.dynamics.E is None and ag.z_prev is None


# --- eşli perturbasyon (gölge ajan, §3.5) -------------------------------------------

def _capture_q(ag: Agent) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []
    orig = ag.regulators.update

    def wrapped(q: torch.Tensor):
        seen.append(q.clone())
        return orig(q)
    ag.regulators.update = wrapped
    return seen


def test_paired_signal_zero_while_c_equals_shadow() -> None:
    """W_r = 0 iken C varsayılan değerlerle çalışır = gölge; ilk referans penceresinde Δq tam 0."""
    M = 6
    ag = make_c(True, M=M, paired=True)
    assert ag.shadow is not None and ag.shadow.config == "A" and ag.shadow.regulators is None
    seen = _capture_q(ag)
    x, y = data()
    for t in range(M):                                                       # referans penceresi
        ag.act(ag.perceive(x[t]))
        ag.learn(y[t], x[t + 1])
    assert len(seen) == M and all(torch.equal(q, torch.zeros_like(q)) for q in seen)
    assert torch.equal(ag.shadow.y_hat, ag.y_hat)


def test_paired_shadow_is_isolated() -> None:
    """Gölge C'nin rezervuarını paylaşmaz (K1 kopya) ve C'nin bloklarına dokunmaz."""
    ag = make_c(True, paired=True)
    assert ag.shadow.reservoir is not ag.reservoir
    assert torch.equal(ag.shadow.reservoir.W_in, ag.reservoir.W_in)
    x, y = data()
    for t in range(10):
        ag.act(ag.perceive(x[t]))
        ag.learn(y[t], x[t + 1])
    assert ag.shadow.decision.W_a is not ag.decision.W_a
    ag.set_eval(True)
    assert ag.shadow.decision.norm.frozen == ag.decision.norm.frozen
    ag.reset_state()
    assert ag.shadow.y_hat is None and ag._q_shadow is None


def test_unpaired_uses_raw_q() -> None:
    ag = make_c(True, paired=False)
    assert ag.shadow is None
    seen = _capture_q(ag)
    x, y = data()
    ag.act(ag.perceive(x[0]))
    out = ag.learn(y[0], x[1])
    assert torch.equal(seen[0], out["targets"]["q"])


def test_no_shadow_without_perturbation() -> None:
    assert make_c(False).shadow is None                                      # K5
    assert make_c(True, mode="hand").shadow is None


# --- Değerlendirmede nedensel homeostatik normalizasyon (§8, KR-030) ---------------------

def _train_then_eval(freeze: bool, x: torch.Tensor, y: torch.Tensor, n_train: int = 15) -> tuple[Agent, list[torch.Tensor]]:
    ag = make_affect(True)
    ag.freeze_norm_eval = freeze
    run(ag, x[:n_train], y[:n_train])
    ag.set_eval(True)
    preds = []
    for t in range(n_train, len(x)):
        preds.append(ag.act(ag.perceive(x[t])).clone())
        if t + 1 < len(x):
            ag.errors(y[t], x[t + 1])
    return ag, preds


@pytest.mark.parametrize("freeze", [True, False])
def test_eval_norm_weights_frozen(freeze: bool) -> None:
    """Değerlendirmede ağırlıklar her iki ayarda donuk; norm istatistikleri yalnızca freeze false iken değişir."""
    x, y = data()
    ag = make_affect(True)
    ag.freeze_norm_eval = freeze
    run(ag, x[:15], y[:15])
    ag.set_eval(True)
    W = (ag.decision.W_a.clone(), ag.cerebellum.W_c.clone(), ag.affect.W_z.clone(), ag.affect.W_h.clone())
    mu = (ag.decision.norm.mu.clone(), ag.cerebellum.norm.mu.clone(), ag.affect.norm.mu.clone())
    t_mu = ag.affect.t_mu.clone()
    for t in range(15, N_STEPS - 1):
        ag.act(ag.perceive(x[t]))
        ag.errors(y[t], x[t + 1])
    for a, b in zip(W, (ag.decision.W_a, ag.cerebellum.W_c, ag.affect.W_z, ag.affect.W_h)):
        assert torch.equal(a, b)
    assert torch.equal(t_mu, ag.affect.t_mu)                                   # hedef istatistikleri donuk
    moved = [not torch.equal(a, b) for a, b in zip(mu, (ag.decision.norm.mu, ag.cerebellum.norm.mu,
                                                        ag.affect.norm.mu))]
    assert all(moved) if not freeze else not any(moved)


def test_eval_norm_is_causal() -> None:
    """freeze false: değerlendirmede t anındaki ŷ yalnızca x_1..x_t'ye bağlı (gelecek değişse de aynı)."""
    x, y = data()
    x2 = x.clone()
    x2[22:] = torch.rand_like(x2[22:])                                         # gelecek farklı
    _, p1 = _train_then_eval(False, x, y)
    _, p2 = _train_then_eval(False, x2, y)
    k = 22 - 15
    for a, b in zip(p1[:k], p2[:k]):                                           # t < 22: geçmiş aynı
        assert torch.equal(a, b)
    assert not torch.equal(p1[k], p2[k])                                       # x_22 değişti → ŷ_22 değişir
