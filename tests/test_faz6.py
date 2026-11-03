"""Faz 6 testleri (ARCHITECTURE.md §3.6, §8, KR-031): hızlı-yavaş okuma, prequential akış, η_t kapısı, çapa."""

import copy

import pytest
import torch

from src.agent import Agent
from src.decision import DecisionReadout
from src.reservoir import LIFReservoir
from tests.test_agent import D_IN, D_OUT, N_STEPS, cfg_for, data, run

N_TRAIN = 15


def make_agent(config: str = "B+Z", **fast) -> Agent:
    """config: "B+Z" ya da "C" (E/M/R açık, perturbasyon). fast: decision.fast üzerine yazılır."""
    cfg = cfg_for("B" if config == "B+Z" else "C")
    cfg["affect"]["enabled"] = True
    if config == "C":
        cfg["affect"]["dynamics"]["enabled"] = True
        cfg["regulators"].update(enabled=True, mode="perturb")
        cfg["regulators"]["perturbation"].update(M=4, noise_std=1.0)
    cfg["decision"]["fast"].update(fast)
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    return Agent(cfg, D_in=D_IN, D_out=D_OUT, seed=0, reservoir=res)


def prequential(ag: Agent, x: torch.Tensor, y: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Değerlendirme akışı: önce ŷ_t skorlanır, sonra (açıksa) ΔW güncellenir. → ŷ listesi, τ listesi."""
    ag.set_eval(True)
    preds, taus = [], []
    for t in range(N_TRAIN, len(x)):
        preds.append(ag.act(ag.perceive(x[t])).clone())
        taus.append(torch.as_tensor(ag.tau).clone())
        if t + 1 < len(x):
            ag.errors(y[t], x[t + 1])
        if ag.decision.dW is not None:
            ag.adapt(y[t])
    return preds, taus


def trained(config: str = "B+Z", **fast) -> tuple[Agent, torch.Tensor, torch.Tensor]:
    x, y = data()
    ag = make_agent(config, **fast)
    run(ag, x[:N_TRAIN], y[:N_TRAIN])
    return ag, x, y


# --- 1. Nedensellik: y_t, ŷ_t'yi etkileyemez ---------------------------------------------

@pytest.mark.parametrize("delay", [1, 2])
def test_zero_leakage(delay: int) -> None:
    """y_t değişince ŷ_t (ve d − 1 adım sonrası) birebir aynı; ilk etki ŷ_{t+d}'de."""
    ag, x, y = trained(enabled=True, eta_base=0.1, delay=delay)
    t0 = 22
    y2 = y.clone()
    y2[t0] = y2[t0] + 10.0                                                     # yalnızca y_t0 farklı
    p1, _ = prequential(copy.deepcopy(ag), x, y)
    p2, _ = prequential(copy.deepcopy(ag), x, y2)
    k = t0 - N_TRAIN
    for a, b in zip(p1[: k + delay], p2[: k + delay]):                         # ŷ_≤t0+d−1 aynı
        assert torch.equal(a, b)
    assert not torch.equal(p1[k + delay], p2[k + delay])                        # ŷ_{t0+d} değişir


def test_delay_must_be_causal() -> None:
    d = DecisionReadout(4, 2, "regression", "normalized", 0.01, 1e-6, "auto")
    with pytest.raises(ValueError):
        d.start_fast(lambda_pull=0.0, rho_max=0.25, delay=0)


# --- 2. Norm sınırı ve elastik çapa ----------------------------------------------------------

@pytest.mark.parametrize("rule", ["normalized", "plain"])
def test_norm_bound_and_pull_back(rule: str) -> None:
    """Aşırı girdi/hata altında ‖ΔW‖_F ≤ ρ‖W_base‖_F; girdi kesilince ΔW → 0 (W_base'e döner)."""
    torch.manual_seed(0)
    F, D, rho = 20, 2, 0.25
    d = DecisionReadout(F, D, "regression", rule, 0.01, 1e-6, "auto")
    d.W_a = torch.randn(D, d.n_in)                                             # [D, n_in] W_base
    W_base = d.W_a.clone()
    d.start_fast(lambda_pull=0.02, rho_max=rho, delay=1)
    limit = rho * float(W_base.norm())
    for _ in range(300):
        d(torch.randn(4, F) * 1e6)                                             # aşırı girdi
        d.update_fast(torch.randn(4, D) * 1e8, eta=0.1)                        # aşırı hata
        assert torch.isfinite(d.dW).all()
        assert float(d.dW.norm()) <= limit * (1 + 1e-5)
    assert float(d.dW.norm()) > 0.5 * limit                                    # sınıra dayanmış
    for _ in range(1000):
        d.update_fast(None, eta=0.1)                                           # girdi/etiket yok
    assert float(d.dW.norm()) < 1e-6 * limit                                   # (1 − λ)^1000 ≈ 2e-9
    assert torch.equal(d.W_a, W_base)                                          # W_base hiç değişmedi


def test_w_base_and_other_blocks_frozen_during_adaptation() -> None:
    ag, x, y = trained("C", enabled=True, eta_mode="modulated", eta_base=0.1)
    ag.set_eval(True)
    frozen = [ag.decision.W_a, ag.cerebellum.W_c, ag.affect.W_z, ag.affect.W_h, ag.regulators.W_r]
    before = [w.clone() for w in frozen]
    _, _ = prequential(ag, x, y)
    for a, b in zip(before, (ag.decision.W_a, ag.cerebellum.W_c, ag.affect.W_z, ag.affect.W_h,
                             ag.regulators.W_r)):
        assert torch.equal(a, b)
    assert float(ag.decision.dW.norm()) > 0                                    # yalnızca ΔW öğrendi


# --- 3. η = 0 → Faz 5 ile bit düzeyinde aynı ------------------------------------------------

@pytest.mark.parametrize("config", ["B+Z", "C"])
@pytest.mark.parametrize("mode", ["fixed", "modulated"])
def test_eta_zero_equals_faz5(config: str, mode: str) -> None:
    """η_t = 0: tahminler ve τ izi donuk Faz 5 değerlendirmesiyle birebir aynı (K7 oranı dahil)."""
    ag, x, y = trained(config)
    f5 = copy.deepcopy(ag)
    f6 = copy.deepcopy(ag)
    f6.fast.update(enabled=True, eta_mode=mode, eta_base=0.0, eta_mod=0.0)
    p5, t5 = prequential(f5, x, y)
    p6, t6 = prequential(f6, x, y)
    assert f5.decision.dW is None and f6.decision.dW is not None
    for a, b in zip(p5 + t5, p6 + t6):
        assert torch.equal(a, b)
    assert torch.equal(f5.reservoir.last_spike_count, f6.reservoir.last_spike_count)


def test_adaptation_does_not_change_tau_trace() -> None:
    """Değerlendirmede ΔW, τ izini etkilemez (duygu uzayı donuk, z yalnızca r'yi okur): fork protokolünün dayanağı."""
    ag, x, y = trained("C")
    a, b = copy.deepcopy(ag), copy.deepcopy(ag)
    b.fast.update(enabled=True, eta_mode="modulated", eta_base=0.1)
    pa, ta = prequential(a, x, y)
    pb, tb = prequential(b, x, y)
    assert all(torch.equal(u, v) for u, v in zip(ta, tb))
    assert not torch.equal(pa[-1], pb[-1])


# --- η_t kapısı ---------------------------------------------------------------------------

def test_eta_gate() -> None:
    ag = make_agent(eta_mode="modulated", eta_base=1e-3, eta_mod=9e-3, gate_scale=2.0)
    ag.tau = 100.0
    assert ag.fast_eta() == pytest.approx(1e-3)                                # τ = τ_0 → kapı kapalı
    ag.tau = torch.tensor([50.0])
    assert ag.fast_eta() == pytest.approx(1e-2)                                # τ yarıya → tam açık
    ag.tau = torch.tensor([400.0])
    assert ag.fast_eta() == pytest.approx(1e-2)                                # kapı 1'de doyar
    ag.tau = torch.tensor([100.0 * 2 ** 0.5])
    assert ag.fast_eta() == pytest.approx(1e-3 + 0.5 * 9e-3)
    ag.fast.update(eta_mod=1.0)
    assert ag.fast_eta() <= ag.cfg["regulators"]["bounds"]["eta"][1]           # K4 üst sınırı
    ag.fast["eta_mode"] = "fixed"
    assert ag.fast_eta() == pytest.approx(1e-3)


# --- Çapa alt-popülasyonu -------------------------------------------------------------------

def _res(frac: float) -> LIFReservoir:
    cfg = cfg_for("B")
    cfg["reservoir"]["anchor_frac"] = frac
    return LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)


def test_anchor_mask_and_k1() -> None:
    r0, r2, r1 = _res(0.0), _res(0.2), _res(1.0)
    assert int(r0.anchor.sum()) == 0 and int(r2.anchor.sum()) == 20 and bool(r1.anchor.all())
    for name in ("W_in", "W_rec", "tau_mult"):                                 # K1: dizilim değişmez
        assert torch.equal(getattr(r0, name), getattr(r2, name))


def test_anchor_neurons_ignore_tau() -> None:
    """anchor_frac = 1: her τ'da çıktı τ_ref'teki ile aynı; anchor_frac = 0: τ çıktıyı değiştirir."""
    x, _ = data()
    from src.encoder import SpikeEncoder
    enc = SpikeEncoder.from_config(cfg_for("B"), D_in=D_IN)
    u = [enc(x[t]) for t in range(10)]

    def roll(res: LIFReservoir, tau: float) -> torch.Tensor:
        res.reset_state()
        return torch.stack([res(ut, g=1.0, tau=tau) for ut in u])

    r1 = _res(1.0)
    assert torch.equal(roll(r1, 40.0), roll(r1, r1.tau_ref))
    r0 = _res(0.0)
    assert not torch.equal(roll(r0, 40.0), roll(r0, r0.tau_ref))
