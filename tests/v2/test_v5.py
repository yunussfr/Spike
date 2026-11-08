"""Faz V5: beyincik, duygu uzayı, duygu-kapılı W_in plastisitesi / yeniden ölçekleme, kalıcı sürpriz
(KV1–KV6, §12.5, §12.6, §12.13, §12.14; KR-044–KR-046)."""

import copy
import math
from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.v2.affect import AffectDynamics, AffectSpaceV2
from src.v2.cerebellum import CerebellumV2
from src.v2.model import V2Model
from src.v2.online import Controls, stream_adapt
from src.v2.online_v5 import (Condition, Drift, DriftMonitor, DynBase, DynSpec, Plasticity, collect_cerebellum_data,
                              collect_dyn_baseline, drift_baseline, drift_series, dyn_baseline_stream, stream_adapt_v5)

BASE = load_config(Path(__file__).resolve().parents[2] / "configs" / "v2_base.yaml")
C, D = 5, 2
CER = {"eta": 1.0e-2, "nlms_eps": 1.0e-4}
AFF = {"a_fast": 0.05, "a_slow": 0.001, "alpha_gate": 0.25, "eps": 1.0e-6}
ALPHA_SM = 0.92


def make_model(seed: int = 0, mode: str = "plif") -> V2Model:
    cfg = copy.deepcopy(BASE)
    cfg["backbone"]["N"] = 20
    cfg["readout"]["taus_ms"] = [16.0, 64.0]
    torch.manual_seed(seed)
    m = V2Model(cfg, C, D, mode)
    with torch.no_grad():
        m.readout.weight.normal_(0, 0.3)
        m.backbone.w_in.weight.mul_(3.0) if mode == "plif" else None
    m.readout.save_anchor()
    m.eval()
    return m


def data(n: int = 200, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, C, generator=g) * 2.0, torch.randn(n, D, generator=g)


def plast(delta_max: float = 0.5, alpha_in: float = 1e-4) -> Plasticity:
    return Plasticity(alpha_in=alpha_in, delta_max=delta_max, kappa_psi=0.9, kappa_x=0.8, eps=1e-4)


def cer_for(m: V2Model, w0: torch.Tensor | None = None, eps2_0: float = 0.1) -> CerebellumV2:
    P = m.backbone.N + D
    return CerebellumV2(torch.zeros(C, P) if w0 is None else w0, eps2_0, CER)


def run5(m: V2Model, x: torch.Tensor, y: torch.Tensor, conds: list[Condition], delay: int = 0, homeo: bool = False,
         start: int = 80, stop: int = 200, burnin: int = 30, p: Plasticity | None = None, chunk: int = 0,
         cer: CerebellumV2 | None = None, ctrl: Controls | None = None, drift: Drift | None = None,
         aff: AffectSpaceV2 | None = None, dyn_base: DynBase | None = None):
    return stream_adapt_v5(m, x, y, start, stop, burnin, conds, cer or cer_for(m), aff or AffectSpaceV2(AFF),
                           p or plast(), ALPHA_SM, 1e-4, 1e-4, homeo, chunk=chunk, label_delay=delay, controls=ctrl,
                           drift=drift, dyn_base=dyn_base)


def rn_ctrl(x: torch.Tensor) -> Controls:
    return Controls(renorm_m0=x.mean(0), renorm_v0=x.var(0), renorm_eps=1e-6)


def test_step_w_in_override_matches() -> None:
    m = make_model()
    x, _ = data(n=30)
    m.reset_state(2)
    a = torch.stack([m.backbone.step(x[t: t + 1].expand(2, -1)) for t in range(30)])
    m.reset_state(2)
    w = m.backbone.w_in.weight.detach().unsqueeze(0).repeat(2, 1, 1)
    th = m.backbone.theta.unsqueeze(0).repeat(2, 1)
    b = torch.stack([m.backbone.step(x[t: t + 1].expand(2, -1), w_in=w, theta=th) for t in range(30)])
    assert torch.allclose(a, b)
    g = make_model(mode="spiking_gru")
    with pytest.raises(ValueError):
        g.backbone.step(x[:1], theta=th[:1])


@pytest.mark.parametrize("homeo", [False, True])
@pytest.mark.parametrize("delay", [0, 7])
def test_no_plasticity_equals_v4(homeo: bool, delay: int) -> None:
    """η_in = 0, duygu kapalı → V4 akışıyla aynı (KV5)."""
    m = make_model()
    x, y = data()
    ref = stream_adapt(m, x, y, 80, 200, 30, [0.0, 0.05], 1e-4, 1e-4, homeo, label_delay=delay)
    res = run5(m, x, y, [Condition("A"), Condition("B", eta0=0.05)], delay=delay, homeo=homeo)
    assert torch.allclose(ref.pred, res.pred, atol=1e-5)
    assert torch.allclose(ref.dw_rel, res.dw_rel, atol=1e-6)
    assert float(res.din_rel.abs().max()) == 0.0


@pytest.mark.parametrize("delay", [0, 1, 7])
def test_zero_leakage_with_plasticity(delay: int) -> None:
    """y_t değişince v̂_{t+d}'ye kadar hiçbir tahmin değişmez (W_in ve okuma gecikmeli etiketle öğrenir)."""
    m = make_model()
    x, y = data()
    conds = [Condition("W", eta0=0.05, eta_in0=0.5), Condition("D", eta0=0.05, eta_in0=0.5, affect=True, gate=True,
                                                                  alpha_eta=2.0)]
    a = run5(m, x, y, conds, delay=delay).pred
    t = 150
    y2 = y.clone()
    y2[t] += 10.0
    b = run5(m, x, y2, conds, delay=delay).pred
    k = t - 80                                                               # skorlanan dizideki indeks
    assert torch.equal(a[:, : k + delay + 1], b[:, : k + delay + 1])
    assert not torch.equal(a[:, k + delay + 1:], b[:, k + delay + 1:])


def test_causal_in_input() -> None:
    """Gelecek girdi (beyincik hedefi x̄ dahil) geçmiş tahmini, a(t)'yi ve η'yı değiştirmez (KV6)."""
    m = make_model()
    x, y = data()
    conds = [Condition("D", eta0=0.05, eta_in0=0.5, affect=True, gate=True, alpha_eta=2.0)]
    a = run5(m, x, y, conds)
    x2 = x.clone()
    x2[160:] = 7.0
    b = run5(m, x2, y, conds)
    k = 160 - 80
    assert torch.equal(a.pred[:, :k], b.pred[:, :k]) and torch.equal(a.a[:, :k], b.a[:, :k])
    assert torch.equal(a.eta_in[:, :k], b.eta_in[:, :k])


def test_model_unchanged_and_ball() -> None:
    """KV1: akış sonrası tüm parametre ve buffer'lar aynı; W_in sapması norm topunda."""
    m = make_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x, y = data(n=400)
    y = y * 5.0
    res = run5(m, x, y, [Condition("W", eta0=0.05, eta_in0=50.0)], delay=3, homeo=True, start=80, stop=400,
               p=plast(delta_max=0.1, alpha_in=0.0))
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert float(res.din_max[0]) <= 0.1 + 1e-5 and float(res.din_rel[0]) > 0.0
    assert torch.isfinite(res.pred).all()


def test_affect_bounds_and_kv5() -> None:
    m = make_model()
    x, y = data(n=300)
    x[200:] *= 4.0                                                           # sürpriz yaratacak girdi kayması
    conds = [Condition("off", eta0=0.01, eta_in0=0.2),
             Condition("D", eta0=0.01, eta_in0=0.2, affect=True, gate=True, alpha_eta=2.0)]
    res = run5(m, x, y, conds, start=80, stop=300)
    assert torch.all(res.eta[0] == 0.01) and torch.all(res.eta_in[0] == 0.2)  # KV5: kapalıyken sabit
    a = res.a[1]
    assert torch.all(a.abs() <= 1.0)                                         # tanh float32'de 1'e doyabilir
    assert torch.allclose(res.eta[1], 0.01 * (1.0 + 2.0 * a.clamp(min=0.0)))
    assert torch.allclose(res.eta_in[1], 0.2 * a.clamp(min=0.0))           # a ≤ 0 → η_in = 0
    assert float(a.max()) > 0.0                                              # kaymada sürpriz yükseldi


def test_external_schedule_used() -> None:
    """v2-C: dış dizi verilen koşul o diziyi kullanır."""
    m = make_model()
    x, y = data()
    L = 120
    seq = torch.rand(L, generator=torch.Generator().manual_seed(3)) * 0.05
    seq_in = torch.rand(L, generator=torch.Generator().manual_seed(4)) * 0.3
    res = run5(m, x, y, [Condition("C", affect=True, gate=True, eta_seq=seq, eta_in_seq=seq_in)])
    assert torch.allclose(res.eta[0], seq) and torch.allclose(res.eta_in[0], seq_in)


def test_cerebellum_fit_and_collect() -> None:
    g = torch.Generator().manual_seed(0)
    phi = torch.randn(500, 7, generator=g)
    w = torch.randn(3, 7, generator=g)
    w_hat, e2 = CerebellumV2.fit(phi, phi @ w.T, ridge=1e-8)
    assert torch.allclose(w_hat, w, atol=1e-3) and e2 < 1e-6
    m = make_model()
    x, _ = data()
    ph, tg = collect_cerebellum_data(m, x, 50, 200, 20, ALPHA_SM)
    assert ph.shape == (149, m.backbone.N + D) and tg.shape == (149, C)


def test_fitted_cerebellum_low_surprise_on_same_data() -> None:
    """Day-0'da ridge'le başlatılan beyincikte aynı dağılımda a ≈ 0; ölçek kaymasında a yükselir."""
    m = make_model()
    x, y = data(n=900)
    ph, tg = collect_cerebellum_data(m, x, 30, 500, 30, ALPHA_SM)
    w0, e2 = CerebellumV2.fit(ph, tg, ridge=1e-3)
    cer = CerebellumV2(w0, e2, {"eta": 0.0, "nlms_eps": 1e-4})
    cond = [Condition("D", affect=True)]
    same = run5(m, x, y, cond, start=500, stop=900, burnin=30, cer=cer).a
    x2 = x.clone()
    x2[500:] = x2[500:] * 3.0 + 2.0
    cer2 = CerebellumV2(w0, e2, {"eta": 0.0, "nlms_eps": 1e-4})
    shifted = run5(m, x2, y, cond, start=500, stop=900, burnin=30, cer=cer2).a
    assert float(shifted.mean()) > float(same.mean()) + 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
@pytest.mark.parametrize("delay", [0, 16])
def test_v5_graph_matches_eager(delay: int) -> None:
    dev = torch.device("cuda")
    x, y = data(n=300)
    x, y = x.to(dev) * 3, y.to(dev)
    m = make_model().to(dev)
    L = 200
    seq = torch.linspace(0, 0.05, L, device=dev)
    conds = [Condition("B", eta0=0.05), Condition("W", eta0=0.05, eta_in0=0.5),
             Condition("D", eta0=0.05, eta_in0=0.5, affect=True, gate=True, alpha_eta=2.0),
             Condition("C", affect=True, gate=True, eta_seq=seq, eta_in_seq=seq * 10),
             Condition("R", eta0=0.05, renorm=True, lam_calm=0.02),
             Condition("G", eta0=0.05, renorm=True, lam_calm=0.001, lam_surp=0.1, affect=True, gate=True, alpha_eta=2.0),
             Condition("Cr", renorm=True, lam_calm=0.001, affect=True, gate=True, eta_seq=seq, lam_seq=seq * 2)]

    def go(chunk: int):
        cer = CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)).to(dev) * 0.1,
                           0.1, CER)
        return stream_adapt_v5(m, x, y, 100, 300, 40, conds, cer, AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-3, 1e-4,
                               True, chunk=chunk, label_delay=delay,
                               controls=Controls(renorm_m0=x.mean(0) * 0.5, renorm_v0=x.var(0) * 2.0, renorm_eps=1e-6))

    e, g = go(0), go(64)
    for f in ("pred", "a", "eta", "eta_in", "lam", "din_rel", "dw_rel", "rate"):
        assert torch.allclose(getattr(e, f), getattr(g, f), atol=1e-4), f


# --- KR-045 duygu-kapılı kanal yeniden ölçeklemesi (§12.14) -------------------------------------------------


def test_renorm_zero_rate_is_identity() -> None:
    m = make_model()
    x, y = data()
    a = run5(m, x, y, [Condition("B", eta0=0.05)], delay=7)
    b = run5(m, x, y, [Condition("R0", eta0=0.05, renorm=True)], delay=7, ctrl=rn_ctrl(x))
    assert torch.allclose(a.pred, b.pred, atol=1e-5)


def test_renorm_fixed_matches_v4_controls() -> None:
    """Duygusuz, sabit λ_norm'lu koşul V4'teki ortak yeniden ölçeklemeyle aynı (λ = 1 − decay)."""
    m = make_model()
    x, y = data()
    lam = 0.05
    ref = stream_adapt(m, x, y, 80, 200, 30, [0.05], 1e-4, 1e-4, False, label_delay=7,
                       controls=Controls(renorm_m0=x.mean(0), renorm_v0=x.var(0), renorm_decay=1.0 - lam, renorm_eps=1e-6))
    res = run5(m, x, y, [Condition("R", eta0=0.05, renorm=True, lam_calm=lam)], delay=7, ctrl=rn_ctrl(x))
    assert torch.allclose(ref.pred, res.pred, atol=1e-5)


def test_renorm_rate_follows_previous_arousal() -> None:
    """λ_norm(t) = λ_calm + (λ_surp − λ_calm)·max(0, a(t−1)); duygu kapalıyken λ_calm (KV4, KV5)."""
    m = make_model()
    x, y = data(n=300)
    x[200:] = x[200:] * 3.0 + 1.0
    conds = [Condition("R", renorm=True, lam_calm=0.01, lam_surp=0.2),
             Condition("G", renorm=True, lam_calm=0.01, lam_surp=0.2, affect=True)]
    res = run5(m, x, y, conds, start=80, stop=300, ctrl=rn_ctrl(x[:150]))
    assert torch.all(res.lam[0] == 0.01)
    a = res.a[1]
    assert float(res.lam[1, 0]) == pytest.approx(0.01)                      # burn-in'de a = 0
    assert torch.allclose(res.lam[1, 1:], 0.01 + 0.19 * a[:-1].clamp(min=0.0), atol=1e-6)
    assert float(res.lam[1].max()) > 0.05


def test_renorm_causal() -> None:
    m = make_model()
    x, y = data()
    conds = [Condition("G", eta0=0.05, renorm=True, lam_calm=0.01, lam_surp=0.3, affect=True, gate=True, alpha_eta=2.0)]
    a = run5(m, x, y, conds, ctrl=rn_ctrl(x))
    x2 = x.clone()
    x2[160:] = -5.0
    b = run5(m, x2, y, conds, ctrl=rn_ctrl(x))
    k = 160 - 80
    assert torch.equal(a.pred[:, :k], b.pred[:, :k]) and torch.equal(a.lam[:, :k + 1], b.lam[:, :k + 1])


def test_renorm_requires_reference() -> None:
    m = make_model()
    x, y = data()
    with pytest.raises(ValueError):
        run5(m, x, y, [Condition("R", renorm=True, lam_calm=0.1)])


# --- KR-046 kalıcı sürpriz: ham hedef, donuk W_cer, sabit taban, girdi sapması, eşikli kapı ---------------------

DR = Drift(lam=0.05, var_floor=1e-2, delta0=0.01)


def persist(name: str, src: int = -1, w: float = 0.5, renorm: bool = True, lo: float = 0.2, hi: float = 0.6,
            **kw) -> Condition:
    return Condition(name, renorm=renorm, lam_surp=0.2, affect=True, cer_raw=True, cer_eta=0.0, cer_src=src,
                     fixed_slow=True, drift_w=w, gate_lo=lo, gate_hi=hi, **kw)


def test_drift_monitor_baseline_and_shift() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, C, generator=g)
    m0, v0 = x.mean(0), x.var(0)
    spec = Drift(lam=0.02, var_floor=1e-2)
    one = torch.ones(())
    assert float(DriftMonitor(m0, v0, spec).step(x[0] + 5.0, torch.zeros(()))) == 0.0   # maske 0: durum Day-0'da
    mon = DriftMonitor(m0, v0, spec)
    vals = torch.stack([mon.step(x[t], one) for t in range(2000)])
    ser = drift_series(x, 0, 2000, m0, v0, spec, block=300)                 # vektörleştirilmiş = adım adım
    assert torch.allclose(ser.float(), vals, rtol=1e-3, atol=1e-6)
    assert drift_baseline(x, 0, 2000, m0, v0, spec) == pytest.approx(float(vals.double().mean()), rel=1e-4)
    assert drift_baseline(x, 0, 2000, m0, v0, spec, stat="max") == pytest.approx(float(vals.max()), rel=1e-4)
    mon = DriftMonitor(m0, v0, spec)
    shifted = torch.stack([mon.step(x[t] * 2.0 + 1.0, one) for t in range(2000)])
    assert float(shifted[-500:].mean()) > 10.0 * float(vals[-500:].mean())


def test_persistent_frozen_cerebellum_and_fixed_baseline() -> None:
    """cer_eta = 0 → W_cer değişmez; fixed_slow → E_slow Day-0 tabanında (1) kalır; diğer koşullar uyarlanır."""
    m = make_model()
    x, y = data(n=300)
    x[200:] = x[200:] * 3.0 + 1.0
    w0 = torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)) * 0.1
    cer, aff = CerebellumV2(w0, 0.5, CER), AffectSpaceV2({**AFF, "a_slow": 0.05})
    conds = [Condition("D", affect=True), persist("P", w=0.0, renorm=False)]
    run5(m, x, y, conds, start=80, stop=300, cer=cer, aff=aff)
    assert torch.equal(cer.W[1], w0) and not torch.equal(cer.W[0], w0)
    assert float(aff.e_slow[1]) == 1.0 and float(aff.e_slow[0]) != 1.0


def test_persistent_surprise_stays_high_after_shift() -> None:
    """Kaymadan sonra: sabit taban + donuk W_cer + ham hedef → a yüksek kalır; uyarlanan taban ve NLMS söndürür."""
    m = make_model()
    x, y = data(n=900)
    ph, tg = collect_cerebellum_data(m, x, 30, 500, 30, ALPHA_SM)
    w0, e2 = CerebellumV2.fit(ph, tg, ridge=1e-3)
    x2 = x.clone()
    x2[600:] = x2[600:] * 3.0 + 2.0
    conds = [Condition("D", affect=True), persist("P", w=0.0, renorm=False)]
    res = run5(m, x2, y, conds, start=500, stop=900, burnin=30, cer=CerebellumV2(w0, e2, CER),
               aff=AffectSpaceV2({**AFF, "a_slow": 0.05}))
    late = slice(300, 400)                                                   # kaymadan 200+ adım sonra
    assert float(res.a[1, late].mean()) > 0.5
    assert float(res.a[0, late].mean()) < float(res.a[1, late].mean()) - 0.3


def test_monitor_surprise_independent_of_renorm() -> None:
    """cer_src = A (izleme yolu) + ham hedef + donuk W_cer: sürpriz koşulun kendi yeniden ölçeklemesinden bağımsız."""
    m = make_model()
    x, y = data(n=300)
    x[200:] = x[200:] * 3.0 + 1.0
    conds = [Condition("A"), persist("Pm", src=0, lo=0.0, hi=0.1), persist("Pm0", src=0, renorm=False),
             persist("P", lo=0.0, hi=0.1)]
    res = run5(m, x, y, conds, start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR,
               cer=CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)) * 0.1,
                                0.5, CER))
    sc = res.extra["s_cer"]
    assert float(res.lam[1].max()) > 0.1                                     # yeniden ölçekleme gerçekten açıldı
    assert torch.equal(sc[1], sc[2]) and torch.equal(res.a[1], res.a[2])
    assert not torch.equal(sc[1], sc[3])                                     # kendi yolu yeniden ölçeklemeyi görür
    assert torch.equal(res.extra["s_drift"][1], res.extra["s_drift"][3])     # sapma yalnızca ham girdiden


def test_gate_threshold_formula() -> None:
    """λ_norm(t) = λ_calm + (λ_surp − λ_calm)·clamp((a(t−1) − lo)/(hi − lo), 0, 1); a < lo iken donuk."""
    m = make_model()
    x, y = data(n=300)
    x[200:] = x[200:] * 3.0 + 1.0
    res = run5(m, x, y, [persist("P", src=-1, w=1.0, lo=0.2, hi=0.6)], start=80, stop=300, ctrl=rn_ctrl(x[:150]),
               drift=DR)
    a, lam = res.a[0], res.lam[0]
    assert torch.allclose(lam[1:], 0.2 * ((a[:-1] - 0.2) / 0.4).clamp(0.0, 1.0), atol=1e-6)
    assert float(lam[0]) == 0.0 and float(lam.max()) == pytest.approx(0.2) and bool((lam[:100] == 0).any())
    sd = res.extra["s_drift"][0]
    assert float(sd[-50:].mean()) > float(sd[:100].mean())                  # kaymada sapma yükseldi


def test_persistent_causal_and_requires_drift() -> None:
    m = make_model()
    x, y = data()
    conds = [Condition("A"), persist("P", eta0=0.05), persist("Pm", src=0, eta0=0.05)]
    a = run5(m, x, y, conds, ctrl=rn_ctrl(x), drift=DR)
    x2 = x.clone()
    x2[160:] = -5.0
    b = run5(m, x2, y, conds, ctrl=rn_ctrl(x), drift=DR)
    k = 160 - 80
    assert torch.equal(a.pred[:, :k], b.pred[:, :k]) and torch.equal(a.lam[:, :k + 1], b.lam[:, :k + 1])
    for f in ("s_cer", "s_drift"):
        assert torch.equal(a.extra[f][:, :k], b.extra[f][:, :k])
    with pytest.raises(ValueError):
        run5(m, x, y, conds, ctrl=rn_ctrl(x))                                # sapma terimi Drift ister
    with pytest.raises(ValueError):
        run5(m, x, y, [persist("P", lo=0.5, hi=0.5)], ctrl=rn_ctrl(x), drift=DR)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
@pytest.mark.parametrize("delay", [0, 16])
def test_persistent_graph_matches_eager(delay: int) -> None:
    dev = torch.device("cuda")
    x, y = data(n=300)
    x, y = x.to(dev) * 3, y.to(dev)
    x[200:] = x[200:] * 2.0 + 1.0
    m = make_model().to(dev)
    conds = [Condition("A"), Condition("R", eta0=0.05, renorm=True, lam_calm=0.02),
             persist("P", eta0=0.05), persist("Pm", src=0, eta0=0.05, gate=True, alpha_eta=2.0),
             persist("Pd", src=0, w=1.0, eta0=0.05), Condition("G", eta0=0.05, renorm=True, lam_calm=0.001,
                                                                lam_surp=0.1, affect=True, gate=True, alpha_eta=2.0)]

    def go(chunk: int):
        cer = CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)).to(dev) * 0.1,
                           0.1, CER)
        return stream_adapt_v5(m, x, y, 100, 300, 40, conds, cer, AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-3, 1e-4,
                               False, chunk=chunk, label_delay=delay, controls=rn_ctrl(x[:150]),
                               drift=Drift(lam=0.05, var_floor=1e-2, delta0=0.05))

    e, g = go(0), go(64)
    for f in ("pred", "a", "eta", "lam", "dw_rel", "rate"):
        assert torch.allclose(getattr(e, f), getattr(g, f), atol=1e-4), f
    for f in ("s_cer", "s_drift"):
        assert torch.allclose(e.extra[f], g.extra[f], rtol=1e-4, atol=1e-4), f


def test_calm_return_to_reference() -> None:
    """lam_return: geçici kaymada açılan kapı kapanınca m, v Day-0 referansına döner; dönüşsüzde kaydırma kalır."""
    m = make_model()
    x, y = data(n=500)
    ctrl = rn_ctrl(x[:150])
    spec = Drift(lam=0.05, var_floor=1e-2)
    spec.delta0 = 4.0 * drift_baseline(x, 0, 150, ctrl.renorm_m0, ctrl.renorm_v0, spec)   # tolerans: taban × 4
    x[150:250] = x[150:250] * 3.0 + 2.0                                      # geçici kayma
    kw = dict(w=1.0, lo=0.2, hi=0.6)
    conds = [Condition("B"), persist("P", **kw), persist("Pr", lam_return=0.2, **kw)]
    res = run5(m, x, y, conds, start=80, stop=500, ctrl=ctrl, drift=spec)
    assert float(res.lam[1, 70:170].max()) > 0.1 and float(res.lam[1, -100:].max()) == 0.0   # açıldı, sonra kapandı
    late = slice(-50, None)
    d_ret = float((res.pred[2, late] - res.pred[0, late]).abs().mean())
    d_no = float((res.pred[1, late] - res.pred[0, late]).abs().mean())
    assert d_ret < 0.3 * d_no


def test_drift_max_combination() -> None:
    """drift_max: E_fast ← E_fast + a_f·(max(s_cer, s_drift) − E_fast), E_slow = 1 sabit, a = tanh(E_fast − 1)."""
    m = make_model()
    x, y = data(n=300)
    x[200:] = x[200:] * 3.0 + 1.0
    af = 0.1
    res = run5(m, x, y, [Condition("A"), persist("Pm", src=0, drift_max=True, a_fast=af)], start=80, stop=300,
               ctrl=rn_ctrl(x[:150]), drift=DR)
    mix = torch.maximum(res.extra["s_cer"][1], res.extra["s_drift"][1])
    ef, a_ref = 1.0, []
    for v in mix.tolist():
        ef += af * (v - ef)
        a_ref.append(math.tanh((ef - 1.0) / (1.0 + 1e-6)))
    assert torch.allclose(res.a[1], torch.tensor(a_ref), atol=1e-4)


# --- KR-047 dinamik duygu: arousal (düzey), valens (eğilim), dominans; gaz / fren / gevşetme / konsolidasyon ----------

DB = DynBase(u0=0.05, q0=1.0)


def dspec(**kw) -> DynSpec:
    base = dict(w_u=0.5, lam_a=0.1, lam_f=0.2, lam_m=0.02, lam_d=0.02, rho_v=0.5, alpha_b=0.05, lam_c=0.05,
                delta_m=0.05)
    flags = {k: kw.pop(k) for k in ("gas", "brake", "loosen", "consolidate", "a_mode", "gas_dom") if k in kw}
    return DynSpec(**{**base, **kw}, **flags)


def dcond(name: str, spec: DynSpec | None = None, **kw) -> Condition:
    """Dinamik duygulu koşul: sensör Pm ile aynı (ortak izleme yolu), kendi beyinciği iç yol (yeniden ölçeklenmiş hedef)."""
    base = dict(eta0=0.05, renorm=True, lam_surp=0.2, affect=True, gate=True, alpha_eta=2.0, cer_eta=0.0,
                fixed_slow=True, drift_max=True, gate_lo=0.2, gate_hi=0.6)
    return Condition(name, dyn=spec or dspec(), **{**base, **kw})


def shifted(n: int = 300, at: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = data(n=n)
    x[at:] = x[at:] * 3.0 + 1.0
    return x, y


def test_affect_dynamics_valence_and_dominance() -> None:
    """Valens eğilimdir (sabit q → 0, düşen q → > 0, artan → < 0), dominans düzeydir; arousal'dan bağımsızdır."""
    one, o = torch.ones(()), torch.ones(1)
    ad = AffectDynamics(o * 0.0, o * 0.1, o * 0.2, o * 0.02, o * 0.05, 1e-6)
    for _ in range(800):
        ad.update_valence(o * 2.0, one)
        ad.update_arousal(o * 3.0, o * 1.0, one)
    assert abs(float(ad.v)) < 1e-3                                          # sabit hata: eğilim yok
    assert float(ad.d) == pytest.approx(-math.tanh(math.log(2.0)), abs=1e-3)   # Day-0'ın 2 katı → d < 0
    assert float(ad.a) == pytest.approx(math.tanh(2.0), abs=1e-3)           # arousal düzeyi (3 = Day-0'ın 3 katı)
    for q in torch.linspace(2.0, 0.5, 100):
        ad.update_valence(o * q, one)
    assert float(ad.v) > 0.2                                                 # düşen hata → rahatlama
    for q in torch.linspace(0.5, 3.0, 100):
        ad.update_valence(o * q, one)
    assert float(ad.v) < -0.2
    ad.q_d.fill_(4.0)
    ad.update_valence(o * 100.0, torch.zeros(()))                            # maske 0: durum ilerlemez
    assert float(ad.q_d) == 4.0 and float(ad.d) < -0.8
    ad.q_d.fill_(1.0)
    ad.update_valence(o, torch.zeros(()))
    assert float(ad.d) == pytest.approx(0.0, abs=1e-5)


def test_affect_dynamics_controls() -> None:
    """Sakin / yeni gün / fren / rahatlama; m = 0 iken fren ve konsolidasyon 0; tüm çıktılar sınırlarda (KV4)."""
    a = torch.tensor([0.0, 0.9, 0.9, 0.0])
    v = torch.tensor([0.0, 0.0, -0.8, 0.8])
    d = torch.tensor([0.0, -0.9, 0.0, 0.0])
    m = torch.tensor([0.0, 0.0, 1.0, 1.0])
    one = torch.ones(4)
    eta, alpha, b, c = AffectDynamics.controls(a, v, d, m, one * 0.01, one * 2.0, one * 0.5, one * 1e-3, one * 0.05,
                                               one, one, one, one)
    assert torch.allclose(eta, torch.tensor([0.01, 0.028, 0.028 * 0.2, 0.01 * 0.6]))
    assert torch.allclose(alpha, torch.tensor([1e-3, 1e-3 * (1 - 0.81), 1e-3 + 0.05 * 0.8, 1e-3]))
    assert torch.allclose(b, torch.tensor([0.0, 0.0, 0.8, 0.0])) and torch.allclose(c, torch.tensor([0.0, 0.0, 0.0, 0.8]))
    g = torch.Generator().manual_seed(0)
    a, v, d = (torch.rand(1000, generator=g) * 2 - 1 for _ in range(3))
    m = torch.rand(1000, generator=g)
    one = torch.ones(1000)
    eta, alpha, b, c = AffectDynamics.controls(a, v, d, m, one * 0.01, one * 2.0, one * 0.5, one * 1e-3, one * 0.05,
                                               one, one, one, one)
    assert float(eta.min()) >= 0.0 and float(eta.max()) <= 0.03 + 1e-7
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1e-3 + 0.05 + 1e-7
    assert float(b.min()) >= 0.0 and float(b.max()) <= 1.0 and float(c.min()) >= 0.0 and float(c.max()) <= 1.0
    z = torch.zeros(1000)                                                    # bayraklar kapalı → B davranışı
    eta, alpha, b, c = AffectDynamics.controls(a, v, d, m, one * 0.01, one * 2.0, z, one * 1e-3, one * 0.05, z, z, z, z)
    assert torch.all(eta == 0.01) and torch.all(alpha == 1e-3) and torch.all(b == 0) and torch.all(c == 0)
    a2, d2, z2 = torch.tensor([0.9, 0.9]), torch.tensor([0.0, -0.5]), torch.zeros(2)   # gas_dom: gaz a⁺·d⁻
    eta, _, _, _ = AffectDynamics.controls(a2, z2, d2, z2, torch.full((2,), 0.01), torch.full((2,), 2.0), z2,
                                           torch.full((2,), 1e-3), z2, torch.ones(2), z2, z2, z2, torch.ones(2))
    assert torch.allclose(eta, torch.tensor([0.01, 0.01 * (1 + 2 * 0.45)]))


def test_dyn_arousal_modes() -> None:
    """a_mode: 0 iç arousal, 1 sensör arousal'ı (yeniden ölçekleme kapısıyla aynı), 2 ikisinin maksimumu."""
    m = make_model()
    x, y = shifted()
    conds = [Condition("A"), dcond("Di"), dcond("Ds", dspec(a_mode=1)), dcond("Dx", dspec(a_mode=2))]
    res = run5(m, x, y, conds, start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR, dyn_base=DB)
    ai, sens = res.extra["a_int"], res.a
    assert torch.allclose(ai[2], sens[2]) and not torch.allclose(ai[1], sens[1])
    assert torch.allclose(ai[3], torch.maximum(ai[1], sens[3]), atol=1e-6)


def test_dyn_leaves_other_conditions_unchanged() -> None:
    """Akışa dyn koşulu eklemek A, B, Pm, R, O çıktılarını değiştirmez."""
    m = make_model()
    x, y = shifted()
    ctrl = rn_ctrl(x[:150])
    conds = [Condition("A"), Condition("B", eta0=0.05), persist("Pm", src=0, drift_max=True, eta0=0.05),
             Condition("R", eta0=0.05, renorm=True, lam_calm=0.02)]
    cer = lambda: CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)) * 0.1,   # noqa: E731
                               0.5, CER)
    a = run5(m, x, y, conds, delay=7, start=80, stop=300, ctrl=ctrl, drift=DR, cer=cer())
    b = run5(m, x, y, conds + [dcond("D")], delay=7, start=80, stop=300, ctrl=ctrl, drift=DR, cer=cer(), dyn_base=DB)
    for f in ("pred", "a", "eta", "lam", "dw_rel"):
        assert torch.allclose(getattr(a, f), getattr(b, f)[:4], atol=1e-6), f
    assert float(b.extra["anc_rel"][:4].abs().max()) == 0.0                 # dyn'siz koşulda çapa W_0


def test_shared_sensor_equals_persistent_sensor() -> None:
    """Dyn koşulunun yeniden ölçekleme sensörü (ortak A yolu, donuk W_cer, ham x̄) Pm'in koşul başı sensörüyle aynı."""
    m = make_model()
    x, y = shifted()
    conds = [Condition("A"), persist("Pm", src=0, drift_max=True, a_fast=0.1, eta0=0.05),
             dcond("D", a_fast=0.1)]
    res = run5(m, x, y, conds, start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR, dyn_base=DB,
               cer=CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)) * 0.1,
                                0.5, CER))
    assert torch.allclose(res.extra["s_sens"][2], res.extra["s_cer"][1], atol=1e-6)
    assert torch.allclose(res.a[2], res.a[1], atol=1e-6) and torch.allclose(res.lam[2], res.lam[1], atol=1e-6)
    assert float(res.lam[2].max()) > 0.1                                     # kaymada kapı açıldı
    assert not torch.allclose(res.extra["s_cer"][2], res.extra["s_cer"][1])  # iç yol ayrı (yeniden ölçeklenmiş hedef)


@pytest.mark.parametrize("delay", [0, 1, 7])
def test_dyn_zero_leakage(delay: int) -> None:
    """y_t değişince t + d'ye kadar v̂, η, α, c değişmez (valens e_v(t)'den sonra güncellenir)."""
    m = make_model()
    x, y = shifted()
    conds = [Condition("A"), dcond("D"), dcond("Dl", dspec(brake=False))]
    kw = dict(delay=delay, start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR, dyn_base=DB)
    a = run5(m, x, y, conds, **kw)
    t = 230
    y2 = y.clone()
    y2[t] += 10.0
    b = run5(m, x, y2, conds, **kw)
    k = t - 80 + delay + 1
    assert torch.equal(a.pred[:, :k], b.pred[:, :k]) and torch.equal(a.eta[:, :k], b.eta[:, :k])
    for f in ("alpha", "cons", "val", "dom"):
        assert torch.equal(a.extra[f][:, :k], b.extra[f][:, :k]), f
    assert not torch.equal(a.pred[1, k:], b.pred[1, k:]) and not torch.equal(a.extra["val"][1, k:], b.extra["val"][1, k:])


def test_dyn_causal_in_input() -> None:
    m = make_model()
    x, y = shifted()
    conds = [Condition("A"), dcond("D")]
    kw = dict(delay=7, start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR, dyn_base=DB)
    a = run5(m, x, y, conds, **kw)
    x2 = x.clone()
    x2[250:] = -5.0
    b = run5(m, x2, y, conds, **kw)
    k = 250 - 80
    assert torch.equal(a.pred[:, :k], b.pred[:, :k])
    for f in ("a_int", "u", "alpha", "cons"):
        assert torch.equal(a.extra[f][:, :k], b.extra[f][:, :k]), f


def test_dyn_bounds_consolidation_and_model_unchanged() -> None:
    """KV4 sınırları; konsolidasyon kapalıyken çapa W_0'da; açıkken W'ye doğru; akış sonunda model aynı (KV1)."""
    m = make_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x, y = shifted(n=500, at=250)
    y = y * 3.0
    conds = [Condition("A"), dcond("D"), dcond("Dnc", dspec(consolidate=False)), dcond("Dg", dspec(brake=False))]
    res = run5(m, x, y, conds, delay=3, start=80, stop=500, ctrl=rn_ctrl(x[:150]), drift=DR, dyn_base=DB)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
    eta, al, cn, br = res.eta[1:], res.extra["alpha"][1:], res.extra["cons"][1:], res.extra["brake"][1:]
    assert float(eta.min()) >= 0.0 and float(eta.max()) <= 0.05 * 3.0 + 1e-6
    assert float(al.min()) >= 0.0 and float(al.max()) <= 1e-4 + 0.05 + 1e-6
    assert float(cn.min()) >= 0.0 and float(cn.max()) <= 1.0 and float(br.min()) >= 0.0 and float(br.max()) <= 1.0
    anc = res.extra["anc_rel"]
    assert float(anc[2]) == 0.0 and float(anc[1]) > 0.0 and float(anc[0]) == 0.0
    assert float(cn[0].max()) > 0.0 and float(br[0].max()) > 0.0 and float(br[2].max()) == 0.0
    assert torch.isfinite(res.pred).all()


def test_dyn_external_schedules_and_requirements() -> None:
    """v2-C: α ve c dış dizileri kullanılır; dyn_base / koşul 0 = A / dizi için dyn zorunlu."""
    m = make_model()
    x, y = shifted()
    L = 220
    g = torch.Generator().manual_seed(3)
    se, sa, sc = torch.rand(L, generator=g) * 0.05, torch.rand(L, generator=g) * 0.01, torch.rand(L, generator=g)
    kw = dict(start=80, stop=300, ctrl=rn_ctrl(x[:150]), drift=DR)
    res = run5(m, x, y, [Condition("A"), dcond("C", eta_seq=se, alpha_seq=sa, cons_seq=sc)], dyn_base=DB, **kw)
    assert torch.allclose(res.eta[1], se) and torch.allclose(res.extra["alpha"][1], sa)
    assert torch.allclose(res.extra["cons"][1], sc)
    with pytest.raises(ValueError):
        run5(m, x, y, [Condition("A"), dcond("D")], **kw)                    # DynBase yok
    with pytest.raises(ValueError):
        run5(m, x, y, [Condition("B", eta0=0.05), dcond("D")], dyn_base=DB, **kw)   # koşul 0 donuk A değil
    with pytest.raises(ValueError):
        run5(m, x, y, [Condition("A"), Condition("X", eta0=0.05, alpha_seq=sa)], **kw)


def test_dead_channels_renorm_finite() -> None:
    """Elektrot kaybı (v2-E): ölü kanalda varyans → 0; yeniden ölçeklenmiş girdi sonlu kalır, kanal Day-0 ortalamasına iner."""
    m = make_model()
    x, y = data(n=600)
    x[250:, :2] = -1.3                                                       # iki kanal sabit ("0 sayım" kodu)
    conds = [Condition("A"), Condition("R", eta0=0.05, renorm=True, lam_calm=0.05), dcond("D")]
    res = run5(m, x, y, conds, delay=3, start=80, stop=600, ctrl=rn_ctrl(x[:200]), drift=DR, dyn_base=DB)
    assert torch.isfinite(res.pred).all() and torch.isfinite(res.extra["s_drift"]).all()
    lam, v0, m0 = 0.05, x[:200].var(0), x[:200].mean(0)                     # R'nin EMA'sı elle: x' → m_0
    mm, vv = x[:200].mean(0).clone(), x[:200].var(0).clone()
    for t in range(30, 600):
        d = x[t] - mm
        mm += lam * d
        vv += lam * (d * d - vv)
    x_rn = m0 + (x[599] - mm) * torch.sqrt(v0 / (vv + 1e-6))
    assert torch.allclose(x_rn[:2], m0[:2], atol=1e-2)


def test_alpha_anchor_override() -> None:
    """alpha_anchor = akışın α'sı → aynı sonuç; farklı α → farklı (eşlenmiş kontroller için)."""
    m = make_model()
    x, y = data()
    a = run5(m, x, y, [Condition("B", eta0=0.05)], delay=7)
    b = run5(m, x, y, [Condition("B", eta0=0.05, alpha_anchor=1e-4), Condition("B2", eta0=0.05, alpha_anchor=0.05)],
             delay=7)
    assert torch.allclose(a.pred[0], b.pred[0], atol=1e-6) and not torch.allclose(b.pred[0], b.pred[1])


def test_dyn_baseline_matches_frozen_stream() -> None:
    """q_0 = donuk modelin aynı dilimdeki ort. (y − v̂)²'si; u_0 > 0."""
    m = make_model()
    x, y = data()
    base = collect_dyn_baseline(m, x, y, 80, 200, 30)
    res = run5(m, x, y, [Condition("A")])
    assert base.u0 > 0.0
    assert base.q0 == pytest.approx(float((y[80:200] - res.pred[0]).pow(2).mean()), rel=1e-4)
    fast = dyn_baseline_stream(m, x, y, 80, 200, 30, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4)
    assert fast.u0 == pytest.approx(base.u0, rel=1e-4) and fast.q0 == pytest.approx(base.q0, rel=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
@pytest.mark.parametrize("delay", [0, 16])
def test_dyn_graph_matches_eager(delay: int) -> None:
    dev = torch.device("cuda")
    x, y = shifted()
    x, y = x.to(dev), y.to(dev) * 3.0
    m = make_model().to(dev)
    L = 200
    g = torch.Generator().manual_seed(3)
    se, sa, sc = (torch.rand(L, generator=g).to(dev) * s for s in (0.05, 0.01, 1.0))
    conds = [Condition("A"), Condition("B", eta0=0.05, alpha_anchor=1e-3), persist("Pm", src=0, drift_max=True, eta0=0.05),
             dcond("D"), dcond("Dn", dspec(gas=False, loosen=False)), dcond("C", eta_seq=se, alpha_seq=sa, cons_seq=sc)]

    def go(chunk: int):
        cer = CerebellumV2(torch.randn(C, m.backbone.N + D, generator=torch.Generator().manual_seed(5)).to(dev) * 0.1,
                           0.5, CER)
        return stream_adapt_v5(m, x, y, 100, 300, 40, conds, cer, AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-3, 1e-4,
                               False, chunk=chunk, label_delay=delay, controls=rn_ctrl(x[:150]),
                               drift=Drift(lam=0.05, var_floor=1e-2, delta0=0.05), dyn_base=DB)

    e, gr = go(0), go(64)
    for f in ("pred", "a", "eta", "lam", "dw_rel"):
        assert torch.allclose(getattr(e, f), getattr(gr, f), atol=1e-4), f
    for f in ("a_int", "val", "dom", "alpha", "cons", "u", "brake", "anc_rel"):
        assert torch.allclose(e.extra[f], gr.extra[f], rtol=1e-4, atol=1e-4), f
