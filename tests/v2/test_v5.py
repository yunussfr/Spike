"""Faz V5: beyincik, duygu uzayı, duygu-kapılı W_in plastisitesi / yeniden ölçekleme, kalıcı sürpriz
(KV1–KV6, §12.5, §12.6, §12.13, §12.14; KR-044–KR-046)."""

import copy
import math
from pathlib import Path

import pytest
import torch

from src.utils.config import load_config
from src.v2.affect import AffectSpaceV2
from src.v2.cerebellum import CerebellumV2
from src.v2.model import V2Model
from src.v2.online import Controls, stream_adapt
from src.v2.online_v5 import (Condition, Drift, DriftMonitor, Plasticity, collect_cerebellum_data, drift_baseline,
                              drift_series, stream_adapt_v5)

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
         aff: AffectSpaceV2 | None = None):
    return stream_adapt_v5(m, x, y, start, stop, burnin, conds, cer or cer_for(m), aff or AffectSpaceV2(AFF),
                           p or plast(), ALPHA_SM, 1e-4, 1e-4, homeo, chunk=chunk, label_delay=delay, controls=ctrl,
                           drift=drift)


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
