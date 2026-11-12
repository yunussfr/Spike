"""Faz V7 (KR-049): sönümlü ReFIT (§12.15), BOCPD sensörü (§12.16), ölü kanal dedektörü (§12.17), Degenhart
hizalaması (§12.18), kanal bozulması augmentation'ı (§12.19), tonik / fazik iş bölümü (§12.20, KR-050) ve akışa
bağlanmaları (KV1, KV2, KV4, KV6)."""

import math

import pytest
import torch

from src.v2.affect import AffectDynamics, AffectSpaceV2
from src.v2.align import FA, Aligner, AlignSpec, bin_means, channel_map, fa_fit, latent_map, procrustes, stable_alignment
from src.v2.augment import AugSpec, augment_windows
from src.v2.changepoint import (BOCPD, BocpdSpec, ScoreRef, block_means, block_scores, bocpd_series, calibrate_temper,
                                fit_score_ref)
from src.v2.dead_channel import DeadChannelDetector, silence_threshold, spike_prob
from src.v2.intent import Intent, reaction_mask, refit_direction, refit_label
from src.v2.online_v5 import (Condition, DynBase, V7Inputs, collect_streams, dyn_baseline_stream,
                              monitor_residual_blocks, stream_adapt_v5)
from tests.v2.test_v5 import AFF, ALPHA_SM, C, CER, D, cer_for, data, dcond, dspec, make_model, plast, rn_ctrl

from src.v2.cerebellum import CerebellumV2


def bspec(block: int = 5, hazard: float = 0.01, r_max: int = 40, temper: float = 1.0, p0: float = 0.5,
          source: str = "x") -> BocpdSpec:
    return BocpdSpec(block=block, hazard=hazard, r_max=r_max, temper=temper, kappa_new=1.0, alpha_new=1.0, r_new=4,
                     var_floor=1e-4, p0=p0, score_eps=1e-3, source=source)


def sref(x: torch.Tensor, block: int = 5, source: str = "x") -> ScoreRef:
    """Adım dizisi x: [n, C] → blok ortalamalarından skor referansı."""
    return fit_score_ref(block_means(x, block), source, 1e-3, 1e-4)


# --- §12.15 sönümlü ReFIT -----------------------------------------------------------------------------------------

def test_refit_direction_damping_and_on_target() -> None:
    p = torch.tensor([[0.0, 0.0], [0.0, 0.0], [9.0, 0.0], [10.0, 0.0]])
    g = torch.tensor([[100.0, 0.0], [0.0, 30.0], [10.0, 0.0], [10.0, 0.0]])
    dd = refit_direction(p, g, r_d=20.0, r_on=2.0, eps=1e-9)
    assert torch.allclose(dd[0], torch.tensor([1.0 - math.exp(-5.0), 0.0]), atol=1e-6)   # yön +x, uzakta ≈ 1
    assert torch.allclose(dd[1], torch.tensor([0.0, 1.0 - math.exp(-1.5)]), atol=1e-6)
    assert torch.equal(dd[2], torch.zeros(2)) and torch.equal(dd[3], torch.zeros(2))     # hedefte (ρ ≤ r_on) 0
    assert (dd.norm(dim=1) <= 1.0).all()


def test_reaction_mask() -> None:
    g = torch.tensor([[0.0, 0.0]] * 3 + [[1.0, 0.0]] * 6)
    m = reaction_mask(g, rt_steps=2)
    assert m.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_refit_label_keeps_speed_uses_direction() -> None:
    mu, sd = torch.tensor([0.5, -1.0]), torch.tensor([2.0, 3.0])
    v = torch.tensor([[1.0, 2.0]])                                           # standart ölçek
    dd = torch.tensor([[0.0, 0.5]])
    y = refit_label(v, dd, mu, sd) * sd + mu                                 # fiziksel
    speed = (v * sd + mu).norm()
    assert torch.allclose(y, torch.tensor([[0.0, 0.5 * float(speed)]]), atol=1e-6)


def intent_for(n: int, seed: int = 3) -> Intent:
    g = torch.Generator().manual_seed(seed)
    dd = torch.nn.functional.normalize(torch.randn(n, D, generator=g), dim=1) * 0.8
    m = (torch.rand(n, generator=g) > 0.2).float()
    return Intent(dd, m, torch.zeros(D), torch.ones(D))


def test_refit_stream_ignores_y_and_requires_zero_delay() -> None:
    """ReFIT akışı hız etiketini (y) hiç okumaz (sızıntı yok); gecikme 0 olmalı; maskede güncelleme yok."""
    m = make_model()
    x, y = data(n=200)
    conds = [Condition("B", eta0=0.05)]
    v7 = V7Inputs(intent=intent_for(200))
    a = stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, v7=v7)
    b = stream_adapt_v5(m, x, y * 100.0 + 7.0, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM,
                        1e-4, 1e-4, False, v7=v7)
    assert torch.equal(a.pred, b.pred)
    assert float(a.dw_rel[0]) > 0.0                                          # öğrendi
    v7_off = V7Inputs(intent=Intent(v7.intent.dd, torch.zeros(200), v7.intent.mu, v7.intent.sd))
    c = stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, v7=v7_off)
    assert float(c.dw_rel[0]) == 0.0                                         # m_int = 0 → güncelleme yok
    with pytest.raises(ValueError):
        stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, label_delay=5, v7=v7)


def test_refit_prequential() -> None:
    """t anındaki niyet girdisi değişince v̂_t değişmez, v̂_{t+1} değişir (önce skorla, sonra güncelle)."""
    m = make_model()
    x, y = data(n=200)
    it = intent_for(200)
    conds = [Condition("B", eta0=0.05)]

    def go(i: Intent):
        return stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4,
                               1e-4, False, v7=V7Inputs(intent=i)).pred

    t = 150
    dd2 = it.dd.clone()
    dd2[t] = -dd2[t]
    m2 = it.m.clone()
    m2[t] = 1.0
    m1 = it.m.clone()
    m1[t] = 1.0
    a, b = go(Intent(it.dd, m1, it.mu, it.sd)), go(Intent(dd2, m2, it.mu, it.sd))
    k = t - 80
    assert torch.equal(a[:, : k + 1], b[:, : k + 1]) and not torch.equal(a[:, k + 1:], b[:, k + 1:])


# --- §12.16 BOCPD (tek boyutlu skor + Day-0'a dönüş, KR-050) --------------------------------------------------------

def test_score_ref_and_block_scores() -> None:
    """y = ln(ort_c z² + ε); ölü kanal skora girmez; ar1 yeniliği ilk bloğu düşürür."""
    g = torch.Generator().manual_seed(0)
    o = torch.randn(400, C, generator=g) * 2.0 + 3.0
    ref = fit_score_ref(o, "x", 1e-3, 1e-4)
    assert torch.allclose(ref.m, o.mean(0), atol=1e-5) and torch.allclose(ref.s, o.std(0), atol=1e-4)
    y = block_scores(o, ref, 1e-3)
    assert y.shape == (400,) and abs(float(y.mean()) - ref.m_y) < 1e-4
    q = o[:3].clone()
    q[:, 0] = 1e3
    live = torch.ones(C)
    live[0] = 0.0
    assert torch.allclose(block_scores(q, ref, 1e-3, live), block_scores(o[:3], ref, 1e-3, live))
    z = torch.zeros(400, C)                                                  # güçlü otokorelasyonlu dizi
    for k in range(1, 400):
        z[k] = 0.95 * z[k - 1] + 0.1 * torch.randn(C, generator=g)
    r1 = fit_score_ref(z, "ar1", 1e-3, 1e-6)
    assert r1.ar1 and float(r1.rho.mean()) > 0.8 and float(r1.s.mean()) < 0.5 * float(z.std(0).mean())
    assert block_scores(z, r1, 1e-3).shape == (399,)


def test_bocpd_stationary_low_shift_high() -> None:
    g = torch.Generator().manual_seed(0)
    ref = sref(torch.randn(4000, C, generator=g))
    sp = bspec(temper=0.5)                                                   # tavlamalı (gerçek kullanımda kalibre)
    calm = bocpd_series(block_means(torch.randn(2000, C, generator=g), 5), ref, sp)
    assert float(calm.max()) < 0.5 and float(calm.mean()) < 0.2              # yanlış alarm yok (taban hazard'dan ~0,1)
    xs = torch.cat([torch.randn(500, C, generator=g), torch.randn(1500, C, generator=g) * 1.5 + 0.8])
    a = bocpd_series(block_means(xs, 5), ref, sp)
    assert float(a[:90].max()) < 0.5                                         # değişimden önce sakin
    first = int((a[100:] > 0.9).nonzero()[0, 0])                             # değişim bloğu 100
    assert first <= 5                                                        # birkaç blokta tespit
    assert float(a[120:].min()) > 0.8                                        # kalıcı kayma: kalıcı alarm


def test_bocpd_reentry_recovers_after_transient() -> None:
    """Geçici sapmadan sonra Day-0'a dönüşlü (p_0 > 0) a geri iner; dönüşsüz (p_0 = 0) biçimde yüksek kalır."""
    g = torch.Generator().manual_seed(4)
    ref = sref(torch.randn(4000, C, generator=g))
    o = block_means(torch.randn(1500, C, generator=g), 5)                   # 300 blok
    o[100:104] += 3.0                                                        # 4 blokluk patlama
    a_re = bocpd_series(o, ref, bspec(p0=0.5))
    a_no = bocpd_series(o, ref, bspec(p0=0.0))
    assert float(a_re[100:104].max()) > 0.9 and float(a_no[100:104].max()) > 0.9   # ikisi de görür
    assert bool((a_re[104:120] < 0.5).any())                                 # dönüşlü: birkaç blokta geri iner
    assert float(a_no[104:].min()) > 0.9                                     # dönüşsüz: π_D geri gelmez


def test_bocpd_probabilities_normalised_and_valid_mask() -> None:
    ref = ScoreRef(torch.zeros(C), torch.ones(C), torch.zeros(C), 0.0, 1.0)
    det = BOCPD(ref, bspec(r_max=6))
    one = torch.ones(())
    for k in range(30):                                                      # kesme (r_max) aşılır
        det.update_block(torch.randn(C) + (2.0 if k > 10 else 0.0), torch.ones(C), one)
        tot = float(torch.exp(det.log_pd) + torch.exp(det.log_pr).sum())
        assert abs(tot - 1.0) < 1e-4
    before = (det.log_pd.clone(), det.log_pr.clone(), det.mu.clone(), det.o_prev.clone())
    det.update_block(torch.full((C,), 50.0), torch.ones(C), torch.zeros(()))
    assert all(torch.equal(b_, a_) for b_, a_ in zip(before, (det.log_pd, det.log_pr, det.mu, det.o_prev)))


def test_bocpd_dead_channels_excluded() -> None:
    """Ölü (live = 0) kanaldaki sapma skora girmez."""
    g = torch.Generator().manual_seed(3)
    ref = fit_score_ref(torch.randn(500, C, generator=g), "x", 1e-3, 1e-4)
    det = BOCPD(ref, bspec())
    live = torch.ones(C)
    live[0] = 0.0
    one = torch.ones(())
    for _ in range(20):
        o = torch.randn(C, generator=g)                                      # canlı kanallar Day-0 dağılımında
        o[0] = -30.0
        det.update_block(o, live, one)
    assert float(det.a) < 0.1
    det.reset_state()
    for _ in range(20):
        o = torch.randn(C, generator=g)
        o[0] = -30.0
        det.update_block(o, torch.ones(C), one)                              # kanal canlı sayılırsa alarm
    assert float(det.a) > 0.9


def test_bocpd_ar1_first_block_only_fills_previous() -> None:
    """ar1: ilk tam blok yalnız o_prev'i doldurur (skor yok); sonraki bloklar block_scores ile aynı skoru verir."""
    g = torch.Generator().manual_seed(6)
    o = torch.randn(50, C, generator=g)
    ref = fit_score_ref(o, "ar1", 1e-3, 1e-4)
    det = BOCPD(ref, bspec(source="ar1"))
    one = torch.ones(())
    det.update_block(o[0], torch.ones(C), one)
    assert float(det.a) == 0.0 and float(det.has_prev) == 1.0 and torch.equal(det.o_prev, o[0])
    ys = block_scores(o[:6], ref, 1e-3)
    for k in range(1, 6):
        det.update_block(o[k], torch.ones(C), one)
        assert float(det.y) == pytest.approx(float(ys[k - 1]), abs=1e-5)
    with pytest.raises(ValueError):
        BOCPD(ref, bspec(source="x"))                                        # referans / kaynak uyuşmazlığı


def test_bocpd_step_blocks_and_mask() -> None:
    """step: yalnız blok sonunda günceller; maskeli (burn-in) adımlı blok sayılmaz."""
    ref = ScoreRef(torch.zeros(C), torch.ones(C), torch.zeros(C), 0.0, 1.0)
    det = BOCPD(ref, bspec(block=4))
    for i in range(8):
        det.step(torch.full((C,), 9.0), torch.zeros(()), torch.ones(C), (i + 1) % 4 == 0)
    assert float(det.a) == 0.0
    for i in range(4):
        det.step(torch.full((C,), 9.0), torch.ones(()), torch.ones(C), (i + 1) % 4 == 0)
        if i < 3:
            assert float(det.a) == 0.0
    assert float(det.a) > 0.0


def test_calibrate_temper_picks_most_sensitive_allowed() -> None:
    g = torch.Generator().manual_seed(1)
    o = block_means(torch.randn(3000, C, generator=g), 10)
    ref = fit_score_ref(o[:200], "x", 1e-3, 1e-4)
    cfg = {"time": {"dt_ms": 4.0}, "bocpd": {"block_ms": 40.0, "hazard_s": 60.0, "r_max": 50, "temper": 1.0,
                                             "kappa_new": 1.0, "alpha_new": 1.0, "recent_s": 1.0, "var_floor": 1e-4,
                                             "reentry_p0": 0.5, "score_eps": 1e-3, "source": "x",
                                             "temper_grid": [0.01, 0.1, 1.0], "calib_false_alarm": 0.05}}
    t, rows = calibrate_temper(o[200:], ref, cfg)
    assert t in (0.01, 0.1, 1.0) and len(rows) == 3
    assert all(fa <= 0.05 for tt, fa in rows if tt == t) or t == 0.01


# --- §12.17 ölü kanal ----------------------------------------------------------------------------------------------

def test_dead_channel_detector() -> None:
    z0 = torch.full((3,), -1.0)
    p = torch.tensor([0.5, 0.01, 0.0])
    th = silence_threshold(p, 1e-3)
    assert math.isclose(float(th[0]), math.log(1e3) / math.log(2.0), rel_tol=1e-5) and math.isinf(float(th[2]))
    det = DeadChannelDetector(z0, p, 1e-3, 1e-6)
    x_dead = torch.tensor([-1.0, -1.0, -1.0])
    for _ in range(int(math.ceil(float(th[0])))):
        d = det.step(x_dead)
    assert d.tolist() == [1.0, 0.0, 0.0]                                     # seyrek kanal henüz değil, p = 0 hiç
    d = det.step(torch.tensor([2.0, -1.0, -1.0]))                            # spike → hemen açılır
    assert d[0] == 0.0
    assert torch.allclose(spike_prob(torch.tensor([[-1.0, 0.3], [0.5, -1.0]]), torch.tensor([-1.0, -1.0]), 1e-6),
                          torch.tensor([0.5, 0.5]))


def test_dead_channel_in_stream_neutral_and_causal() -> None:
    """Ölü sayılan kanal Day-0 ortalamasına çekilir; diğer koşullar etkilenmez; gelecek girdi geçmişi değiştirmez."""
    m = make_model()
    x, y = data(n=260)
    z0 = torch.full((C,), -5.0)
    x[150:, 1] = -5.0                                                        # kanal 1 susar
    det = DeadChannelDetector(z0, torch.full((C,), 0.5), 1e-3, 1e-6)
    conds = [Condition("B", eta0=0.05), Condition("Bd", eta0=0.05, dead=True)]
    ctrl = rn_ctrl(x[:80])

    def go(xx):
        return stream_adapt_v5(m, xx, y, 80, 260, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4,
                               1e-4, False, controls=ctrl, v7=V7Inputs(dead=det))

    r = go(x)
    ref = stream_adapt_v5(m, x, y, 80, 260, 30, [Condition("B", eta0=0.05)], cer_for(m), AffectSpaceV2(AFF), plast(),
                          ALPHA_SM, 1e-4, 1e-4, False, controls=ctrl)
    assert torch.equal(r.pred[0], ref.pred[0])                               # dedektörsüz koşul aynı
    nd = r.extra["n_dead"][1]
    assert float(nd[: 150 - 80].max()) == 0.0 and float(nd[-1]) == 1.0
    assert not torch.equal(r.pred[1, -50:], r.pred[0, -50:])
    x2 = x.clone()
    x2[200:] = 3.0
    r2 = go(x2)
    assert torch.equal(r.pred[:, : 200 - 80], r2.pred[:, : 200 - 80])


# --- §12.18 Degenhart -----------------------------------------------------------------------------------------------

def synth_fa(n: int, lam: torch.Tensor, psi: torch.Tensor, mu: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    z = torch.randn(n, lam.shape[1], generator=g)
    return z @ lam.T + mu + torch.randn(n, lam.shape[0], generator=g) * psi.sqrt()


def test_fa_procrustes_and_stable_channels() -> None:
    g = torch.Generator().manual_seed(2)
    Cn, q = 30, 3
    lam0 = torch.randn(Cn, q, generator=g)
    psi = torch.full((Cn,), 0.1)
    fa0 = fa_fit(synth_fa(20000, lam0, psi, torch.zeros(Cn), g), q, 200, 1e-4)
    # yeni gün: aynı gizil uzay döndürülmüş; son 6 kanalın yüklemeleri değişmiş (kararsız)
    R, _ = torch.linalg.qr(torch.randn(q, q, generator=g))
    lamk = lam0 @ R
    lamk[-6:] = torch.randn(6, q, generator=g)
    fak = fa_fit(synth_fa(20000, lamk, psi, torch.full((Cn,), 0.5), g), q, 200, 1e-4)
    O, mask = stable_alignment(fa0.lam, fak.lam, torch.ones(Cn), 0.6, 5)
    assert torch.allclose(O @ O.T, torch.eye(q), atol=1e-4)
    assert float(mask[-6:].sum()) == 0.0 and float(mask.sum()) == math.ceil(0.6 * Cn)
    assert float((fa0.lam[mask > 0] - fak.lam[mask > 0] @ O).norm() / fa0.lam[mask > 0].norm()) < 0.15
    A, b = channel_map(fa0, fak, O)
    assert A.shape == (Cn, Cn) and b.shape == (Cn,)
    I = torch.eye(5)
    assert torch.allclose(procrustes(I[:, :3] @ torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]]), I[:, :3]),
                          torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-6)
    assert latent_map(fa0).shape == (q, Cn)


def test_aligner_schedule_causal_and_gated() -> None:
    g = torch.Generator().manual_seed(4)
    Cn, q = 12, 2
    lam0 = torch.randn(Cn, q, generator=g)
    x = synth_fa(3000, lam0, torch.full((Cn,), 0.2), torch.zeros(Cn), g)
    fa0 = fa_fit(x[:1000], q, 50, 1e-4)
    sp = AlignSpec(q=q, fa_bin=5, t_cal=200, t_refit=100, t_win=600, stable_frac=0.6, n_iter=3, em_iters=30,
                   psi_floor=1e-4, a_on=0.5)
    al = Aligner(fa0, sp, gated=True, device=torch.device("cpu"))
    live = torch.ones(Cn)
    assert not al.maybe_refit(x, 1000, 0, 0.1, live) and al.t_on is None    # kapılı: a < a_on
    assert not al.maybe_refit(x, 1100, 0, 0.9, live) and al.t_on == 1100    # açıldı, T_cal dolmadı
    assert al.maybe_refit(x, 1300, 0, 0.9, live) and float(al.ready) == 1.0
    A1 = al.A.clone()
    x2 = x.clone()
    x2[1300:] = 99.0                                                         # gelecek girdi haritayı etkilemez
    al2 = Aligner(fa0, sp, gated=True, device=torch.device("cpu"))
    al2.maybe_refit(x2, 1100, 0, 0.9, live)
    al2.maybe_refit(x2, 1300, 0, 0.9, live)
    assert torch.allclose(A1, al2.A)
    assert not al.maybe_refit(x, 1350, 0, 0.9, live)                         # T_refit dolmadı
    un = Aligner(fa0, sp, gated=False, device=torch.device("cpu"))
    assert un.maybe_refit(x, 200, 0, 0.0, live) and un.t_on == 0            # mod 2: akış başından


def test_alignment_in_stream_uses_reconstruction() -> None:
    """Hazır hizalamada kararsız kanallar A·x + b'den gelir; kararlı kanallar yeniden ölçekleme yolundan."""
    m = make_model()
    x, y = data(n=400)
    fa0 = fa_fit(x[:200], 2, 30, 1e-4)
    sp = AlignSpec(q=2, fa_bin=2, t_cal=60, t_refit=10 ** 6, t_win=10 ** 6, stable_frac=0.6, n_iter=2, em_iters=20,
                   psi_floor=1e-4, a_on=0.5)
    al = Aligner(fa0, sp, gated=False, device=torch.device("cpu"))
    conds = [Condition("B", eta0=0.05), Condition("G", eta0=0.05, align=2)]
    r = stream_adapt_v5(m, x, y, 250, 400, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, v7=V7Inputs(aligners={2: al}, align_check=20))
    assert r.extra["align_fits"][2] == 1
    assert float(al.ready) == 1.0 and float(al.stable.sum()) < C
    assert not torch.equal(r.pred[0], r.pred[1])
    with pytest.raises(ValueError):                                          # yuvası olmayan mod
        stream_adapt_v5(m, x, y, 250, 400, 30, [Condition("G", align=1)], cer_for(m), AffectSpaceV2(AFF), plast(),
                        ALPHA_SM, 1e-4, 1e-4, False, v7=V7Inputs(aligners={2: al}))


# --- BOCPD sensörü akışta --------------------------------------------------------------------------------------------

def test_bocpd_sensor_drives_gate_and_model_unchanged() -> None:
    m = make_model()
    x, y = data(n=600)
    x[400:] = x[400:] * 2.0 + 3.0
    det = BOCPD(sref(x[:300]), bspec(block=5))
    ctrl = rn_ctrl(x[:300])
    lam_c, lam_s = 0.0, 0.2
    conds = [Condition("A"), Condition("P7", eta0=0.05, renorm=True, lam_calm=lam_c, lam_surp=lam_s, affect=True,
                                       sensor="bocpd", gate_lo=0.0, gate_hi=1.0)]
    w0 = m.backbone.w_in.weight.clone()
    r = stream_adapt_v5(m, x, y, 300, 600, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=ctrl, v7=V7Inputs(bocpd=det))
    ab = r.extra["a_bocpd"][1]
    assert float(ab[: 100 - 5].max()) < 0.5 and float(ab[130:].min()) > 0.9
    assert torch.allclose(r.a[1], ab)                                        # sensör arousal'ı = BOCPD a
    assert float(r.lam[1, -1]) == pytest.approx(lam_s, rel=1e-2)              # dönüşlü BOCPD: a ≤ ~1 − H·p_0
    assert torch.equal(m.backbone.w_in.weight, w0)                           # KV1
    with pytest.raises(ValueError):
        stream_adapt_v5(m, x, y, 300, 600, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=ctrl)


def test_v7_defaults_equal_v5() -> None:
    m = make_model()
    x, y = data()
    conds = [Condition("B", eta0=0.05), Condition("R", eta0=0.05, renorm=True, lam_calm=0.02)]
    ctrl = rn_ctrl(x[:80])
    a = stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=ctrl, label_delay=5)
    b = stream_adapt_v5(m, x, y, 80, 200, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=ctrl, label_delay=5, v7=V7Inputs())
    assert torch.equal(a.pred, b.pred)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
def test_v7_graph_matches_eager() -> None:
    dev = torch.device("cuda")
    x, y = data(n=600)
    x[400:, :2] = -4.0
    x[350:] = x[350:] * 1.5 + 1.0
    x, y = x.to(dev), y.to(dev)
    m = make_model().to(dev)
    ref = sref(x[:300]).to(dev)
    fa0 = fa_fit(x[:300], 2, 20, 1e-4)
    sp = AlignSpec(q=2, fa_bin=2, t_cal=40, t_refit=40, t_win=400, stable_frac=0.6, n_iter=2, em_iters=20,
                   psi_floor=1e-4, a_on=0.5)
    ctrl = rn_ctrl(x[:300])
    it = intent_for(600)
    it = Intent(it.dd.to(dev), it.m.to(dev), it.mu.to(dev), it.sd.to(dev))
    conds = [Condition("A"), Condition("B", eta0=0.05),
             Condition("D7", eta0=0.05, renorm=True, lam_calm=0.0, lam_surp=0.2, affect=True, gate=True, alpha_eta=2.0,
                       sensor="bocpd", dead=True, align=1),
             Condition("G7", eta0=0.05, align=2)]

    def go(chunk: int):
        v7 = V7Inputs(bocpd=BOCPD(ref, bspec(block=5)),
                      dead=DeadChannelDetector(torch.full((C,), -4.0, device=dev), torch.full((C,), 0.3, device=dev),
                                               1e-3, 1e-6),
                      aligners={1: Aligner(fa0, sp, True, dev), 2: Aligner(fa0, sp, False, dev)}, intent=it,
                      align_check=40)
        return stream_adapt_v5(m, x, y, 300, 600, 40, conds, CerebellumV2(torch.zeros(C, m.backbone.N + D, device=dev),
                                                                            0.1, CER),
                               AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-3, 1e-4, True, chunk=chunk, controls=ctrl, v7=v7)

    e, g = go(0), go(40)
    for f in ("pred", "a", "eta", "lam", "dw_rel", "rate"):
        assert torch.allclose(getattr(e, f), getattr(g, f), atol=1e-4), f
    assert torch.allclose(e.extra["a_bocpd"], g.extra["a_bocpd"], atol=1e-5)
    assert e.extra["align_fits"] == g.extra["align_fits"]


# --- §12.19 augmentation --------------------------------------------------------------------------------------------

def test_augment_disabled_identity_and_drop() -> None:
    x = 3.0 * torch.tanh(torch.randn(8, 20, 10) / 3.0)                      # x_norm aralığı (|x| < kırpma)
    z0 = torch.full((8, 10), -2.0)
    off = AugSpec(False, 0, 0, 0, 0, 0, 3.0)
    assert augment_windows(x, z0, off, torch.Generator().manual_seed(0)) is x
    sp = AugSpec(True, 1.0, 0.3, 0.0, 0.0, 0.0, 3.0)
    out = augment_windows(x, z0, sp, torch.Generator().manual_seed(0))
    dropped = (out == -2.0).all(1)                                           # [B, C] tüm pencere boyunca
    assert dropped.any(1).all() and int(dropped.sum(1).max()) <= math.ceil(0.3 * 10)
    keep = ~dropped
    assert torch.equal(out.transpose(1, 2)[keep], x.transpose(1, 2)[keep])  # dokunulmayan kanallar aynı
    a = augment_windows(x, z0, AugSpec(True, 0.5, 0.3, 0.5, 0.2, 0.2, 3.0), torch.Generator().manual_seed(7))
    b = augment_windows(x, z0, AugSpec(True, 0.5, 0.3, 0.5, 0.2, 0.2, 3.0), torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and (a.abs() < 3.0).all()


def test_block_and_bin_means() -> None:
    x = torch.arange(12.0).view(6, 2)
    assert torch.equal(block_means(x, 3), torch.tensor([[2.0, 3.0], [8.0, 9.0]]))
    assert torch.equal(bin_means(x, 4), torch.tensor([[3.0, 4.0]]))


# --- §12.16 cer_mon kaynağı ve §12.20 tonik / fazik iş bölümü (KR-050) ------------------------------------------------

def wb_for(m) -> torch.Tensor:
    """Küçük modelde izleme yolu ridge'i (W_b): x̄(t+1) ~ φ(t)."""
    x, _ = data(n=300)
    bar, vp, _, r = collect_streams(m, x, 40, 300, 2, 20, ALPHA_SM, keep_r=True)
    w, _ = CerebellumV2.fit(torch.cat([r, vp], 1)[:-1], bar[1:], 1e-3)
    return w


def test_collect_streams_matches_single_stream() -> None:
    """Çok akışlı toplayıcı, akış içindeki adımlarda tek akışla aynı φ ve x̄'i verir (yeterli burn-in)."""
    m = make_model()
    x, _ = data(n=300)
    b1, v1, _, r1 = collect_streams(m, x, 100, 300, 1, 100, ALPHA_SM, keep_r=True)
    b2, v2, P, r2 = collect_streams(m, x, 100, 300, 2, 100, ALPHA_SM, proj=torch.eye(m.backbone.N)[:3], keep_r=True)
    assert b1.shape == b2.shape == (200, C) and r2.shape == (200, m.backbone.N) and P.shape == (200, 3)
    assert torch.allclose(P, r2[:, :3], atol=1e-6)
    assert torch.allclose(b1[:100], b2[:100], atol=1e-6)                    # ilk akış aynı başlangıçtan


def test_cer_mon_source_requires_inputs_and_is_causal() -> None:
    m = make_model()
    x, y = data(n=600)
    w_b = wb_for(m)
    o = monitor_residual_blocks(m, x, w_b, 60, 300, 5, 2, 20, ALPHA_SM, 1000)   # Day-0 kısmının artığı
    ref = fit_score_ref(o, "cer_mon", 1e-3, 1e-6)
    conds = [Condition("A"), Condition("P", eta0=0.05, renorm=True, lam_surp=0.2, affect=True, sensor="bocpd")]

    def go(xx, **v7kw):
        det = BOCPD(ref, bspec(block=5, source="cer_mon", temper=0.3))
        return stream_adapt_v5(m, xx, y, 300, 600, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4,
                               1e-4, False, controls=rn_ctrl(x[:300]), v7=V7Inputs(bocpd=det, **v7kw))

    with pytest.raises(ValueError):
        go(x)                                                                # W_b yok
    a = go(x, w_b=w_b).extra["a_bocpd"][1]
    x2 = x.clone()
    x2[500:] = x2[500:] * 3.0 + 2.0
    b = go(x2, w_b=w_b).extra["a_bocpd"][1]
    k = 500 - 300
    assert torch.equal(a[:k], b[:k])                                         # nedensel
    assert float(b[k + 30:].mean()) > float(a[k + 30:].mean())               # kaymayı görür
    with pytest.raises(ValueError):                                          # koşul 0 donuk A değil
        stream_adapt_v5(m, x, y, 300, 600, 30, conds[::-1], cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4,
                        1e-4, False, controls=rn_ctrl(x[:300]),
                        v7=V7Inputs(bocpd=BOCPD(ref, bspec(block=5, source="cer_mon")), w_b=w_b))


def test_copy_norm_is_scale_free() -> None:
    """Kopya oranı: hata ve kopya hatası aynı k² ile ölçeklenince arousal değişmez; ‖e‖²/ε²_0 biçimi değişir."""
    o, one = torch.ones(2), torch.ones(())

    def run(copy: float, k: float) -> float:
        ad = AffectDynamics(o * 0.0, o * 0.1, o * 0.2, o * 0.02, o * 0.02, 1e-6)
        ad.set_internal(o * copy, o * (2.0 if copy else 0.5), o * 0.3)       # r_0 = 2 (kopya) ya da ε²_0 = 0,5
        for _ in range(300):
            ad.update_arousal(o * 1.2 * k ** 2, o, one, o * 0.3 * k ** 2)    # e² = 1,2·k², Δx̄² = 0,3·k²
        return float(ad.a[0])

    assert run(1.0, 1.0) == pytest.approx(run(1.0, 0.5), abs=1e-6)          # e²/Δx̄² = 4 = 2·r_0 → a = tanh(1)
    assert run(1.0, 1.0) == pytest.approx(math.tanh(1.0), abs=1e-3)
    assert run(0.0, 1.0) == pytest.approx(math.tanh(1.4), abs=1e-3)          # mutlak: 1,2/0,5 = 2,4
    assert run(0.0, 0.5) < 0.0                                               # aynı düzen, k = 0,5 → ölçekle değişir


def test_abs_path_equals_previous_arousal() -> None:
    """d1 verilse de ‖e‖²/ε²_0 biçimi (§12.6) eski tek-EMA'lı arousal'la aynıdır (doğrusal EMA)."""
    o, one = torch.ones(1), torch.ones(())
    old = AffectDynamics(o * 0.4, o * 0.1, o * 0.2, o * 0.02, o * 0.02, 1e-6)
    new = AffectDynamics(o * 0.4, o * 0.1, o * 0.2, o * 0.02, o * 0.02, 1e-6)
    new.set_internal(o * 0.0, o * 0.25, o * 7.0)                             # ε²_0 = 0,25; k0 yok sayılır
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        e2, u, d1 = torch.rand(1, generator=g), torch.rand(1, generator=g) * 2, torch.rand(1, generator=g)
        a1 = old.update_arousal(e2 / 0.25, u, one)
        a2 = new.update_arousal(e2, u, one, d1)
        assert torch.allclose(a1, a2, atol=1e-5)


def test_routed_controls_split() -> None:
    """η fazikten, α (gevşetme) tonikten; aynı arousal verilince `controls` ile aynı."""
    z, o = torch.zeros(2), torch.ones(2)
    a_ph, a_ton = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    d = -o
    eta, alpha, b, c = AffectDynamics.routed_controls(a_ph, a_ton, z, d, z, o * 1e-3, o * 2.0, z, o * 1e-4, o * 1e-3,
                                                      o, o, o, o)
    assert torch.allclose(eta, torch.tensor([3e-3, 1e-3])) and torch.allclose(alpha, torch.tensor([1e-4, 0.0]))
    ref = AffectDynamics.controls(a_ph, z, d, z, o * 1e-3, o * 2.0, z, o * 1e-4, o * 1e-3, o, o, o, o)
    same = AffectDynamics.routed_controls(a_ph, a_ph, z, d, z, o * 1e-3, o * 2.0, z, o * 1e-4, o * 1e-3, o, o, o, o)
    assert all(torch.equal(p_, q_) for p_, q_ in zip(ref, same))


def split_setup():
    m = make_model()
    x, y = data(n=600)
    x[400:] = x[400:] * 2.0 + 1.0
    base = dyn_baseline_stream(m, x, y, 100, 300, 30, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4)
    det = BOCPD(sref(x[:300]), bspec(block=5))
    ph = dict(a_mode=3, ph_norm=1)
    kw = dict(sensor="bocpd", gate_lo=0.0, gate_hi=1.0, cer_eta=1e-3, drift_max=False)
    conds = [Condition("A"), dcond("S", dspec(**ph, brake=False, consolidate=False, rho_v=0.0), **kw),
             dcond("S0", dspec(**ph, ph_on=False, brake=False, consolidate=False, rho_v=0.0), **kw),
             dcond("N0", dspec(a_mode=3, ph_norm=0, brake=False, consolidate=False, rho_v=0.0), **kw)]
    return m, x, y, base, det, conds


def test_split_stream_bounds_and_phasic_off() -> None:
    """İş bölümü: η ∈ [η_0, η_0(1 + α_η)], α ∈ [0, α_0]; fazik kol kapalıyken η = η_0; λ_norm tonikten (KV4)."""
    m, x, y, base, det, conds = split_setup()
    assert base.r0 is not None and base.r0 > 0 and base.d0 is not None and base.d0 > 0
    w0 = m.backbone.w_in.weight.clone()
    r = stream_adapt_v5(m, x, y, 300, 600, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=rn_ctrl(x[:300]), dyn_base=base, v7=V7Inputs(bocpd=det))
    eta0, top = 0.05, 0.05 * 3.0
    for e in (1, 2, 3):
        assert float(r.eta[e].min()) >= eta0 - 1e-9 and float(r.eta[e].max()) <= top + 1e-9
        assert float(r.extra["alpha"][e].max()) <= 1e-4 + 1e-12 and float(r.extra["alpha"][e].min()) >= 0.0
        assert torch.allclose(r.a[e], r.extra["a_bocpd"][e])                 # tonik arousal = BOCPD
    assert torch.allclose(r.eta[2], torch.full_like(r.eta[2], eta0))         # fazik kapalı → gaz yok
    assert float(r.extra["a_int"][2].abs().max()) == 0.0
    assert torch.allclose(r.lam[1], r.lam[2])                                # λ_norm yalnız tonikten
    assert torch.equal(m.backbone.w_in.weight, w0)                           # KV1
    with pytest.raises(ValueError):                                          # kopya oranı r_0 ister
        stream_adapt_v5(m, x, y, 300, 600, 30, conds, cer_for(m), AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-4, 1e-4,
                        False, controls=rn_ctrl(x[:300]), dyn_base=DynBase(base.u0, base.q0),
                        v7=V7Inputs(bocpd=BOCPD(sref(x[:300]), bspec(block=5))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
def test_split_and_cer_mon_graph_matches_eager() -> None:
    dev = torch.device("cuda")
    m0, x, y, base, _, conds = split_setup()
    w_b = wb_for(m0).to(dev)
    m = m0.to(dev)
    x, y = x.to(dev), y.to(dev)
    ref = fit_score_ref(torch.randn(200, C) * 0.05, "cer_mon", 1e-3, 1e-6).to(dev)

    def go(chunk: int):
        det = BOCPD(ref, bspec(block=5, source="cer_mon"))
        return stream_adapt_v5(m, x, y, 300, 600, 40, conds,
                               CerebellumV2(torch.zeros(C, m.backbone.N + D, device=dev), 0.1, CER),
                               AffectSpaceV2(AFF), plast(), ALPHA_SM, 1e-3, 1e-4, True, chunk=chunk,
                               controls=rn_ctrl(x[:300]), dyn_base=base, v7=V7Inputs(bocpd=det, w_b=w_b))

    e, g = go(0), go(40)
    for f in ("pred", "a", "eta", "lam", "dw_rel"):
        assert torch.allclose(getattr(e, f), getattr(g, f), atol=1e-4), f
    for k in ("a_bocpd", "a_int", "e2", "d1"):
        assert torch.allclose(e.extra[k], g.extra[k], atol=1e-4), k
