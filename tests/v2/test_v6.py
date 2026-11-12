"""Faz V6 yardımcıları: kayan pencere R², toparlanma süresi, elektrot kaybı (experiments/v2_v6.py, §12.9 v2-E)."""

import copy
from pathlib import Path

import torch

from src.v2.metrics import r2_columns
from experiments.v2_v6 import Ctx, drop_channels, recovery, rolling_r2
from src.data.primate import Session
from src.utils.config import load_config
from src.v2.encoder import SpikeEncoderV2
from src.v2.online_v5 import V5Result
from src.v2.train import prepare_session

CFG_DIR = Path(__file__).resolve().parents[2] / "configs"
BASE = load_config(CFG_DIR / "v2_base.yaml")


def test_conditions_sensor_and_matching() -> None:
    """D'nin sensörü Pm ile aynı ayarlar; iç beyincik ayrı yol; O yalnız Day-k'de açık; 2. geçiş eşleme ve karıştırma."""
    cfg = load_config(CFG_DIR / "v2_v6.yaml")
    cfg["v6"]["fixed"] = ["A", "B", "R5", "Pm", "Pm3", "O", "Bg", "Pg"]
    cfg["v6"]["variants"] = {"D": {}, "D_nu": {"arousal": "internal", "w_u": 0.0, "brake": True, "consolidate": True,
                                               "rho_v": 0.5}, "Ds": {"arousal": "sensor"},
                             "Dsd": {"arousal": "sensor", "gas_dom": True}, "Dx": {"arousal": "max", "w_u": 0.0},
                             "Ds_nv": {"arousal": "sensor", "brake": False, "consolidate": False, "rho_v": 0.0},
                             "Ds_ng": {"arousal": "sensor", "gate": False}}
    cfg["v6"]["matched"] = ["D", "Ds"]
    ctx = Ctx(cfg, "val")
    c0, c1 = ctx.pass1("crossday", 0), ctx.pass1("crossday", 1)
    names = [c.name for c in c1]
    assert names[:8] == ["A", "B", "R5", "Pm", "Pm3", "O", "Bg", "Pg"] and "D" in names and "Ds" in names
    gm = cfg["v6"]["global_matched"]
    bg, pg = c1[names.index("Bg")], c1[names.index("Pg")]
    assert bg.eta0 == pg.eta0 == gm["eta"] and bg.alpha_anchor == pg.alpha_anchor == gm["alpha"] and pg.cer_src == 0
    o0, o1 = c0[names.index("O")], c1[names.index("O")]
    assert o0.lam_calm == 0.0 and o1.lam_calm > 0.0
    pm, d = c1[names.index("Pm")], c1[names.index("D")]
    for f in ("renorm", "lam_calm", "lam_surp", "affect", "fixed_slow", "a_fast", "drift_max", "drift_w", "gate_lo",
              "gate_hi", "lam_return"):
        assert getattr(pm, f) == getattr(d, f), f
    assert pm.lam_calm == 0.0 and pm.cer_src == 0 and pm.cer_raw and pm.dyn is None
    assert d.cer_src == -1 and not d.cer_raw and d.dyn is not None and d.gate
    assert c1[names.index("Pm3")].eta0 == 3.0 * pm.eta0 and c1[names.index("Ds_ng")].gate is False
    assert c1[names.index("Ds_nv")].dyn.brake is False and c1[names.index("D_nu")].dyn.w_u == 0.0
    assert d.dyn.a_mode == 1 and c1[names.index("Ds")].dyn.a_mode == 1 and c1[names.index("Dx")].dyn.a_mode == 2
    assert c1[names.index("Dsd")].dyn.gas_dom and not d.dyn.gas_dom
    assert not d.dyn.brake and not d.dyn.consolidate and d.dyn.rho_v == 0.0          # val seçimi: valens kontrolü yok
    ui = c1[names.index("D_nu")]                                               # internal ancak açıkça istenirse
    assert ui.dyn.a_mode == 0 and ui.dyn.brake and ui.dyn.consolidate and ui.dyn.rho_v == 0.5
    E, L = len(c1), 50
    g = torch.Generator().manual_seed(0)
    res = V5Result(torch.zeros(E, L, 2), torch.zeros(E, 4), torch.zeros(E), torch.zeros(E), torch.zeros(E),
                   torch.zeros(E, L), torch.rand(E, L, generator=g), torch.zeros(E, L), torch.zeros(E, L),
                   {"alpha": torch.rand(E, L, generator=g), "cons": torch.rand(E, L, generator=g)})
    c2 = ctx.pass2(c1, res, seed=100)
    n2 = [c.name for c in c2]
    assert n2 == ["A", "Bm_D", "P_D", "C_D", "Bm_Ds", "P_Ds", "C_Ds"]
    i = names.index("D")
    bm, p, cc = c2[1], c2[2], c2[3]
    assert bm.eta0 == float(res.eta[i].mean()) and bm.alpha_anchor == float(res.extra["alpha"][i].mean())
    assert p.eta0 == bm.eta0 and p.alpha_anchor == bm.alpha_anchor and p.cer_src == 0 and p.lam_surp == pm.lam_surp
    assert torch.equal(cc.eta_seq.sort().values, res.eta[i].sort().values)  # karıştırma: aynı dağılım
    assert torch.equal(cc.cons_seq.sort().values, res.extra["cons"][i].sort().values)
    assert not torch.equal(cc.eta_seq, res.eta[i]) and cc.dyn is not None


def test_rolling_r2_matches_direct() -> None:
    g = torch.Generator().manual_seed(0)
    y = torch.randn(300, 2, generator=g)
    p = y + 0.5 * torch.randn(300, 2, generator=g)
    st, r = rolling_r2(p, y, 50, 25)
    assert list(st) == list(range(0, 251, 25))
    for s, v in zip(st, r):
        assert abs(sum(r2_columns(p[s: s + 50], y[s: s + 50])) / 2 - v) < 1e-9


def test_recovery_time_and_censoring() -> None:
    g = torch.Generator().manual_seed(1)
    y = torch.randn(1000, 2, generator=g)
    p = y + 0.1 * torch.randn(1000, 2, generator=g)
    bad = p.clone()
    bad[500:700] = torch.randn(200, 2, generator=g)                          # kayıptan 200 adım sonra toparlanma
    rec, cens, ref, _, _ = recovery(bad, y, 500, 50, 10, 0.05, 4.0)
    assert not cens and ref > 0.9
    assert 0.8 <= rec <= 1.0                                                 # ilk temiz pencere: 700 + 50 adım − 500 → 1,0 s
    bad[500:] = torch.randn(500, 2, generator=g)
    rec, cens, _, _, _ = recovery(bad, y, 500, 50, 10, 0.05, 4.0)
    assert cens and rec == 2.0                                               # dilim sonuna kadar (500 adım × 4 ms)


def test_drop_channels_and_encoding() -> None:
    g = torch.Generator().manual_seed(2)
    n, C = 2000, 12
    s = Session("indy_20160622_01", torch.poisson(torch.full((n, C), 2.0), generator=g), torch.randn(n, 2, generator=g),
                1600, 1800)
    s2, ch = drop_channels(s, 1900, 4, seed=7)
    assert len(ch) == 4 and ch == drop_channels(s, 1900, 4, seed=7)[1]     # seed'e bağlı, tekrarlanabilir
    assert torch.equal(s2.x[:1900], s.x[:1900]) and torch.all(s2.x[1900:, ch] == 0)
    keep = [c for c in range(C) if c not in ch]
    assert torch.equal(s2.x[:, keep], s.x[:, keep]) and not torch.all(s.x[1900:, ch] == 0)
    cfg = copy.deepcopy(BASE)
    a, b = prepare_session(cfg, s, torch.device("cpu")), prepare_session(cfg, s2, torch.device("cpu"))
    assert torch.equal(a.x[:1900], b.x[:1900])                               # kodlayıcı istatistikleri train'den
    enc = SpikeEncoderV2.from_config(cfg, C)
    enc.fit(s.x[:1600])
    zero = enc.normalize(torch.zeros(1, C))[0, ch]                           # "0 sayım" kodu
    assert torch.allclose(b.x[1900:, ch], zero.expand(100, -1))
