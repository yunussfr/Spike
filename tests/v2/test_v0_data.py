"""Faz V0 testleri: 4 ms veri, zaman ayarları, pencereler, günler arası sıra, nedensel Wiener öznitelikleri."""

import copy
from pathlib import Path

import pytest
import torch

from experiments.v2_ceiling import block_features
from src.data.primate import Session
from src.utils.config import load_config
from src.v2.data import (check_time, check_tune_disjoint, crossday_sessions, gather_windows, load_v2_session, ms_to_steps,
                         train_window_starts)

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs" / "v2_base.yaml")
CACHE = ROOT / CFG["data"]["cache_dir"]
NAME = "indy_20170131_02"                                                    # en kısa oturum
has_caches = (CACHE / f"{NAME}_b4_s4_r0.8.pt").exists() and (CACHE / f"{NAME}_b20_s20_r0.8.pt").exists()


def fake_session(n: int = 3000, val_start: int = 2000, test_start: int = 2500, D: int = 3) -> Session:
    g = torch.Generator().manual_seed(0)
    return Session("fake_20200101_01", torch.randint(0, 2, (n, D), generator=g).float(),
                   torch.randn(n, 2, generator=g), val_start, test_start)


def test_check_time_rejects_mismatch() -> None:
    assert check_time(CFG) == 4.0
    bad = copy.deepcopy(CFG)
    bad["data"]["stride"] = 0.008
    with pytest.raises(ValueError):
        check_time(bad)


def test_ms_to_steps() -> None:
    assert ms_to_steps(2000.0, 4.0) == 500
    assert ms_to_steps(1000.0, 8.0) == 125
    with pytest.raises(ValueError):
        ms_to_steps(10.0, 4.0)                                               # tam kat değil


def test_train_windows_stay_in_train() -> None:
    s = fake_session()
    starts = train_window_starts(s, CFG)
    L = ms_to_steps(CFG["windows"]["length_ms"], 4.0)
    S = ms_to_steps(CFG["windows"]["stride_ms"], 4.0)
    assert int(starts[0]) == 0
    assert bool((starts[1:] - starts[:-1] == S).all())
    assert int(starts[-1]) + L <= s.val_start                                # val'e taşma yok
    assert int(starts[-1]) + S + L > s.val_start                             # sığan son pencere alındı
    w = gather_windows(s.x, starts, L)                                       # [W, L, D]
    assert w.shape == (len(starts), L, s.D_in)
    assert torch.equal(w[1, 0], s.x[S])


def test_crossday_order() -> None:
    for primate in CFG["primates"]:
        day0, rest = crossday_sessions(CFG, primate)
        assert day0 == CFG["primates"][primate][0] and len(rest) == 2
    bad = copy.deepcopy(CFG)
    bad["primates"]["indy"] = list(reversed(bad["primates"]["indy"]))
    with pytest.raises(ValueError):
        crossday_sessions(bad, "indy")


def test_block_features_causal_and_exact() -> None:
    s = fake_session(n=200, val_start=150, test_start=175)
    cs = torch.zeros(len(s.x) + 1, s.D_in, dtype=torch.int64)
    cs[1:] = torch.cumsum(s.x.long(), 0)
    w, K, t = 5, 4, 100
    f = block_features(cs, slice(t, t + 1), w, K)[0]                         # [K·D]
    for b in range(K):
        direct = s.x[t - (b + 1) * w + 1: t - b * w + 1].sum(0)
        assert torch.equal(f[b * s.D_in:(b + 1) * s.D_in].float(), direct)
    x2 = s.x.clone()
    x2[t + 1:] = 7.0                                                         # gelecek değişir
    cs2 = torch.zeros_like(cs)
    cs2[1:] = torch.cumsum(x2.long(), 0)
    assert torch.equal(block_features(cs2, slice(0, t + 1), w, K), block_features(cs, slice(0, t + 1), w, K))
    assert block_features(cs, slice(0, 1), w, K)[0, s.D_in:].abs().sum() == 0   # başta eksik geçmiş = 0


@pytest.mark.skipif(not has_caches, reason="4 ms / 20 ms önbellekleri yok")
def test_4ms_matches_20ms_cache() -> None:
    """20 ms kutusu j, 4 ms kutularının (5j−5, 5j] toplamıdır; bölme sınırları 4 ms'de ±1 kutu içinde."""
    s4 = load_v2_session(CFG, NAME)
    c20 = torch.load(CACHE / f"{NAME}_b20_s20_r0.8.pt")
    x4, x20 = s4.x.long(), c20["x"].long()
    j = torch.arange(1, len(x20))
    summed = sum(x4[5 * j - k] for k in range(5))                            # [n20−1, D]
    assert torch.equal(summed, x20[1:])
    assert torch.allclose(s4.y[::5][: len(x20)], c20["y"])
    assert abs(s4.val_start - 5 * c20["val_start"]) < 5
    assert abs(s4.test_start - 5 * c20["test_start"]) < 5


def test_tune_sessions_disjoint_from_test() -> None:
    """Ayar oturumları test oturumlarından ayrı (KR-049, hakem raporu madde 2); çakışma hata."""
    check_tune_disjoint(CFG)
    for primate in CFG["primates"]:
        day0, tune = crossday_sessions(CFG, primate, tune=True)
        assert day0 == CFG["primates"][primate][0]
        assert tune and not set(tune) & set(CFG["primates"][primate])
    bad = copy.deepcopy(CFG)
    bad["tune_primates"]["indy"] = bad["tune_primates"]["indy"] + [CFG["primates"]["indy"][-1]]
    with pytest.raises(ValueError):
        check_tune_disjoint(bad)
    with pytest.raises(ValueError):
        crossday_sessions(bad, "indy", tune=True)
