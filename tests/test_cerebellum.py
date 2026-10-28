"""Faz 2: Cerebellum testleri (K1, K3, K6)."""

import copy
import sys
from pathlib import Path

import pytest
import torch

from src.cerebellum import Cerebellum
from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from tests.conftest import CFG

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import cerebellum_test  # noqa: E402

F, A = 20, 1


def make(rule: str) -> Cerebellum:
    cfg = copy.deepcopy(CFG)
    cfg["cerebellum"]["rule"] = rule
    return Cerebellum.from_config(cfg, n_features=F, D_a=A, D_pred=1)


@pytest.mark.parametrize("rule", ["plain", "normalized"])
def test_forward_shape_and_signature(rule: str) -> None:
    """forward yalnızca r(t) ve a(t)'yi alır: x(t+1) için bir giriş yok (K6)."""
    cb = make(rule)
    x_hat = cb(torch.randn(3, F), torch.zeros(3, A))
    assert x_hat.shape == (3, 1)
    import inspect
    assert list(inspect.signature(Cerebellum.forward).parameters) == ["self", "r", "a"]


def test_plain_rule_matches_architecture() -> None:
    """plain: ΔW_c = η · e · [r, a]ᵀ (ARCHITECTURE.md §3.3)."""
    cb = make("plain")
    r, a = torch.randn(1, F), torch.randn(1, A)
    cb(r, a)
    e, eta = torch.tensor([[0.5]]), 0.1
    cb.update(e, eta)
    assert torch.allclose(cb.W_c, eta * e * torch.cat([r, a], dim=-1))


def test_normalized_rule_is_stable_with_large_features() -> None:
    """normalized: [f, v] ölçeğinde (|v| ~ 1e2) girdilerle bile ıraksamaz ve hatayı azaltır."""
    torch.manual_seed(0)
    cb = make("normalized")
    w_true = torch.randn(F)
    errs = []
    for t in range(3000):
        r = torch.randn(1, F) * 300 + 50                                     # büyük ölçek + kayma
        y = (r - 50) / 300 @ w_true.unsqueeze(1) + 0.9                       # [1, 1]
        e = y - cb(r, torch.zeros(1, A))
        cb.update(e, eta=CFG["regulators"]["defaults"]["eta"])
        errs.append(e.pow(2).item())
    assert all(torch.isfinite(torch.tensor(errs)))
    # NLMS, η = 0.01, ~22 bağımsız yön: 3000 adımda hata kabaca ~1/4'e iner
    assert sum(errs[-500:]) / 500 < 0.5 * sum(errs[:500]) / 500


def test_update_changes_only_w_c() -> None:
    """Güncelleme yalnızca W_c'yi değiştirir; rezervuar (K1) ve başka hiçbir tensör değişmez."""
    D_in = 1
    res = LIFReservoir.from_config(CFG, D_in=D_in, seed=0, calibrate=False)
    enc = SpikeEncoder.from_config(CFG, D_in=D_in)
    cb = Cerebellum.from_config(CFG, n_features=2 * res.N, D_a=A, D_pred=D_in)
    before_res = {k: v.clone() for k, v in res.state_dict().items()}
    r = res(enc(torch.rand(1, D_in)), g=1.0, tau=100.0)
    V_before = res.V.clone()
    W_before = cb.W_c.clone()
    x_hat = cb(r, torch.zeros(1, A))
    cb.update(torch.rand(1, D_in) - x_hat, eta=0.01)
    assert not torch.equal(cb.W_c, W_before)
    for k, v in res.state_dict().items():
        assert torch.equal(v, before_res[k]), k
    assert torch.equal(res.V, V_before)                                      # durum da dokunulmadı


def test_update_requires_forward() -> None:
    cb = make("plain")
    with pytest.raises(RuntimeError):
        cb.update(torch.zeros(1, 1), eta=0.01)


def test_no_future_leak_in_pipeline() -> None:
    """K6: r(t) ve dolayısıyla x̂(t+1), x(t+1)'den bağımsız. Gelecek değiştirilince tahmin aynı kalır."""
    cfg = copy.deepcopy(CFG)
    cfg["reservoir"]["N"] = 100
    cfg["encoder"]["level"]["range"] = [0.0, 1.0]
    cfg["chaotic_test"] = {"washout": 0}
    torch.manual_seed(1)
    x1 = torch.rand(40, 1)
    x2 = x1.clone()
    t_cut = 25
    x2[t_cut + 1:] = 1.0 - x2[t_cut + 1:]                                     # t_cut'tan sonraki gelecek farklı
    r1, _ = cerebellum_test.collect_features(cfg, x1, seed=0, device=torch.device("cpu"))
    r2, _ = cerebellum_test.collect_features(cfg, x2, seed=0, device=torch.device("cpu"))
    assert torch.equal(r1[: t_cut + 1], r2[: t_cut + 1])                     # t ≤ t_cut aynı
    assert not torch.equal(r1[t_cut + 1:], r2[t_cut + 1:])                   # sonrası farklı (test anlamlı)
    cb = Cerebellum.from_config(cfg, n_features=r1.shape[1], D_a=A, D_pred=1)
    cb.W_c.normal_()
    assert torch.equal(cb(r1[t_cut: t_cut + 1], torch.zeros(1, A)),
                       cb(r2[t_cut: t_cut + 1], torch.zeros(1, A)))


def test_level_range_from_train_only() -> None:
    """Seviye aralığı yalnızca eğitim bölümünden: test bölümündeki uç değerler aralığı değiştirmez."""
    cfg = copy.deepcopy(CFG)
    cfg["encoder"]["level"]["range"] = "train"
    cfg["data"] = {"n_train": 10}
    x = torch.cat([torch.linspace(0.2, 0.8, 10), torch.tensor([5.0, -5.0])]).unsqueeze(1)
    out = cerebellum_test.resolve_level_range(cfg, x)
    assert out["encoder"]["level"]["range"] == pytest.approx([0.2, 0.8])
