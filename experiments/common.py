"""Deney script'lerinin ortak parçaları: rezervuar kurulumu, r(t) toplama, ridge referansı, R².

Ridge yalnızca karşılaştırma içindir (üst sınır); sistemin öğrenme kuralı değildir (K2).
"""

import math
from typing import Any

import numpy as np
import torch

from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from src.utils.metrics import ridge_predict
from src.utils.seeding import set_seed


def build_reservoir(cfg: dict[str, Any], D_in: int, seed: int, device: torch.device,
                    x_calib: torch.Tensor | None = None) -> tuple[LIFReservoir, torch.Tensor]:
    """Sabit rezervuarı kurar ve θ(τ)'yu kalibre eder. Döndürür: rezervuar, ölçülen oranlar [G].

    `reservoir.calibration.source: train` ise kalibrasyon girdisi rastgele değil, verilen train
    dizisinin başından alınır (x_calib: [S, D_in]; test/val'e bakılmaz).
    """
    set_seed(seed)
    res = LIFReservoir.from_config(cfg, D_in=D_in, seed=seed, calibrate=False).to(device)
    c = cfg["reservoir"]["calibration"]
    if c.get("source", "uniform") == "train":
        if x_calib is None:
            raise ValueError("calibration.source: train için x_calib gerekli")
        washout = math.ceil(c["washout_tau_multiple"] * max(c["tau_grid"]) / (res.T * res.dt))
        enc = SpikeEncoder.from_config(cfg, D_in=D_in)
        x = x_calib[: washout + c["n_measure"]].to(device)
        u_seq = torch.cat([enc(x[t: t + 1]) for t in range(len(x))])       # [S, T, C]
        rates = res.calibrate(u_seq, g=cfg["regulators"]["defaults"]["g"],
                              target_rate=c["target_rate_frac"] * res.rho, washout=washout, n_iter=c["n_iter"])
    else:
        rates = res.calibrate_from_config(cfg, D_in, seed)
    return res, rates


@torch.no_grad()
def collect_features(cfg: dict[str, Any], res: LIFReservoir, x: torch.Tensor, device: torch.device,
                     g: float | None = None, tau: float | None = None) -> torch.Tensor:
    """x: [n, D_in] → r: [n, 2N] (CPU). Dizi başında bir kez sıfırlanır, içeride sıfırlanmaz (K6)."""
    g = cfg["regulators"]["defaults"]["g"] if g is None else g
    tau = cfg["regulators"]["defaults"]["tau"] if tau is None else tau
    enc = SpikeEncoder.from_config(cfg, D_in=x.shape[1])
    res.reset_state()
    r = torch.empty(len(x), 2 * res.N)
    xd = x.to(device)
    for t in range(len(x)):
        r[t] = res(enc(xd[t: t + 1]), g=g, tau=tau)[0].cpu()
    res.reset_state()
    return r


def r2_neurobench(pred: torch.Tensor, y: torch.Tensor) -> float:
    """NeuroBench `R2` metriği (x ve y bileşeninin R²'sinin ortalaması). pred, y: [n, 2]."""
    from neurobench.metrics.workload import R2
    m = R2()
    m.reset()
    m(None, pred.float().cpu(), (None, y.float().cpu()))
    return float(m.compute())


def r2_columns(pred: torch.Tensor, y: torch.Tensor) -> list[float]:
    """Kendi R² hesabımız, sütun başına (çapraz kontrol). [n, D] → D değer."""
    pred, y = pred.double(), y.double()
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    return (1.0 - ss_res / ss_tot).tolist()


def _gram(X: torch.Tensor, Y: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor,
          chunk: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """Standartlaştırılmış [Z, 1] için ZᵀZ ve ZᵀY, parça parça (büyük oturumda bellek için). → [F+1, F+1], [F+1, D]"""
    F = X.shape[1]
    G = torch.zeros(F + 1, F + 1, dtype=torch.float64)
    B = torch.zeros(F + 1, Y.shape[1], dtype=torch.float64)
    for i in range(0, len(X), chunk):
        Z = (X[i: i + chunk].double() - mu) / sd
        Z = torch.cat([Z, torch.ones(len(Z), 1, dtype=torch.float64)], dim=1)
        G += Z.T @ Z
        B += Z.T @ Y[i: i + chunk].double()
    return G, B


def _solve(G: torch.Tensor, B: torch.Tensor, lam: float) -> torch.Tensor:
    reg = lam * torch.eye(len(G), dtype=G.dtype)
    reg[-1, -1] = 0.0                                                        # bias düzenlenmez
    return torch.linalg.solve(G + reg, B)


def ridge_reference(r: torch.Tensor, y: torch.Tensor, train: slice, val: slice, test: slice,
                    grid: list[float], washout: int) -> tuple[torch.Tensor, float]:
    """Aynı r üzerinde çevrimdışı ridge: λ val'de seçilir, son model train'de eğitilir. → test tahmini, λ."""
    tr = slice(train.start + washout, train.stop)
    mu = r[tr].double().mean(dim=0)
    sd = r[tr].double().std(dim=0).clamp(min=1e-8)

    def Z(s: slice) -> torch.Tensor:
        return (r[s].double() - mu) / sd

    G, B = _gram(r[tr], y[tr], mu, sd)
    errs = []
    for lam in grid:
        W = _solve(G, B, lam)
        errs.append(((ridge_predict(Z(val), W) - y[val].double()) ** 2).mean().item())
    lam = grid[int(np.argmin(errs))]
    return ridge_predict(Z(test), _solve(G, B, lam)).float(), lam


def firing_rate(r: torch.Tensor, N: int, rows: slice) -> float:
    """f bölümünün (ilk N sütun) ortalaması = ortalama ateşleme oranı."""
    return r[rows, :N].mean().item()
