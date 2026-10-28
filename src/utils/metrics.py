"""Değerlendirme yardımcıları: çevrimdışı ridge okuma (üst sınır referansı) ve hata ölçüleri.

Ridge burada yalnızca karşılaştırma/teşhis içindir; sistemin öğrenme kuralı değildir (K2).
"""

import torch


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """X: [n, F], Y: [n, D] → W: [F+1, D] (son satır bias, bias düzenlenmez)."""
    Xb = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1)        # [n, F+1]
    reg = lam * torch.eye(Xb.shape[1], dtype=X.dtype)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def ridge_predict(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """X: [n, F], W: [F+1, D] → [n, D]"""
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1) @ W


def r2_per_column(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Sütun başına R². → [D]"""
    pred = ridge_predict(X, W)                                               # [n, D]
    ss_res = ((Y - pred) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum(dim=0)
    return 1.0 - ss_res / ss_tot


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target) ** 2).mean().item()


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE / hedefin standart sapması."""
    return (((pred - target) ** 2).mean().sqrt() / target.std()).item()
