"""Kanal bozulması augmentation'ı (ARCHITECTURE.md §12.19, KR-049; Sussillo vd. 2016 esinli).

Aşama 1–2'de her eğitim penceresine, x_norm alanında (CUDA Graph'ın dışında):
  olasılık p_drop:  k ~ U{1, …, k_max} rastgele kanal → x_c(t) = z0_c (sıfır sayımın x_norm'u), tüm pencere
  olasılık p_gain:  x_c(t) ← clip·tanh((x_c(t)·(1 + σ_g·ξ_c) + σ_b·ζ_c) / clip),   ξ, ζ ~ N(0, 1) kanal başına
Rastgelelik verilen üreteçten (eğitim seed'i). Val ve akış bozulmasız.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AugSpec:
    enabled: bool
    p_drop: float
    k_max_frac: float
    p_gain: float
    sigma_gain: float
    sigma_bias: float
    clip: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AugSpec":
        a = cfg.get("augment") or {"enabled": False}
        if not a.get("enabled", False):
            return cls(False, 0.0, 0.0, 0.0, 0.0, 0.0, float(cfg["encoder"]["clip"]))
        return cls(True, float(a["p_drop"]), float(a["k_max_frac"]), float(a["p_gain"]), float(a["sigma_gain"]),
                   float(a["sigma_bias"]), float(cfg["encoder"]["clip"]))


@torch.no_grad()
def augment_windows(x: torch.Tensor, z0: torch.Tensor, spec: AugSpec, gen: torch.Generator) -> torch.Tensor:
    """x: [B, L, C] x_norm pencereleri, z0: [B, C] pencerenin oturumuna ait sıfır kodu → bozulmuş kopya [B, L, C].

    Rastgele sayılar CPU üretecinden çekilir (tekrarlanabilir), sonra cihaza taşınır."""
    if not spec.enabled:
        return x
    B, _, C = x.shape
    dev = x.device
    out = x.clone()
    k_max = max(1, math.ceil(spec.k_max_frac * C))
    # kanal düşürme
    drop_w = torch.rand(B, generator=gen) < spec.p_drop                      # [B]
    k = torch.randint(1, k_max + 1, (B,), generator=gen)                     # [B]
    ranks = torch.rand(B, C, generator=gen).argsort(1).argsort(1)            # [B, C] rastgele sıra
    drop = (ranks < k.unsqueeze(1)) & drop_w.unsqueeze(1)                    # [B, C]
    # kazanç / kayma
    gain_w = (torch.rand(B, generator=gen) < spec.p_gain).float().unsqueeze(1)   # [B, 1]
    g = 1.0 + spec.sigma_gain * torch.randn(B, C, generator=gen) * gain_w
    b = spec.sigma_bias * torch.randn(B, C, generator=gen) * gain_w
    g, b, drop = g.to(dev), b.to(dev), drop.to(dev)
    pert = spec.clip * torch.tanh((out * g.unsqueeze(1) + b.unsqueeze(1)) / spec.clip)
    out = torch.where(gain_w.to(dev).unsqueeze(1) > 0, pert, out)
    return torch.where(drop.unsqueeze(1), z0.unsqueeze(1).expand_as(out), out)
