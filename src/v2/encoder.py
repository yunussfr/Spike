"""SpikeEncoderV2 (ARCHITECTURE.md §12.3): Anscombe + kanal başına z-skoru + yumuşak kırpma.

  x_ans  = √(max(0, x) + c_A)
  z      = (x_ans − μ_ch) / (σ_ch + ε)          μ_ch, σ_ch yalnızca train'den (KV6)
  x_norm = c · tanh(z / c)
  x̄(t)   = α·x̄(t−1) + (1 − α)·x_norm(t),   α = exp(−Δt / τ_smooth)   (beyincik hedefi, §12.5)

x_norm noktasal (durumsuz) bir dönüşümdür; yalnızca x̄ durum taşır. Öğrenmez (`update` yok).
"""

import math
from typing import Any

import torch
from torch import nn


class SpikeEncoderV2(nn.Module):
    def __init__(self, num_channels: int, dt_ms: float, anscombe_c: float, clip: float, smooth_tau_ms: float,
                 eps: float, std_min: float) -> None:
        super().__init__()
        self.C = num_channels
        self.anscombe_c, self.clip, self.eps, self.std_min = anscombe_c, clip, eps, std_min
        self.alpha = math.exp(-dt_ms / smooth_tau_ms)
        self.register_buffer("mean", torch.zeros(num_channels))             # [C]
        self.register_buffer("std", torch.ones(num_channels))               # [C]
        self.fitted = False
        self.x_smooth = torch.zeros(1, num_channels)                         # [B, C]

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_channels: int) -> "SpikeEncoderV2":
        e = cfg["encoder"]
        return cls(num_channels, cfg["time"]["dt_ms"], e["anscombe_c"], e["clip"], e["smooth_tau_ms"], e["eps"],
                   e["std_min"])

    def anscombe(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.clamp(min=0.0) + self.anscombe_c)

    @torch.no_grad()
    def fit(self, x_train: torch.Tensor) -> None:
        """Kanal istatistikleri yalnızca train sayımlarından. x_train: [n, C] ya da [B, n, C]"""
        a = self.anscombe(x_train.float()).reshape(-1, self.C)               # [n, C]
        self.mean.copy_(a.mean(0))
        self.std.copy_(a.std(0).clamp(min=self.std_min))
        self.fitted = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Durumsuz dönüşüm; herhangi bir şekil [..., C] → aynı şekil."""
        if not self.fitted:
            raise RuntimeError("SpikeEncoderV2.fit(x_train) önce çağrılmalı")
        z = (self.anscombe(x.float()) - self.mean) / (self.std + self.eps)
        return self.clip * torch.tanh(z / self.clip)

    def reset_state(self, batch_size: int = 1) -> None:
        self.x_smooth = torch.zeros(batch_size, self.C, device=self.mean.device)

    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tek adım. x_raw: [B, C] → x_norm: [B, C], x̄: [B, C]"""
        x_norm = self.normalize(x_raw)
        self.x_smooth = self.alpha * self.x_smooth + (1.0 - self.alpha) * x_norm.detach()
        return x_norm, self.x_smooth

    @torch.no_grad()
    def smooth_sequence(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Tüm dizi için x̄ (sıfır başlangıç durumu, nedensel). x_norm: [n, C] → [n, C]"""
        out = torch.empty_like(x_norm)
        s = torch.zeros(self.C, device=x_norm.device)
        for t in range(len(x_norm)):
            s = self.alpha * s + (1.0 - self.alpha) * x_norm[t]
            out[t] = s
        return out
