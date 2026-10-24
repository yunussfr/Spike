"""Spike kodlayıcı: delta + seviye kanalları (ARCHITECTURE.md §3.1)."""

from typing import Any

import torch


class SpikeEncoder:
    """x: [B, D_in] → u: [B, T, C], C = (2 + L)·D_in (seviye kapalıysa 2·D_in).

    Kanal düzeni:
      [0, D_in)            delta pozitif: x_t − x_{t−1} > θ_enc
      [D_in, 2·D_in)       delta negatif: x_{t−1} − x_t > θ_enc
      [2·D_in, C)          seviye (termometre): boyut i'nin j. kanalı x_t[i] > b_j
                           (boyut başına L kanal, boyut sırasıyla ardışık)
    Spike vektörü ortam adımının T iç adımı boyunca sabit tutulur.
    Dizinin ilk adımında önceki değer yoktur, o adımda delta spike'ı üretilmez.
    """

    def __init__(
        self,
        D_in: int,
        T: int,
        theta_enc: float,
        level_enabled: bool,
        n_levels: int,
        level_range: tuple[float, float],
    ) -> None:
        self.D_in = D_in
        self.T = T
        self.theta_enc = theta_enc
        self.level_enabled = level_enabled
        self.n_levels = n_levels if level_enabled else 0
        lo, hi = level_range
        j = torch.arange(1, n_levels + 1, dtype=torch.float32)
        self.level_thresholds = lo + j * (hi - lo) / (n_levels + 1)          # [L]
        self.x_prev: torch.Tensor | None = None                               # [B, D_in]

    @classmethod
    def from_config(cls, cfg: dict[str, Any], D_in: int) -> "SpikeEncoder":
        e = cfg["encoder"]
        return cls(
            D_in=D_in, T=cfg["time"]["T"], theta_enc=e["theta_enc"],
            level_enabled=e["level"]["enabled"], n_levels=e["level"]["n_levels"],
            level_range=tuple(e["level"]["range"]),
        )

    @staticmethod
    def n_channels(cfg: dict[str, Any], D_in: int) -> int:
        """Config'e göre kodlayıcı çıktı kanal sayısı C."""
        level = cfg["encoder"]["level"]
        return (2 + (level["n_levels"] if level["enabled"] else 0)) * D_in

    @property
    def n_out(self) -> int:
        return (2 + self.n_levels) * self.D_in

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D_in] → u: [B, T, C] (0/1)."""
        if self.x_prev is None:
            self.x_prev = x.clone()
        diff = x - self.x_prev                                               # [B, D_in]
        parts = [(diff > self.theta_enc).to(x.dtype),                        # [B, D_in]
                 (-diff > self.theta_enc).to(x.dtype)]                       # [B, D_in]
        if self.level_enabled:
            b = self.level_thresholds.to(device=x.device, dtype=x.dtype)    # [L]
            level = (x.unsqueeze(-1) > b).to(x.dtype)                        # [B, D_in, L]
            parts.append(level.flatten(1))                                   # [B, L·D_in]
        spikes = torch.cat(parts, dim=-1)                                    # [B, C]
        self.x_prev = x.clone()
        return spikes.unsqueeze(1).expand(-1, self.T, -1).contiguous()      # [B, T, C]

    __call__ = forward

    def reset_state(self) -> None:
        self.x_prev = None
