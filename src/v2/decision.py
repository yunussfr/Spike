"""Okuma: sızıntılı entegratör başlıkları (ARCHITECTURE.md §12.7).

  r^(k)(t) = κ_k·r^(k)(t−1) + (1 − κ_k)·s(t),   κ_k = exp(−Δt/τ_read,k)
  v̂(t)     = (1/K) Σ_k (W^(k)·r̃^(k)(t) + b^(k)),   r̃ = r ⊙ [g_fast·1, g_slow·1]

Çevrimdışı ileri yol ve çapa kaydı (V3) burada; çevrimiçi çapalı NLMS akışı `src/v2/online.py`'de (V4).
"""

import math
from typing import Any

import torch
from torch import nn


class MultiHeadReadout(nn.Module):
    def __init__(self, num_neurons: int, out_dim: int, dt_ms: float, taus_ms: list[float]) -> None:
        super().__init__()
        self.N, self.half, self.K = num_neurons, num_neurons // 2, len(taus_ms)
        self.register_buffer("kappa", torch.tensor([math.exp(-dt_ms / t) for t in taus_ms]).view(-1, 1, 1))  # [K, 1, 1]
        # K başlık tek tensörde (tek çekirdek; ayrı nn.Linear'lar eğitimi ~%60 yavaşlatıyordu, KR-040).
        # Başlatma nn.Linear ile aynı dağılım: U(−1/√N, 1/√N).
        bound = 1.0 / math.sqrt(num_neurons)
        self.weight = nn.Parameter(torch.empty(self.K, out_dim, num_neurons).uniform_(-bound, bound))   # [K, D, N]
        self.bias = nn.Parameter(torch.empty(self.K, out_dim).uniform_(-bound, bound))                  # [K, D]
        self.register_buffer("w_anchor", torch.zeros(self.K, out_dim, num_neurons))   # W_0^(k), Day-0 sonunda donar
        self.register_buffer("b_anchor", torch.zeros(self.K, out_dim))
        self.register_buffer("anchored", torch.tensor(False))
        self.reset_state(1)

    @torch.no_grad()
    def save_anchor(self) -> None:
        """Aşama 2 sonu: başlık ağırlıklarını çapa W_0 olarak dondurur (§12.7, §12.8)."""
        self.w_anchor.copy_(self.weight)
        self.b_anchor.copy_(self.bias)
        self.anchored.fill_(True)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], num_neurons: int, out_dim: int) -> "MultiHeadReadout":
        return cls(num_neurons, out_dim, cfg["time"]["dt_ms"], cfg["readout"]["taus_ms"])

    def reset_state(self, batch_size: int = 1) -> None:
        self.traces = torch.zeros(self.K, batch_size, self.N, device=self.kappa.device)   # [K, B, N]

    def gate(self, g_fast: float, g_slow: float) -> torch.Tensor:
        return torch.cat([torch.full((self.half,), g_fast), torch.full((self.N - self.half,), g_slow)]).to(self.kappa.device)

    def step(self, s: torch.Tensor, g_fast: float = 1.0, g_slow: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """s: [B, N] → v̂: [B, D_out], r̄: [B, N] (başlıkların kapılı izlerinin ortalaması, beyincik için)."""
        self.traces = self.kappa * self.traces + (1.0 - self.kappa) * s.unsqueeze(0)
        gated = self.traces if g_fast == g_slow == 1.0 else self.traces * self.gate(g_fast, g_slow)
        v_hat = torch.einsum("kbn,kdn->bd", gated, self.weight) / self.K + self.bias.mean(0)   # [B, D]
        return v_hat, gated.mean(0)
