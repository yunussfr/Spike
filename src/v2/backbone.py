"""TemporalBackbone (ARCHITECTURE.md §12.4): hızlı/yavaş alt-popülasyonlu, çevrimdışı eğitilen zamansal omurga.

N nöronun ilk yarısı hızlı (τ_mem ∈ tau_mem_fast_ms), ikinci yarısı yavaş (τ_mem ∈ tau_mem_slow_ms).
τ = τ_min + (τ_max − τ_min)·σ(λ), λ öğrenilir; τ her zaman kendi aralığında kalır.

Modlar:
  plif        I = α_syn·I + W_in·x + W_rec·s(t−1)
              V = β·V + (1−β)·I − ϑ·s(t−1)                   yumuşak sıfırlama, s(t−1) ile (snnTorch Leaky ile aynı biçim)
              s = H(V − ϑ)                                     geri yönde hızlı-sigmoid surrogate
  spiking_gru z, r, h̃ GRU kapıları; V = (1−z)·V + z·h̃ − ϑ·s(t−1); dışarıya yalnızca ikili s
  tau_gru     aynı kapılar, eşiksiz; çıktı sürekli V (üst sınır kontrolü, KV7'den muaf)

KV1: parametreler yalnızca çevrimdışı gradyanla öğrenilir. Çevrimiçi (eval) değişebilen tek şey ϑ'dır
(`homeostasis_update`). Omurgada `update` yoktur.
"""

import math
from typing import Any

import torch
from torch import nn

MODES = ("plif", "spiking_gru", "tau_gru")


class FastSigmoidSpike(torch.autograd.Function):
    """İleri: H(v); geri: 1 / (1 + k|v|)² (Zenke & Ganguli 2018)."""

    @staticmethod
    def forward(ctx: Any, v: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.scale = scale
        return (v >= 0.0).to(v.dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (v,) = ctx.saved_tensors
        return grad / (1.0 + ctx.scale * v.abs()) ** 2, None


def spike_fn(v: torch.Tensor, scale: float) -> torch.Tensor:
    return FastSigmoidSpike.apply(v, scale)


class TemporalBackbone(nn.Module):
    def __init__(self, input_dim: int, cfg: dict[str, Any], dt_ms: float, mode: str | None = None) -> None:
        super().__init__()
        b = cfg                                                              # cfg["backbone"]
        self.mode = mode or b["mode"]
        if self.mode not in MODES:
            raise ValueError(f"bilinmeyen omurga modu: {self.mode}")
        N = b["N"]
        if N % 2:
            raise ValueError("backbone.N çift olmalı (yarı hızlı, yarı yavaş)")
        self.C, self.N, self.half = input_dim, N, N // 2
        self.dt = dt_ms
        self.surrogate_scale = b["surrogate_scale"]
        h = b["homeostasis"]
        self.target_rate, self.homeo_lr, self.rate_decay = h["target_rate"], h["lr"], h["rate_decay"]
        self.theta_min, self.theta_max = h["theta_min"], h["theta_max"]
        self.drop = nn.Dropout(b["dropout"])

        lo_f, hi_f = b["tau_mem_fast_ms"]
        lo_s, hi_s = b["tau_mem_slow_ms"]
        self.register_buffer("tau_min", torch.cat([torch.full((self.half,), lo_f), torch.full((self.half,), lo_s)]))
        self.register_buffer("tau_max", torch.cat([torch.full((self.half,), hi_f), torch.full((self.half,), hi_s)]))
        self.syn_lo, self.syn_hi = b["tau_syn_ms"]
        u_lo, u_hi = b["tau_init_u"]
        self.logit_tau_mem = nn.Parameter(self._logit_init(N, u_lo, u_hi))  # [N]
        self.logit_tau_syn = nn.Parameter(self._logit_init(N, u_lo, u_hi))  # [N]

        gain = b["w_rec_gain"]
        if self.mode == "plif":
            self.w_in = nn.Linear(input_dim, N)
            self.w_rec = nn.Linear(N, N, bias=False)
            # dalgalanma güdümlü başlatma (Rossbroich vd. 2022)
            nn.init.normal_(self.w_in.weight, 0.0, 1.0 / math.sqrt(input_dim))
            nn.init.zeros_(self.w_in.bias)
            nn.init.orthogonal_(self.w_rec.weight, gain=gain)
        else:
            self.w_z, self.w_r, self.w_h = (nn.Linear(input_dim, N) for _ in range(3))
            self.u_z, self.u_r, self.u_h = (nn.Linear(N, N, bias=False) for _ in range(3))
            for lin in (self.w_z, self.w_r, self.w_h):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            for lin in (self.u_z, self.u_r, self.u_h):
                nn.init.orthogonal_(lin.weight, gain=gain)
            with torch.no_grad():                                            # chrono başlatma: z ≈ Δt/τ_mem
                tau0 = self.tau_mem()
                self.w_z.bias.copy_(-torch.log((tau0 / dt_ms - 1.0).clamp(min=b["chrono_min"])))

        th0 = b["theta_init_spiking_gru"] if self.mode == "spiking_gru" else b["theta_init"]
        self.register_buffer("theta", torch.full((N,), float(th0)))         # ϑ [N]; yalnızca homeostazla değişir
        self.register_buffer("rate_ema", torch.full((N,), float(self.target_rate)))
        self.reset_state(1)

    @staticmethod
    def _logit_init(n: int, u_lo: float, u_hi: float) -> torch.Tensor:
        u = torch.rand(n).clamp(u_lo, u_hi)
        return torch.log(u / (1.0 - u))

    def tau_mem(self) -> torch.Tensor:
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.logit_tau_mem)   # [N] ms

    def tau_syn(self) -> torch.Tensor:
        return self.syn_lo + (self.syn_hi - self.syn_lo) * torch.sigmoid(self.logit_tau_syn)      # [N] ms

    @property
    def spiking(self) -> bool:
        return self.mode != "tau_gru"

    def reset_state(self, batch_size: int = 1) -> None:
        dev = self.theta.device
        self.v = torch.zeros(batch_size, self.N, device=dev)                 # [B, N]
        self.i_syn = torch.zeros(batch_size, self.N, device=dev)             # [B, N] (yalnızca plif)
        self.s = torch.zeros(batch_size, self.N, device=dev)                 # [B, N] son çıktı

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Tek adım. x: [B, C] → s: [B, N] (tau_gru'da sürekli V)."""
        x_in = self.drop(x)
        s_prev = self.drop(self.s)
        if self.mode == "plif":
            beta = torch.exp(-self.dt / self.tau_mem())                      # [N]
            alpha = torch.exp(-self.dt / self.tau_syn())
            self.i_syn = alpha * self.i_syn + self.w_in(x_in) + self.w_rec(s_prev)
            self.v = beta * self.v + (1.0 - beta) * self.i_syn - self.theta * self.s
            out = spike_fn(self.v - self.theta, self.surrogate_scale)
        else:
            z = torch.sigmoid(self.w_z(x_in) + self.u_z(s_prev))
            r = torch.sigmoid(self.w_r(x_in) + self.u_r(s_prev))
            h = torch.tanh(self.w_h(x_in) + self.u_h(r * s_prev))
            if self.mode == "spiking_gru":
                self.v = (1.0 - z) * self.v + z * h - self.theta * self.s
                out = spike_fn(self.v - self.theta, self.surrogate_scale)
            else:
                self.v = (1.0 - z) * self.v + z * h
                out = self.v
        self.s = out
        return out

    @torch.no_grad()
    def homeostasis_update(self, s: torch.Tensor) -> None:
        """Çevrimiçi yerel eşik adaptasyonu (KV1, KV7): ϑ ← clip(ϑ + η_h·(r̄ − ρ)). s: [B, N]"""
        if not self.spiking:
            return
        self.rate_ema.mul_(self.rate_decay).add_((1.0 - self.rate_decay) * s.mean(0))
        self.theta.add_(self.homeo_lr * (self.rate_ema - self.target_rate)).clamp_(self.theta_min, self.theta_max)


def homeostatic_loss(rate: torch.Tensor, f_min: float, f_max: float) -> torch.Tensor:
    """rate: [N] nöron başına ortalama oran (pencere ve batch üzerinde) → skaler."""
    return (torch.relu(f_min - rate) ** 2 + torch.relu(rate - f_max) ** 2).mean()
