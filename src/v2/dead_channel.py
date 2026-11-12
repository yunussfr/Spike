"""Ölü kanal dedektörü (ARCHITECTURE.md §12.17, KR-049).

Day-0 train'de kanalın 4 ms'lik kutuda spike olasılığı p_c. Son spike'tan beri geçen kutu sayısı z_c:
    ölü_c ⇔ z_c · (−ln(1 − p_c)) ≥ −ln α_dead        (Poisson/Bernoulli: bu kadar sessizliğin olasılığı < α_dead)
"Sıfır sayım" x_norm = normalize(0)_c ile tanınır (kodlayıcı deterministik). Spike gelince kanal hemen açılır.
p_c = 0 (Day-0'da hiç ateşlemeyen) kanal değerlendirilmez. Etiketsiz, nedensel (KV6); öğrenmez (KV2).
"""

import math

import torch


def silence_threshold(p: torch.Tensor, alpha_dead: float) -> torch.Tensor:
    """p: [C] kutu başına spike olasılığı → ölü sayılmak için gereken sessiz kutu sayısı [C] (p = 0 → inf)."""
    rate = -torch.log1p(-p.clamp(max=1.0 - 1e-9))                           # −ln(1 − p)
    return torch.where(p > 0, -math.log(alpha_dead) / rate.clamp(min=1e-30), torch.full_like(p, float("inf")))


def zero_code(x_norm: torch.Tensor, z0: torch.Tensor, tol: float) -> torch.Tensor:
    """x_norm: [..., C], z0: [C] sıfır sayımın x_norm'u → [..., C] bool (bu kutuda spike yok)."""
    return (x_norm - z0).abs() <= tol


class DeadChannelDetector:
    def __init__(self, z0: torch.Tensor, p: torch.Tensor, alpha_dead: float, tol: float) -> None:
        """z0: [C] normalize(0), p: [C] Day-0 train spike olasılığı (cihazda)."""
        self.z0, self.tol = z0.clone(), tol
        self.th = silence_threshold(p, alpha_dead)                           # [C]
        self.z = torch.zeros_like(z0)                                        # [C] sessiz kutu sayısı
        self.dead = torch.zeros_like(z0)                                     # [C] 0/1

    def reset_state(self) -> None:
        self.z.zero_()
        self.dead.zero_()

    def step(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: [C] x_norm → ölü: [C] 0/1 (yerinde). Burn-in dahil her adım sayar (girdi gerçek geçmiştir)."""
        silent = zero_code(xt, self.z0, self.tol).float()
        self.z.mul_(silent).add_(silent)
        self.dead.copy_((self.z >= self.th).float())
        return self.dead


def spike_prob(x_norm_train: torch.Tensor, z0: torch.Tensor, tol: float) -> torch.Tensor:
    """Day-0 train x_norm [n, C] → kutu başına spike olasılığı p [C]."""
    return (~zero_code(x_norm_train, z0, tol)).float().mean(0)
