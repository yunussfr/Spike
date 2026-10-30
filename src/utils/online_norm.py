"""Nedensel çevrimiçi standartlaştırma ve normalize LMS (ARCHITECTURE.md §3.3).

Okuma blokları (beyincik, karar, duygu uzayı) aynı girdi hazırlığını paylaşır:
  φ̃ = [(φ − μ) / √(σ² + ε), b]
μ, σ² her öznitelik için o ana kadarki girdilerin üstel ortalaması/varyansıdır (gelecekten
bilgi yok, K6). b = √F sabit bias girişidir (`bias_scale: auto`).
"""

import math

import torch


class OnlineStandardizer:
    def __init__(
        self,
        n_features: int,
        rate: float,
        eps: float,
        bias_scale: float | str,
        device: torch.device | str = "cpu",
    ) -> None:
        self.rate = rate
        self.eps = eps
        self.n_out = n_features + 1                                          # + bias
        self.bias = math.sqrt(n_features) if bias_scale == "auto" else float(bias_scale)
        self.mu = torch.zeros(n_features, device=device)                    # [F] çevrimiçi ortalama
        self.var = torch.ones(n_features, device=device)                    # [F] çevrimiçi varyans
        self.n_seen = 0
        self.frozen = False                                                  # True: istatistikler güncellenmez

    @torch.no_grad()
    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """φ: [B, F] → φ̃: [B, F+1]. Önce bu adımın girdisiyle istatistiği günceller, sonra standartlaştırır."""
        if not self.frozen:
            # İlk adımlarda kısa geçmişe ağırlık ver (rate en az 1/n), sonra sabit oranda unut.
            self.n_seen += 1
            rate = max(self.rate, 1.0 / self.n_seen)
            self.mu = (1 - rate) * self.mu + rate * phi.mean(dim=0)
            self.var = (1 - rate) * self.var + rate * ((phi - self.mu) ** 2).mean(dim=0)
        z = (phi - self.mu) / (self.var + self.eps).sqrt()                   # [B, F]
        bias = torch.full((len(z), 1), self.bias, dtype=z.dtype, device=z.device)
        return torch.cat([z, bias], dim=-1)

    def state(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.mu.clone(), self.var.clone(), self.n_seen

    def load(self, state: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        self.mu, self.var, self.n_seen = state[0].clone(), state[1].clone(), state[2]


def nlms_step(err: torch.Tensor, phi: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize LMS ağırlık değişimi (η hariç). err: [B, D], phi: [B, F] → ΔW: [D, F]."""
    return (err.T @ phi / len(phi)) / (eps + (phi ** 2).sum(dim=-1).mean())
