"""Beyincik bloğu (ARCHITECTURE.md §3.3): bir sonraki girdiyi tahmin eder, tahmin hatasıyla öğrenir.

K3: yalnızca e = x(t+1) − x̂(t+1) ile eğitilir, δ'yı hiç görmez.
K6: forward yalnızca r(t) ve a(t)'yi alır; x(t+1)'e erişimi yoktur. update, x(t+1) geldikten
    sonra dışarıda hesaplanan e ile çağrılır.

İki öğrenme modu (config `cerebellum.rule`):
  plain      — §3.3'ün birebir aynısı: ΔW_c = η · e · φᵀ,  φ = [r, a].
  normalized — varsayılan (ARCHITECTURE.md §3.3): φ her öznitelik için kendi geçmişinin
               üstel ortalaması ve varyansıyla çevrimiçi standartlaştırılır (nedensel), sabit
               bir bias girişi b eklenir ve normalize LMS uygulanır:
               ΔW_c = η · e · φ̃ᵀ / (ε + ‖φ̃‖²).
               b = √F (bias_scale: auto): standartlaştırılmış F özniteliğin toplam enerjisi ~F
               olduğu için bias girişi aynı enerjiyi alır; b = 1 iken bias çok yavaş öğrenilir.
               Sebep: r = [f, v]'de ‖φ‖² ~ 1e6 mertebesinde; plain kural η < ~1e-7 dışında ıraksar.
"""

import math
from typing import Any

import torch


class Cerebellum:
    def __init__(
        self,
        n_features: int,
        D_a: int,
        D_pred: int,
        rule: str,
        norm_rate: float,
        eps: float,
        bias_scale: float | str,
        device: torch.device | str = "cpu",
    ) -> None:
        if rule not in ("plain", "normalized"):
            raise ValueError(f"rule 'plain' ya da 'normalized' olmalı, verilen: {rule}")
        self.rule = rule
        self.norm_rate = norm_rate
        self.eps = eps
        self.n_in = n_features + D_a + (1 if rule == "normalized" else 0)   # normalized: + bias
        self.bias = math.sqrt(n_features + D_a) if bias_scale == "auto" else float(bias_scale)
        self.W_c = torch.zeros(D_pred, self.n_in, device=device)            # [D_pred, F+A(+1)]
        self.mu = torch.zeros(n_features + D_a, device=device)              # [F+A] çevrimiçi ortalama
        self.var = torch.ones(n_features + D_a, device=device)              # [F+A] çevrimiçi varyans
        self.n_seen = 0
        self.phi: torch.Tensor | None = None                                 # [B, n_in] son forward girdisi

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], n_features: int, D_a: int, D_pred: int, device: torch.device | str = "cpu"
    ) -> "Cerebellum":
        c = cfg["cerebellum"]
        return cls(n_features=n_features, D_a=D_a, D_pred=D_pred, rule=c["rule"],
                   norm_rate=c["norm_rate"], eps=c["eps"], bias_scale=c["bias_scale"], device=device)

    def _features(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """φ = [r, a] (normalized modda standartlaştırılmış + bias). [B, F], [B, A] → [B, n_in]"""
        phi = torch.cat([r, a], dim=-1)                                      # [B, F+A]
        if self.rule == "plain":
            return phi
        # Nedensel istatistik: önce bu adımın girdisiyle güncelle, sonra standartlaştır.
        # İlk adımlarda kısa geçmişe ağırlık ver (rate en az 1/n), sonra sabit oranda unut.
        self.n_seen += 1
        rate = max(self.norm_rate, 1.0 / self.n_seen)
        batch_mean = phi.mean(dim=0)                                         # [F+A]
        self.mu = (1 - rate) * self.mu + rate * batch_mean
        self.var = (1 - rate) * self.var + rate * ((phi - self.mu) ** 2).mean(dim=0)
        z = (phi - self.mu) / (self.var + self.eps).sqrt()                   # [B, F+A]
        bias = torch.full((len(z), 1), self.bias, dtype=z.dtype, device=z.device)
        return torch.cat([z, bias], dim=-1)

    @torch.no_grad()
    def forward(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """r: [B, F], a: [B, A] → x̂(t+1): [B, D_pred]. x(t+1) burada yoktur (K6)."""
        self.phi = self._features(r, a)                                      # [B, n_in]
        return self.phi @ self.W_c.T                                         # [B, D_pred]

    __call__ = forward

    @torch.no_grad()
    def update(self, e: torch.Tensor, eta: float) -> None:
        """e: [B, D_pred] = x(t+1) − x̂(t+1). Yalnızca W_c değişir (K1, K3)."""
        if self.phi is None:
            raise RuntimeError("update'ten önce forward çağrılmalı")
        phi = self.phi                                                       # [B, n_in]
        dW = e.T @ phi / len(phi)                                            # [D_pred, n_in]
        if self.rule == "normalized":
            dW = dW / (self.eps + (phi ** 2).sum(dim=-1).mean())
        self.W_c += eta * dW

    def reset_state(self) -> None:
        """Diziler arası: son girdiyi unutur. Ağırlıklar ve normalizasyon istatistikleri kalır."""
        self.phi = None
