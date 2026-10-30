"""Beyincik bloğu (ARCHITECTURE.md §3.3): bir sonraki girdiyi tahmin eder, tahmin hatasıyla öğrenir.

K3: yalnızca e = x(t+1) − x̂(t+1) ile eğitilir, δ'yı hiç görmez.
K6: forward yalnızca r(t) ve a(t)'yi alır; x(t+1)'e erişimi yoktur. update, x(t+1) geldikten
    sonra dışarıda hesaplanan e ile çağrılır.

İki öğrenme modu (config `cerebellum.rule`):
  plain      — §3.3'ün birebir aynısı: ΔW_c = η · e · φᵀ,  φ = [r, a]. Faz 2'de ıraksadı.
  normalized — varsayılan (ARCHITECTURE.md §3.3): φ her öznitelik için kendi geçmişinin
               üstel ortalaması ve varyansıyla çevrimiçi standartlaştırılır (nedensel), sabit
               bir bias girişi b eklenir ve normalize LMS uygulanır:
               ΔW_c = η · e · φ̃ᵀ / (ε + ‖φ̃‖²).
               b = √F (bias_scale: auto): standartlaştırılmış F özniteliğin toplam enerjisi ~F
               olduğu için bias girişi aynı enerjiyi alır; b = 1 iken bias çok yavaş öğrenilir.
               Sebep: r = [f, v]'de ‖φ‖² ~ 1e6 mertebesinde; plain kural η < ~1e-7 dışında ıraksar.
"""

from typing import Any

import torch

from src.utils.online_norm import OnlineStandardizer, nlms_step


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
        self.eps = eps
        self.norm = (OnlineStandardizer(n_features + D_a, norm_rate, eps, bias_scale, device)
                     if rule == "normalized" else None)
        self.n_in = self.norm.n_out if self.norm else n_features + D_a      # normalized: + bias
        self.W_c = torch.zeros(D_pred, self.n_in, device=device)            # [D_pred, F+A(+1)]
        self.phi: torch.Tensor | None = None                                 # [B, n_in] son forward girdisi

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], n_features: int, D_a: int, D_pred: int, device: torch.device | str = "cpu"
    ) -> "Cerebellum":
        c = cfg["cerebellum"]
        return cls(n_features=n_features, D_a=D_a, D_pred=D_pred, rule=c["rule"],
                   norm_rate=c["norm_rate"], eps=c["eps"], bias_scale=c["bias_scale"], device=device)

    @torch.no_grad()
    def forward(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """r: [B, F], a: [B, A] → x̂(t+1): [B, D_pred]. x(t+1) burada yoktur (K6)."""
        phi = torch.cat([r, a], dim=-1)                                      # [B, F+A]
        self.phi = self.norm(phi) if self.norm else phi                      # [B, n_in]
        return self.phi @ self.W_c.T                                         # [B, D_pred]

    __call__ = forward

    @torch.no_grad()
    def update(self, e: torch.Tensor, eta: float) -> None:
        """e: [B, D_pred] = x(t+1) − x̂(t+1). Yalnızca W_c değişir (K1, K3)."""
        if self.phi is None:
            raise RuntimeError("update'ten önce forward çağrılmalı")
        if self.norm:
            self.W_c += eta * nlms_step(e, self.phi, self.eps)
        else:
            self.W_c += eta * (e.T @ self.phi / len(self.phi))

    def reset_state(self) -> None:
        """Diziler arası: son girdiyi unutur. Ağırlıklar ve normalizasyon istatistikleri kalır."""
        self.phi = None
