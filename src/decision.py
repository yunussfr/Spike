"""Karar bloğu (ARCHITECTURE.md §3.6): görev çıktısını üretir, sonuç hatasıyla öğrenir.

Regresyon modu: ŷ(t) = W_a · φ̃(r(t)); β kullanılmaz (yalnızca ayrık görevlerde, Faz 7).
K3: yalnızca δ = y − ŷ ile eğitilir, beyinciğin tahmin hatası e'yi hiç görmez.

Öğrenme kuralı (config `decision.rule`), beyincikle aynı (§3.3):
  plain      — ΔW_a = η · δ · rᵀ
  normalized — varsayılan: r nedensel standartlaştırılır (+ bias), ΔW_a = η · δ · φ̃ᵀ / (ε + ‖φ̃‖²)
"""

from typing import Any

import torch

from src.utils.online_norm import OnlineStandardizer, nlms_step


class DecisionReadout:
    def __init__(
        self,
        n_features: int,
        D_out: int,
        mode: str,
        rule: str,
        norm_rate: float,
        eps: float,
        bias_scale: float | str,
        device: torch.device | str = "cpu",
    ) -> None:
        if mode != "regression":
            raise NotImplementedError("Faz 3: yalnızca regresyon modu (ayrık mod Faz 7)")
        if rule not in ("plain", "normalized"):
            raise ValueError(f"rule 'plain' ya da 'normalized' olmalı, verilen: {rule}")
        self.mode = mode
        self.rule = rule
        self.eps = eps
        self.norm = (OnlineStandardizer(n_features, norm_rate, eps, bias_scale, device)
                     if rule == "normalized" else None)
        self.n_in = self.norm.n_out if self.norm else n_features
        self.W_a = torch.zeros(D_out, self.n_in, device=device)             # [D_out, F(+1)]
        self.phi: torch.Tensor | None = None                                 # [B, n_in] son forward girdisi

    @classmethod
    def from_config(cls, cfg: dict[str, Any], n_features: int, D_out: int,
                    device: torch.device | str = "cpu") -> "DecisionReadout":
        c = cfg["decision"]
        return cls(n_features=n_features, D_out=D_out, mode=c["mode"], rule=c["rule"],
                   norm_rate=c["norm_rate"], eps=c["eps"], bias_scale=c["bias_scale"], device=device)

    @torch.no_grad()
    def forward(self, r: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """r: [B, F] → ŷ: [B, D_out]. β regresyon modunda kullanılmaz."""
        self.phi = self.norm(r) if self.norm else r                          # [B, n_in]
        return self.phi @ self.W_a.T                                         # [B, D_out]

    __call__ = forward

    @torch.no_grad()
    def update(self, delta: torch.Tensor, eta: float) -> None:
        """delta: [B, D_out] = y − ŷ. Yalnızca W_a değişir (K3)."""
        if self.phi is None:
            raise RuntimeError("update'ten önce forward çağrılmalı")
        if self.norm:
            self.W_a += eta * nlms_step(delta, self.phi, self.eps)
        else:
            self.W_a += eta * (delta.T @ self.phi / len(self.phi))

    def reset_state(self) -> None:
        """Diziler arası: son girdiyi unutur. Ağırlıklar ve normalizasyon istatistikleri kalır."""
        self.phi = None
