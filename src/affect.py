"""Duygu uzayı (ARCHITECTURE.md §3.4): sistemin o anki durumunu düşük boyutlu z vektörüne sıkıştırır.

  z(t) = tanh(W_z · φ̃(r(t)))                       [B, k], varsayılan k = 3
  [q̂, û, ŝ] = W_h · [z, 1]                          üç doğrusal tahmin başı
Hedefler (AffectTargets, δ ve e'yi yalnızca OKUR — K3):
  q = −‖δ‖²                       anlık performans
  u = ‖e‖²                        belirsizlik
  s = EMA_kısa(u) − EMA_uzun(u)   eğilim (iyileşme < 0 < kötüleşme)
Hedefler çevrimiçi standartlaştırılır (ölçekleri çok farklı); başlar standart hedefi tahmin eder.

Öğrenme (yerel, açık formül, autograd yok — K2). ε_h = hedef̃ − baş:
  ΔW_h = η_h · ε_h · [z, 1]ᵀ / (ε + ‖[z, 1]‖²)
  ΔW_z = η_z · ((W_hᵀ ε_h)[:k] ⊙ (1 − z²)) · φ̃ᵀ / (ε + ‖φ̃‖²)
Başlardan W_z'ye hata yalnızca tek bir doğrusal katman (W_h) üzerinden gelir.

İsteğe bağlı iki düzenleme (config; varsayılanları 0 ve [1, 1, 1] → yukarıdaki formülün aynısı):
  - `preact_l2` (λ): ön-aktivasyon a = W_z·φ̃ üzerinde λ/2·‖a‖² cezası → z'nin ±1'e doymasını
    engeller (doygunlukta 1 − z² ≈ 0 olur ve W_z öğrenemez).
      ΔW_z = η_z · (((W_hᵀ (w ⊙ ε_h))[:k] ⊙ (1 − z²)) − λ·a) · φ̃ᵀ / (ε + ‖φ̃‖²)
  - `head_weights` (w): başların W_z'ye geri dönen hatadaki ağırlığı (ör. u başını güçlendirmek).
    Başların kendi güncellemesi (ΔW_h) bundan etkilenmez.

Faz 4'te z hiçbir şeyi değiştirmez: regülatörler (Faz 5) yazılmadı, karar ve beyincik z'yi görmez.
"""

from typing import Any

import torch

from src.utils.online_norm import OnlineStandardizer

HEADS = ("q", "u", "s")


class AffectTargets:
    """δ ve e'den q, u, s hedeflerini üretir. EMA durumunu tutar; δ ve e'yi değiştirmez."""

    def __init__(self, ema_short: float, ema_long: float) -> None:
        self.a_short, self.a_long = ema_short, ema_long
        self.m_short: torch.Tensor | None = None                            # [B]
        self.m_long: torch.Tensor | None = None                             # [B]

    @torch.no_grad()
    def __call__(self, delta: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """delta: [B, D_out], e: [B, D_in] → {q, u, s}: her biri [B]."""
        q = -(delta ** 2).sum(dim=-1)                                        # [B]
        u = (e ** 2).sum(dim=-1)                                             # [B]
        if self.m_short is None:
            self.m_short, self.m_long = u.clone(), u.clone()
        else:
            self.m_short = (1 - self.a_short) * self.m_short + self.a_short * u
            self.m_long = (1 - self.a_long) * self.m_long + self.a_long * u
        return {"q": q, "u": u, "s": self.m_short - self.m_long}

    def reset_state(self) -> None:
        self.m_short = self.m_long = None


class AffectSpace:
    def __init__(
        self,
        n_features: int,
        k: int,
        lr_z: float,
        lr_head: float,
        norm_rate: float,
        target_rate: float,
        eps: float,
        bias_scale: float | str,
        seed: int,
        preact_l2: float = 0.0,
        head_weights: list[float] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.k = k
        self.preact_l2 = preact_l2
        self.head_w = torch.tensor(head_weights or [1.0] * len(HEADS), device=device)   # [3]
        self.lr_z, self.lr_head, self.eps = lr_z, lr_head, eps
        self.norm = OnlineStandardizer(n_features, norm_rate, eps, bias_scale, device)
        gen = torch.Generator().manual_seed(seed)                            # kendi üreteci: global RNG'ye dokunmaz
        F = self.norm.n_out
        self.W_z = (torch.randn(k, F, generator=gen) / F ** 0.5).to(device)  # [k, F+1], ön-aktivasyon ~ N(0, 1)
        self.W_h = torch.zeros(len(HEADS), k + 1, device=device)             # [3, k+1]
        # Hedef standartlaştırması (baş başına çevrimiçi ortalama/varyans)
        self.t_rate = target_rate
        self.t_mu = torch.zeros(len(HEADS), device=device)                   # [3]
        self.t_var = torch.ones(len(HEADS), device=device)                   # [3]
        self.t_seen = 0
        self.frozen = False
        self.phi: torch.Tensor | None = None                                 # [B, F+1]
        self.a: torch.Tensor | None = None                                   # [B, k] ön-aktivasyon
        self.z: torch.Tensor | None = None                                   # [B, k]
        self.heads: torch.Tensor | None = None                               # [B, 3] standart ölçekte

    @classmethod
    def from_config(cls, cfg: dict[str, Any], n_features: int, seed: int,
                    device: torch.device | str = "cpu") -> "AffectSpace":
        a = cfg["affect"]
        return cls(n_features=n_features, k=a["k"], lr_z=a["lr_z"], lr_head=a["lr_head"],
                   norm_rate=a["norm_rate"], target_rate=a["target_rate"], eps=a["eps"],
                   bias_scale=a["bias_scale"], seed=seed + a["seed_offset"],
                   preact_l2=a["preact_l2"], head_weights=a["head_weights"], device=device)

    def _zb(self) -> torch.Tensor:
        return torch.cat([self.z, torch.ones(len(self.z), 1, dtype=self.z.dtype, device=self.z.device)], dim=-1)

    @torch.no_grad()
    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """r: [B, 2N] → z: [B, k], başlar {q̂, û, ŝ}: her biri [B] (hedefin özgün ölçeğinde)."""
        self.norm.frozen = self.frozen
        self.phi = self.norm(r)                                              # [B, F+1]
        self.a = self.phi @ self.W_z.T                                       # [B, k]
        self.z = torch.tanh(self.a)                                          # [B, k]
        self.heads = self._zb() @ self.W_h.T                                 # [B, 3]
        raw = self.heads * (self.t_var + self.eps).sqrt() + self.t_mu        # [B, 3] özgün ölçek
        return self.z, {h: raw[:, i] for i, h in enumerate(HEADS)}

    __call__ = forward

    @torch.no_grad()
    def update(self, q: torch.Tensor, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """q, u, s: [B]. Yalnızca W_z ve W_h değişir. Döndürür: standart ölçekte baş hataları [B, 3]."""
        if self.phi is None:
            raise RuntimeError("update'ten önce forward çağrılmalı")
        target = torch.stack([q, u, s], dim=-1)                              # [B, 3]
        if not self.frozen:
            self.t_seen += 1
            rate = max(self.t_rate, 1.0 / self.t_seen)
            self.t_mu = (1 - rate) * self.t_mu + rate * target.mean(dim=0)
            self.t_var = (1 - rate) * self.t_var + rate * ((target - self.t_mu) ** 2).mean(dim=0)
        err = (target - self.t_mu) / (self.t_var + self.eps).sqrt() - self.heads   # [B, 3]
        if self.frozen:
            return err
        zb = self._zb()                                                      # [B, k+1]
        back = ((err * self.head_w) @ self.W_h)[:, : self.k] * (1 - self.z ** 2)   # [B, k] tek doğrusal katman üzerinden
        back = back - self.preact_l2 * self.a                                # L2 cezası (λ = 0 → etkisiz)
        B = len(err)
        self.W_h += self.lr_head * (err.T @ zb / B) / (self.eps + (zb ** 2).sum(-1).mean())
        self.W_z += self.lr_z * (back.T @ self.phi / B) / (self.eps + (self.phi ** 2).sum(-1).mean())
        return err

    def reset_state(self) -> None:
        """Diziler arası: son girdiyi unutur. Ağırlıklar ve istatistikler kalır."""
        self.phi = self.a = self.z = self.heads = None
