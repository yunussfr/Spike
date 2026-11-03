"""Karar bloğu (ARCHITECTURE.md §3.6): görev çıktısını üretir, sonuç hatasıyla öğrenir.

Regresyon modu: ŷ(t) = W_a · φ̃(r(t)); β kullanılmaz (yalnızca ayrık görevlerde, Faz 7).
K3: yalnızca δ = y − ŷ ile eğitilir, beyinciğin tahmin hatası e'yi hiç görmez.

Öğrenme kuralı (config `decision.rule`), beyincikle aynı (§3.3):
  plain      — ΔW_a = η · δ · rᵀ
  normalized — varsayılan: r nedensel standartlaştırılır (+ bias), ΔW_a = η · δ · φ̃ᵀ / (ε + ‖φ̃‖²)

Faz 6 hızlı-yavaş okuma (§3.6, KR-031): değerlendirmede W_a = W_base donuk kalır, yalnızca hızlı
bileşen ΔW öğrenir. Önce tahmin, sonra uyum (prequential):
  ŷ_t      = (W_base + ΔW_t) · φ̃_t                                   (skor bu tahminden)
  ΔW_{t+1} = (1 − λ_pull) · ΔW_t + η_t · e_{t−d+1} · φ̃_{t−d+1}ᵀ / (‖φ̃‖² + ε)
  ‖ΔW‖_F > ρ · ‖W_base‖_F ise ΔW, ρ · ‖W_base‖_F normuna geri ölçeklenir.
d ≥ 1 geri bildirim gecikmesi: (φ̃_t, e_t) çifti ancak d adım sonra uygulanır.
"""

from collections import deque
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
        self.dW: torch.Tensor | None = None                                  # [D_out, n_in] hızlı bileşen (Faz 6); None → kapalı
        self.fast: dict[str, float] = {}                                     # lambda_pull, rho_max, delay
        self._queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()      # bekleyen (φ̃, e) çiftleri

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
        W = self.W_a if self.dW is None else self.W_a + self.dW              # [D_out, n_in]
        return self.phi @ W.T                                                # [B, D_out]

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

    # --- Faz 6: hızlı bileşen ΔW (W_base = W_a donuk) --------------------------------

    def start_fast(self, lambda_pull: float, rho_max: float, delay: int) -> None:
        """ΔW_0 = 0 ile hızlı bileşeni açar. W_a bundan sonra W_base olarak okunur, `update` çağrılmamalı."""
        if delay < 1:
            raise ValueError(f"delay ≥ 1 olmalı (nedensellik), verilen: {delay}")
        self.dW = torch.zeros_like(self.W_a)                                 # [D_out, n_in]
        self.fast = {"lambda_pull": lambda_pull, "rho_max": rho_max, "delay": delay}
        self._queue.clear()

    @torch.no_grad()
    def update_fast(self, delta: torch.Tensor | None, eta: float) -> None:
        """delta: [B, D_out] = y_t − ŷ_t (ŷ_t skorlandıktan sonra) ya da None (etiket yok).

        Çift kuyruğa girer; d adım bekleyen çift uygulanır. Sönüm her çağrıda uygulanır. W_a değişmez.
        """
        if self.dW is None:
            raise RuntimeError("update_fast'ten önce start_fast çağrılmalı")
        if delta is not None:
            if self.phi is None:
                raise RuntimeError("update_fast'ten önce forward çağrılmalı")
            self._queue.append((self.phi, delta))
        self.dW = (1 - self.fast["lambda_pull"]) * self.dW                  # elastik çapa: W_base'e geri çekim
        if len(self._queue) >= self.fast["delay"]:
            phi, d = self._queue.popleft()
            step = nlms_step(d, phi, self.eps) if self.norm else d.T @ phi / len(phi)   # [D_out, n_in]
            self.dW = self.dW + eta * step
        limit = self.fast["rho_max"] * float(self.W_a.norm())                # ρ · ‖W_base‖_F
        n = float(self.dW.norm())
        if n > limit:
            self.dW = self.dW * (limit / n)                                  # norm sınırına geri ölçekle

    def reset_state(self) -> None:
        """Diziler arası: son girdiyi unutur. Ağırlıklar, ΔW ve normalizasyon istatistikleri kalır."""
        self.phi = None
