"""CerebellumV2 (ARCHITECTURE.md §12.5, §12.13; KV3, KV6): yumuşatılmış sonraki girdiyi efference copy ile tahmin eder.

    φ(t)           = [r̄(t), v̂(t−1)]                    r̄: okuma izlerinin başlık ortalaması (kapılanmamış), v̂: efference copy
    x̂_smooth(t+1) = W_cer·φ(t)
    e_cer(t+1)     = x̄(t+1) − x̂_smooth(t+1)
    ΔW_cer         = η_cer·e_cer·φᵀ / (‖φ‖² + ε)        yerel NLMS; yalnızca e_cer'den (KV3)

Kararlar (KR-044):
- Beyinciğe **kapılanmamış** izler verilir (§12.10 nokta 4): g a(t)'ye, a(t) e_cer'e bağlı; kapılı iz döngü kurardı.
- W_cer Day-0 train diliminde ridge ile başlatılır (`fit`, Aşama 2b); E_slow Day-0 artık hatasından başlar.
  Böylece Day-0'da sürpriz ≈ 0, yeni günün başında yükselir.

Durum koşul başınadır (E akış): W [E, C, P], son φ [E, P], son tahmin x̂ [E, C]. Tüm güncellemeler yerinde
(CUDA Graph yakalamasıyla uyumlu).
"""

from typing import Any

import torch


class CerebellumV2:
    def __init__(self, w0: torch.Tensor, eps2_0: float, cfg: dict[str, Any]) -> None:
        """w0: [C, P] Day-0 ridge çözümü; eps2_0: Day-0 ortalama ‖e_cer‖²/C; cfg: config `cerebellum`."""
        self.w0 = w0
        self.eps2_0 = float(eps2_0)
        self.eta = cfg["eta"]
        self.eps = cfg["nlms_eps"]

    @staticmethod
    @torch.no_grad()
    def fit(phi: torch.Tensor, target: torch.Tensor, ridge: float) -> tuple[torch.Tensor, float]:
        """phi: [T, P] (φ(t)), target: [T, C] (x̄(t+1)) → W: [C, P], Day-0 artık ε² (adım ve kanal ortalaması).

        Ridge: W = Yᵀ Φ (ΦᵀΦ + λ·T·I)⁻¹ (float64)."""
        P = phi.shape[1]
        f, y = phi.double(), target.double()
        A = f.T @ f + ridge * len(f) * torch.eye(P, dtype=torch.float64, device=f.device)
        w = torch.linalg.solve(A, f.T @ y).T                                 # [C, P]
        eps2 = float(((y - f @ w.T) ** 2).mean())
        return w.float(), eps2

    def reset_state(self, E: int) -> None:
        dev = self.w0.device
        C, P = self.w0.shape
        self.W = self.w0.unsqueeze(0).repeat(E, 1, 1).clone()               # [E, C, P]
        self.phi = torch.zeros(E, P, device=dev)                             # φ(t−1)
        self.x_hat = torch.zeros(E, C, device=dev)                           # x̂_smooth(t), φ(t−1)'den
        self.valid = torch.zeros((), device=dev)                             # ilk adımda önceki tahmin yok

    def forward(self, r_bar: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
        """r_bar: [E, N], v_prev: [E, D] → x̂_smooth(t+1): [E, C]. φ(t) saklanır."""
        self.phi.copy_(torch.cat([r_bar, v_prev], 1))
        self.x_hat.copy_(torch.einsum("ecp,ep->ec", self.W, self.phi))
        self.valid.fill_(1.0)
        return self.x_hat

    def error(self, x_bar: torch.Tensor) -> torch.Tensor:
        """x_bar: [E, C] (ya da ortak [C]) gelen x̄(t) → e_cer(t) = x̄(t) − x̂_smooth(t): [E, C] (ilk adımda 0)."""
        return (x_bar - self.x_hat) * self.valid

    def update(self, e_cer: torch.Tensor, mt: torch.Tensor, eta: torch.Tensor | None = None) -> None:
        """NLMS, x̂(t)'yi üreten φ(t−1) ile; mt: skor maskesi (burn-in'de güncelleme yok).

        eta: [E] koşul başına adım (None → config `eta`; 0 → W_cer donuk, kalıcı sürpriz için)."""
        st = self.phi / (self.phi.pow(2).sum(1, keepdim=True) + self.eps)  # [E, P]
        rate = self.eta * mt if eta is None else (eta * mt).view(-1, 1, 1)
        self.W.add_(rate * torch.einsum("ec,ep->ecp", e_cer, st))
