"""AffectSpaceV2 (ARCHITECTURE.md §12.6, §12.13; KV3, KV4, KV5): beklenmedik sürprizden regülatörler.

    ε²(t)   = mean_c e_cer(t)²   (akış bunu Day-0 tabanına bölünmüş ve ops. girdi sapmasıyla karışık verir, KR-046)
    E_fast ← (1 − a_f)·E_fast + a_f·ε²
    E_slow ← (1 − a_s)·E_slow + a_s·ε²
    a(t)    = tanh((E_fast − E_slow) / (E_slow + ε))                 arousal ∈ (−1, 1)
    g_fast  = 1 + α_gate·a,  g_slow = 1 − α_gate·a
    η(t)    = η_0·(1 + α_η·max(0, a))
    η_in(t) = η_in,0·max(0, a)                                        sakinken 0 (KR-044)

Duygu uzayı öğrenmez (KV3): yalnızca durumunu ilerletir. Koşul başına (E) durum; güncellemeler yerinde ve
skor maskesiyle (burn-in'de durum ilerlemez). `enabled = 0` olan koşulda g = 1, η = η_0, η_in = η_in,0 (KV5).
"""

from typing import Any

import torch


class AffectSpaceV2:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.a_f, self.a_s = cfg["a_fast"], cfg["a_slow"]
        self.alpha_gate, self.eps = cfg["alpha_gate"], cfg["eps"]

    def reset_state(self, E: int, e0: float, device: torch.device, a_fast: torch.Tensor | None = None,
                    slow_on: torch.Tensor | None = None) -> None:
        """a_fast: [E] koşul başına E_fast adımı (None → config); slow_on: [E] 0 → E_slow e0'da sabit
        (kalıcı sürpriz: taban Day-0'da kalır, yeni güne alışıp sürprizi söndürmez)."""
        self.af = torch.full((E,), float(self.a_f), device=device) if a_fast is None else a_fast.clone()
        self.slow = torch.ones(E, device=device) if slow_on is None else slow_on.clone()
        self.e_fast = torch.full((E,), float(e0), device=device)
        self.e_slow = torch.full((E,), float(e0), device=device)
        self.a = torch.zeros(E, device=device)

    def update(self, eps2: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
        """eps2: [E] → a: [E] (yerinde)."""
        self.e_fast.add_((self.af * mt) * (eps2 - self.e_fast))
        self.e_slow.add_((self.a_s * self.slow * mt) * (eps2 - self.e_slow))
        self.a.copy_(torch.tanh((self.e_fast - self.e_slow) / (self.e_slow + self.eps)))
        return self.a

    def forward(self, enabled: torch.Tensor, gate: torch.Tensor, eta0: torch.Tensor, alpha_eta: torch.Tensor,
                eta_in0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Koşul başına [E] bayrak / değerler → g_fast [E], η [E], η_in [E] (g_slow = 2 − g_fast)."""
        ap = self.a.clamp(min=0.0)
        g_fast = 1.0 + self.alpha_gate * self.a * gate * enabled
        eta = eta0 * (1.0 + alpha_eta * ap * enabled)
        eta_in = torch.where(enabled > 0, eta_in0 * ap, eta_in0)
        return g_fast, eta, eta_in
