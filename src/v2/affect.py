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


class AffectDynamics:
    """Dinamik duygu (KR-047): arousal düzey, valens eğilim, dominans düzey; gaz / fren / gevşetme / konsolidasyon.

        E_a ← E_a + λ_a·(mix − E_a),  mix = (1 − w_u)·s_int + w_u·u      s_int: iç beyincik hatası / ε²_0, u: başlık
        a   = tanh((E_a − 1)/(1 + ε))                                      uyuşmazlığı / u_0 (Day-0 tabanı 1, sabit)
        Q_f, Q_m, Q_d ← EMA(q),  q = mean e_v² / q_0                       okuma hatası (gecikmeli etiket)
        v   = tanh((Q_m − Q_f)/(Q_m + ε))                                  > 0: hata düşüyor (rahatlama)
        d   = −tanh(ln(Q_d + ε))                                           0: Day-0 düzeyi, → −1: eski ağırlıklar çalışmıyor
        b = v⁻·m (fren),  ℓ = a⁺·d⁻ (gevşetme),  c = v⁺·m (konsolidasyon),  m: son konsolidasyondan beri öğrenme ∈ [0, 1]
        η = η_0·(1 + α_η·a⁺)·(1 − ρ_v·v⁺)·(1 − b),   α = α_0·(1 − ℓ) + α_b·b

    a ve v farklı EMA'lardan gelir (a sabit Day-0 tabanına göre, v iki uyarlanan EMA'nın farkı): valens = −arousal
    çöküşü olmaz. Koşul başına [E] durum, yerinde güncellenir (CUDA Graph); maske 0 iken durum ilerlemez.
    Bayrak tensörleri (0/1): gas, brake, loosen, consolidate (ablasyonlar).
    """

    def __init__(self, w_u: torch.Tensor, lam_a: torch.Tensor, lam_f: torch.Tensor, lam_m: torch.Tensor,
                 lam_d: torch.Tensor, eps: float) -> None:
        """Hepsi [E]: başlık uyuşmazlığı ağırlığı ve EMA adımları (a: arousal; f, m: valens hızlı / orta; d: dominans)."""
        self.w_u, self.lam_a, self.lam_f, self.lam_m, self.lam_d, self.eps = w_u, lam_a, lam_f, lam_m, lam_d, eps
        E, dev = w_u.shape[0], w_u.device
        self.e_a = torch.ones(E, device=dev)                                 # Day-0 tabanı = 1
        self.q_f = torch.ones(E, device=dev)
        self.q_m = torch.ones(E, device=dev)
        self.q_d = torch.ones(E, device=dev)
        self.a = torch.zeros(E, device=dev)
        self.v = torch.zeros(E, device=dev)
        self.d = torch.zeros(E, device=dev)

    def set_internal(self, copy_norm: torch.Tensor, scale: torch.Tensor, k0: torch.Tensor) -> None:
        """İç (fazik) sürprizin normalizasyonu (§12.20, KR-050); hepsi [E].

        copy_norm 0: s = EMA(e²) / scale  (scale = ε²_0, §12.6 biçimi);
        copy_norm 1: s = EMA(e²) / (EMA(Δx̄²)·scale)  (scale = r_0, kopya oranı; k0 = Day-0 ort. Δx̄²).
        Başlangıçta s = 1 (Day-0 tabanı)."""
        self.copy = copy_norm
        self.scale = scale
        self.k_e = torch.where(copy_norm > 0, k0, torch.ones_like(k0))
        self.n_e = self.k_e * scale
        self.u_e = torch.ones_like(k0)

    def update_arousal(self, s_int: torch.Tensor, u: torch.Tensor, mt: torch.Tensor,
                       d1: torch.Tensor | None = None) -> torch.Tensor:
        """u: [E] Day-0 tabanına göre (taban 1); mt: [] maske → a: [E] (yerinde).

        d1 None: s_int [E] zaten tabana bölünmüş iç hata (§12.6): E_a ← E_a + λ_a((1 − w_u)s_int + w_u u − E_a).
        d1 [E] verilirse (set_internal sonrası): s_int = ham ort. e², d1 = ort. (x̄(t) − x̄(t−1))²; EMA'lar ayrı tutulur ve
        E_a = (1 − w_u)·EMA(e²)/(EMA(Δx̄²)·scale) + w_u·EMA(u) (kopya oranında) ya da EMA(e²)/scale (§12.6 ile aynı)."""
        if d1 is None:
            mix = (1.0 - self.w_u) * s_int + self.w_u * u
            self.e_a.add_((self.lam_a * mt) * (mix - self.e_a))
        else:
            la = self.lam_a * mt
            self.n_e.add_(la * (s_int - self.n_e))
            self.k_e.add_(la * self.copy * (d1 - self.k_e))
            self.u_e.add_(la * (u - self.u_e))
            s = self.n_e / (self.k_e.clamp(min=1e-12) * self.scale)
            self.e_a.copy_((1.0 - self.w_u) * s + self.w_u * self.u_e)
        self.a.copy_(torch.tanh((self.e_a - 1.0) / (1.0 + self.eps)))
        return self.a

    def update_valence(self, q: torch.Tensor, mt: torch.Tensor) -> None:
        """q: [E] okuma hatası / q_0 (e_v(t) = y_{t−d} − v̂_{t−d}); mt: o etiketin maskesi. v, d bir sonraki adıma."""
        self.q_f.add_((self.lam_f * mt) * (q - self.q_f))
        self.q_m.add_((self.lam_m * mt) * (q - self.q_m))
        self.q_d.add_((self.lam_d * mt) * (q - self.q_d))
        self.v.copy_(torch.tanh((self.q_m - self.q_f) / (self.q_m + self.eps)))
        self.d.copy_(-torch.tanh(torch.log(self.q_d + self.eps)))

    @staticmethod
    def controls(a: torch.Tensor, v: torch.Tensor, d: torch.Tensor, m: torch.Tensor, eta0: torch.Tensor,
                 alpha_eta: torch.Tensor, rho_v: torch.Tensor, alpha0: torch.Tensor, alpha_b: torch.Tensor,
                 gas: torch.Tensor, brake: torch.Tensor, loosen: torch.Tensor, consolidate: torch.Tensor,
                 gas_dom: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hepsi [E] → η [E] ∈ [0, η_0(1 + α_η)], α [E] ∈ [0, α_0 + α_b], fren b [E] ∈ [0, 1], konsolidasyon c [E] ∈ [0, 1].

        gas_dom = 1 → gaz a⁺·d⁻ (yalnız eski ağırlıklar çalışmazken; model uyum sağlayınca kendiliğinden kapanır)."""
        ap, vp, vn, dn = a.clamp(min=0.0), v.clamp(min=0.0), (-v).clamp(min=0.0), (-d).clamp(min=0.0)
        b = vn * m * brake
        ell = ap * dn * loosen
        c = vp * m * consolidate
        g_amt = ap if gas_dom is None else ap * torch.where(gas_dom > 0, dn, torch.ones_like(dn))
        eta = eta0 * (1.0 + alpha_eta * g_amt * gas) * (1.0 - rho_v * vp).clamp(min=0.0) * (1.0 - b)
        alpha = alpha0 * (1.0 - ell) + alpha_b * b
        return eta, alpha, b, c

    @staticmethod
    def routed_controls(a_eta: torch.Tensor, a_alpha: torch.Tensor, v: torch.Tensor, d: torch.Tensor, m: torch.Tensor,
                        eta0: torch.Tensor, alpha_eta: torch.Tensor, rho_v: torch.Tensor, alpha0: torch.Tensor,
                        alpha_b: torch.Tensor, gas: torch.Tensor, brake: torch.Tensor, loosen: torch.Tensor,
                        consolidate: torch.Tensor, gas_dom: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tonik / fazik iş bölümü (§12.20): η a_eta'dan (fazik), α, fren, konsolidasyon a_alpha'dan (tonik).

        a_eta = a_alpha ise `controls` ile aynıdır (fren ve konsolidasyon arousal'a bağlı değildir)."""
        eta, _, _, _ = AffectDynamics.controls(a_eta, v, d, m, eta0, alpha_eta, rho_v, alpha0, alpha_b, gas, brake,
                                               loosen, consolidate, gas_dom)
        _, alpha, b, c = AffectDynamics.controls(a_alpha, v, d, m, eta0, alpha_eta, rho_v, alpha0, alpha_b, gas, brake,
                                                 loosen, consolidate, gas_dom)
        return eta, alpha, b, c
