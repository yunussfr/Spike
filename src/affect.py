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

Regülatörler (Faz 5, ARCHITECTURE.md §3.5) z'den g, τ, η, β üretir; z'nin davranışa tek yolu budur.
Karar ve beyincik z'yi doğrudan görmez.
"""

import math

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
        self.freeze_norm = True                                              # False: donukken r normu nedensel güncellenir (KR-030)
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
        self.norm.frozen = self.frozen and self.freeze_norm
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


# --- Duygu dinamiği E/M/R (ARCHITECTURE.md §3.5, Faz 5) ------------------------------

class AffectDynamics:
    """z → E (fazik duygu), M (tonik duygudurum), R (karşıt süreç). Regülatörler E'yi okur.

      z_in  = z − μ_z;   μ_z ← μ_z + c_μ·(z − μ_z)
      drive = max(0, |E − M| − θ_R)·sign(E − M)
      R ← (1 − γ_decay)·R + γ_up·(1 − |R|)·drive
      λ_eff = λ_max·(1 − κ·|M|)
      E ← E + α_up·(1 − |E|)·z_in − α_down·(E − M) − λ_eff·R
      M ← (1 − β_M)·M + β_M·E
    E ve R yalnızca sonda [−1, 1]'e kırpılır; adım başına hız sınırı yok. Öğrenmez (`update` yok),
    δ ve e'yi görmez (K3). Durum yeni dizide sıfırlanır (K6).
    """

    def __init__(
        self,
        alpha_up: float,
        alpha_down: float,
        mood_rate: float,
        gamma_up: float,
        gamma_decay: float,
        threshold: float,
        lambda_max: float,
        kappa: float,
        center_rate: float,
    ) -> None:
        if not alpha_up > alpha_down:
            raise ValueError("alpha_up > alpha_down olmalı (duygu söndüğünden hızlı yükselir)")
        if not gamma_up < alpha_up:
            raise ValueError("gamma_up < alpha_up olmalı (önce duygu, sonra regülatör)")
        self.alpha_up, self.alpha_down, self.mood_rate = alpha_up, alpha_down, mood_rate
        self.gamma_up, self.gamma_decay, self.threshold = gamma_up, gamma_decay, threshold
        self.lambda_max, self.kappa, self.center_rate = lambda_max, kappa, center_rate
        self.E: torch.Tensor | None = None                                   # [B, k]
        self.M: torch.Tensor | None = None                                   # [B, k]
        self.R: torch.Tensor | None = None                                   # [B, k]
        self.mu: torch.Tensor | None = None                                  # [B, k] z'nin kayan ortalaması

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AffectDynamics":
        d = cfg["affect"]["dynamics"]
        return cls(alpha_up=d["alpha_up"], alpha_down=d["alpha_down"], mood_rate=d["mood_rate"],
                   gamma_up=d["gamma_up"], gamma_decay=d["gamma_decay"], threshold=d["threshold"],
                   lambda_max=d["lambda_max"], kappa=d["kappa"], center_rate=d["center_rate"])

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """z: [B, k] → {E, M, R}: her biri [B, k]."""
        if self.E is None:
            self.E, self.M, self.R = (torch.zeros_like(z) for _ in range(3))
            self.mu = z.clone()                                              # ilk adımda z_in = 0
        self.mu = self.mu + self.center_rate * (z - self.mu)
        z_in = z - self.mu                                                   # [B, k]
        error = self.E - self.M
        drive = torch.clamp(error.abs() - self.threshold, min=0.0) * error.sign()
        R = (1 - self.gamma_decay) * self.R + self.gamma_up * (1 - self.R.abs()) * drive
        R = R.clamp(-1.0, 1.0)
        lam = self.lambda_max * (1 - self.kappa * self.M.abs())              # [B, k]
        E = (self.E + self.alpha_up * (1 - self.E.abs()) * z_in
             - self.alpha_down * (self.E - self.M) - lam * R).clamp(-1.0, 1.0)
        self.M = (1 - self.mood_rate) * self.M + self.mood_rate * E
        self.E, self.R = E, R
        return {"E": self.E, "M": self.M, "R": self.R}

    __call__ = forward

    def reset_state(self) -> None:
        self.E = self.M = self.R = self.mu = None


# --- Regülatörler (ARCHITECTURE.md §3.5, Faz 5) ---------------------------------------

PARAMS = ("g", "tau", "eta", "beta")
LOG_SCALED = ("tau",)                                                        # K4: τ log-ölçekte


class Regulators:
    """z → {g, τ, η, β} (K4).

      ham = σ(W_r · z + b_r)                       [B, 4]
      g, η, β = alt + ham · (üst − alt)
      τ = exp(log τ_alt + ham_τ · (log τ_üst − log τ_alt))
    `tau_ceiling: c` verilirse τ asimetrik eşlenir (§3.5, KR-030): a = (W_r · z)_τ,
      log τ = log τ_0 + Δ_alt·tanh(a)  (a < 0, Δ_alt = log(τ_0/τ_alt))
      log τ = log τ_0 + Δ_üst·tanh(a)  (a ≥ 0, Δ_üst = log(min(c·τ_0, τ_üst)/τ_0))
    b_r, varsayılan değerleri üretecek şekilde başlatılır; W_r = 0 → başlangıçta çıktı = varsayılanlar.
    `enabled: false` (K5) iken çıktı her zaman varsayılanlardır ve update hiçbir şey yapmaz.

    Eşleme modları (`regulators.mode`):
      - `perturb`: W_r perturbasyonla öğrenilir (§3.5). Dönüşümlü pencereler: M adım referans
        (mevcut W_r) → W_r + N(0, σ²) ile M adım deneme → deneme penceresinin ortalama q'su referanstan
        büyükse tutulur, değilse geri alınır. Referans her denemeden hemen önce yeniden ölçülür;
        böylece veri kayarken eski bir referansla karşılaştırma yapılmaz.
      - `hand`: elle yazılmış eşleme (PHASES.md Faz 5 ara adımı). "Belirsizlik yüksekse η artar,
        τ kısalır": W_r'nin η satırı +c·w_u, τ satırı −c·w_u; w_u, duygu uzayının u başının z ağırlıkları
        (û_std = w_u·z + sabit). `set_hand_mapping` ile her adım güncellenir; g ve β varsayılanda kalır.
    """

    def __init__(
        self,
        k: int,
        defaults: dict[str, float],
        bounds: dict[str, list[float]],
        enabled: bool,
        mode: str,
        M: int,
        noise_std: float,
        hand_gain: float,
        seed: int,
        tau_ceiling: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if mode not in ("perturb", "hand"):
            raise ValueError(f"regulators.mode perturb | hand olmalı, verilen: {mode}")
        self.enabled, self.mode = enabled, mode
        self.M, self.noise_std, self.hand_gain = M, noise_std, hand_gain
        self.defaults = {p: float(defaults[p]) for p in PARAMS}
        self.lo = torch.tensor([self._to_lin(p, bounds[p][0]) for p in PARAMS], device=device)   # [4]
        self.hi = torch.tensor([self._to_lin(p, bounds[p][1]) for p in PARAMS], device=device)   # [4]
        pos = (torch.tensor([self._to_lin(p, self.defaults[p]) for p in PARAMS], device=device)
               - self.lo) / (self.hi - self.lo)                                                # [4] ∈ (0, 1)
        if not ((pos > 0) & (pos < 1)).all():
            raise ValueError("varsayılanlar sınırların içinde olmalı (K4)")
        self.b_r = torch.log(pos / (1 - pos))                                # [4] σ(b_r) = varsayılan konum
        self.tau_asym: tuple[float, float, float] | None = None              # (τ_0, Δ_alt, Δ_üst)
        if tau_ceiling is not None:
            if tau_ceiling <= 1:
                raise ValueError(f"regulators.tau_ceiling 1'den büyük olmalı, verilen: {tau_ceiling}")
            t0, (t_lo, t_hi) = self.defaults["tau"], bounds["tau"]
            self.tau_asym = (t0, math.log(t0 / t_lo), math.log(min(tau_ceiling * t0, t_hi) / t0))
        self.W_r = torch.zeros(len(PARAMS), k, device=device)                # [4, k]
        self.gen = torch.Generator().manual_seed(seed)                       # kendi üreteci: global RNG'ye dokunmaz
        self.frozen = False
        # Perturbasyon durumu
        self.phase = "ref"                                                   # ref | trial
        self.q_sum, self.n = 0.0, 0
        self.q_ref: float | None = None
        self.W_backup: torch.Tensor | None = None                            # [4, k]
        self.n_trials = self.n_accepted = 0

    @staticmethod
    def _to_lin(p: str, x: float) -> float:
        return math.log(x) if p in LOG_SCALED else x

    @classmethod
    def from_config(cls, cfg: dict[str, Any], seed: int,
                    device: torch.device | str = "cpu") -> "Regulators":
        r = cfg["regulators"]
        pt = r["perturbation"]
        return cls(k=cfg["affect"]["k"], defaults=r["defaults"], bounds=r["bounds"], enabled=r["enabled"],
                   mode=r["mode"], M=pt["M"], noise_std=pt["noise_std"], hand_gain=r["hand_gain"],
                   seed=seed + pt["seed_offset"], tau_ceiling=r["tau_ceiling"], device=device)

    def set_hand_mapping(self, w_u: torch.Tensor) -> None:
        """w_u: [k] — u başının z ağırlıkları. Yalnızca `hand` modunda."""
        self.W_r.zero_()
        self.W_r[PARAMS.index("eta")] = self.hand_gain * w_u
        self.W_r[PARAMS.index("tau")] = -self.hand_gain * w_u

    @torch.no_grad()
    def forward(self, z: torch.Tensor | None, B: int = 1) -> dict[str, torch.Tensor]:
        """z: [B, k] ya da None (ilk adım) → {g, τ, η, β}: her biri [B]."""
        if not self.enabled or z is None:
            return {p: torch.full((B if z is None else len(z),), self.defaults[p], device=self.lo.device)
                    for p in PARAMS}
        a = z @ self.W_r.T                                                   # [B, 4]
        raw = torch.sigmoid(a + self.b_r)                                    # [B, 4] ∈ (0, 1)
        lin = self.lo + raw * (self.hi - self.lo)                            # [B, 4]
        out = {p: (lin[:, i].exp() if p in LOG_SCALED else lin[:, i]) for i, p in enumerate(PARAMS)}
        if self.tau_asym is not None:
            t0, d_lo, d_hi = self.tau_asym
            a_t = torch.tanh(a[:, PARAMS.index("tau")])                      # [B]
            out["tau"] = t0 * torch.exp(torch.where(a_t < 0, d_lo * a_t, d_hi * a_t))   # a = 0 → tam τ_0
        return out

    __call__ = forward

    @torch.no_grad()
    def update(self, q: torch.Tensor) -> dict[str, Any] | None:
        """q: [B] (q = −‖δ‖², adım 10). Her adım çağrılır; pencere sonunda karar verir.

        Döndürür: deneme bittiyse {q_ref, q_trial, accepted}, değilse None.
        """
        if not self.enabled or self.frozen or self.mode != "perturb":
            return None
        self.q_sum += float(q.mean())
        self.n += 1
        if self.n < self.M:
            return None
        q_mean = self.q_sum / self.n
        self.q_sum, self.n = 0.0, 0
        if self.phase == "ref":
            self.q_ref = q_mean
            self.W_backup = self.W_r.clone()
            noise = torch.randn(self.W_r.shape, generator=self.gen) * self.noise_std
            self.W_r += noise.to(self.W_r.device)
            self.phase = "trial"
            return None
        accepted = q_mean > self.q_ref                                       # q büyük = hata küçük
        if not accepted:
            self.W_r = self.W_backup
        self.n_trials += 1
        self.n_accepted += int(accepted)
        self.phase = "ref"
        return {"q_ref": self.q_ref, "q_trial": q_mean, "accepted": accepted}

    def abort_trial(self) -> None:
        """Yarım kalan denemeyi geri alır ve pencereyi sıfırlar (ör. değerlendirmeye geçerken)."""
        if self.phase == "trial":
            self.W_r = self.W_backup
        self.phase, self.q_sum, self.n = "ref", 0.0, 0

    def reset_state(self) -> None:
        """Diziler arası: W_r ve perturbasyon penceresi kalır (pencereler dizileri aşabilir)."""
