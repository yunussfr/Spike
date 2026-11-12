"""BOCPD değişim noktası sensörü (ARCHITECTURE.md §12.16, KR-049, KR-050; Adams & MacKay 2007).

Tek boyutlu skor üzerinde, Day-0 rejimine dönüşlü:
  v(t)   kaynak [C] (x_norm ya da donuk beyincik artığı);  o_k: B adımlık blok ortalaması  [C]
  z_k,c  = (o_k,c − m_c)/s_c   (x, cer_mon)   ya da   ((o_k,c − m_c) − ρ_c (o_{k−1},c − m_c))/s_c   (ar1)
  y_k    = ln(Σ_c canlı_c z²_k,c / Σ_c canlı_c + ε_y)
  Day-0 rejimi D : y ~ N(m_y, s²_y)                     (Day-0 train, sabit)
  Yeni rejim r   : y ~ Normal-Gamma(μ, κ, α, β) → Student-t;  önsel μ = m_y, κ = κ_new, α = α_new, β = α_new·s²_y
  log ℓ = τ_T log p(y);   Z = π_D ℓ_D + Σ π_r ℓ_r
  π'_D = π_D ℓ_D (1 − H) + H p_0 Z;   π'_{r+1} = π_r ℓ_r (1 − H);   π'_0 = H (1 − p_0) Z;   son koşu birikir
  a = 1 − π_D (arousal: P(şu anki segment Day-0 değil)),  n = Σ_{r < r_new} π_r (yakın zamanda değişim)

Durum yerinde güncellenir (CUDA Graph); blok sınırı Python tarafında bilinir (`step(..., block_end=True)`), maske 0
iken hiçbir şey ilerlemez. Öğrenmez (KV2): yalnızca Bayesçi durumunu ilerletir; etiket kullanmaz (KV6).
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

LOG_2PI = math.log(2.0 * math.pi)
SOURCES = ("x", "ar1", "cer_mon")


@dataclass
class BocpdSpec:
    block: int                            # blok uzunluğu (adım)
    hazard: float                         # blok başına değişim olasılığı H
    r_max: int                            # koşu uzunluğu kesmesi (blok)
    temper: float                         # τ_T
    kappa_new: float
    alpha_new: float
    r_new: int                            # n(t) için "yakın" koşu sınırı (blok)
    var_floor: float
    p0: float                             # değişimden sonra Day-0 rejimine dönüş olasılığı (0: dönüşsüz)
    score_eps: float                      # y = ln(ort z² + ε_y)
    source: str                           # x | ar1 | cer_mon

    def __post_init__(self) -> None:
        if not 0.0 <= self.p0 < 1.0:
            raise ValueError(f"bocpd.reentry_p0 [0, 1) aralığında olmalı: {self.p0}")
        if self.source not in SOURCES:
            raise ValueError(f"bocpd.source {SOURCES} içinden olmalı: {self.source}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], temper: float | None = None, source: str | None = None) -> "BocpdSpec":
        b, dt = cfg["bocpd"], cfg["time"]["dt_ms"]
        block = int(round(b["block_ms"] / dt))
        if abs(block * dt - b["block_ms"]) > 1e-9 or block < 1:
            raise ValueError(f"bocpd.block_ms ({b['block_ms']}) Δt'nin tam katı olmalı")
        blk_s = block * dt / 1000.0
        return cls(block, min(1.0, blk_s / b["hazard_s"]), int(b["r_max"]), float(temper if temper is not None else b["temper"]),
                   float(b["kappa_new"]), float(b["alpha_new"]), max(1, int(round(b["recent_s"] / blk_s))),
                   float(b["var_floor"]), float(b["reentry_p0"]), float(b["score_eps"]),
                   str(source if source is not None else b["source"]))


@dataclass
class ScoreRef:
    """Day-0 skor referansı (yalnız Day-0 train, KV6)."""
    m: torch.Tensor                       # [C] blok ortalamalarının ortalaması
    s: torch.Tensor                       # [C] standart sapma (ar1: yeniliğin)
    rho: torch.Tensor                     # [C] AR(1) katsayısı (x, cer_mon: 0)
    m_y: float                            # Day-0 skor ortalaması
    s2_y: float                           # Day-0 skor varyansı
    ar1: bool = False

    def to(self, device: torch.device) -> "ScoreRef":
        return ScoreRef(self.m.to(device), self.s.to(device), self.rho.to(device), self.m_y, self.s2_y, self.ar1)


def block_means(x: torch.Tensor, block: int) -> torch.Tensor:
    """x: [n, C] → [⌊n/B⌋, C] ardışık blok ortalamaları (artan kısım atılır)."""
    nb = x.shape[0] // block
    return x[: nb * block].reshape(nb, block, x.shape[1]).mean(1)


def block_scores(o: torch.Tensor, ref: ScoreRef, eps: float, live: torch.Tensor | None = None) -> torch.Tensor:
    """o: [nb, C] ardışık blok ortalamaları → y: [nb] (ar1: [nb − 1], ilk blok yeniliksiz)."""
    z = o - ref.m
    if ref.ar1:
        z = z[1:] - ref.rho * z[:-1]
    z = z / ref.s
    lv = torch.ones(o.shape[1], device=o.device) if live is None else live
    return torch.log((z.pow(2) * lv).sum(1) / lv.sum().clamp(min=1.0) + eps)


def fit_score_ref(o: torch.Tensor, source: str, eps: float, var_floor: float) -> ScoreRef:
    """Day-0 blok ortalamaları o: [nb, C] → ScoreRef (m, s, ρ kanal başına; m_y, s²_y skordan)."""
    o = o.double()
    m = o.mean(0)
    ar1 = source == "ar1"
    if ar1:
        zc = o - m
        rho = ((zc[1:] * zc[:-1]).sum(0) / zc[:-1].pow(2).sum(0).clamp(min=1e-12)).clamp(-0.999, 0.999)
        s = (zc[1:] - rho * zc[:-1]).std(0)
    else:
        rho = torch.zeros_like(m)
        s = o.std(0)
    s = s.clamp(min=math.sqrt(var_floor))
    ref = ScoreRef(m.float(), s.float(), rho.float(), 0.0, 1.0, ar1)
    y = block_scores(o.float(), ref, eps).double()
    ref.m_y, ref.s2_y = float(y.mean()), max(float(y.var()), var_floor)
    return ref


class BOCPD:
    def __init__(self, ref: ScoreRef, spec: BocpdSpec) -> None:
        if ref.ar1 != (spec.source == "ar1"):
            raise ValueError("ScoreRef ile BocpdSpec.source uyuşmuyor (ar1)")
        dev = ref.m.device
        self.spec, self.ref = spec, ref
        R, C = spec.r_max, ref.m.shape[0]
        self.log_1mh = math.log1p(-spec.hazard) if spec.hazard < 1.0 else -float("inf")
        self.log_h = math.log(spec.hazard) if spec.hazard > 0.0 else -float("inf")
        self.log_p0 = math.log(spec.p0) if spec.p0 > 0.0 else -float("inf")
        self.log_1mp0 = math.log1p(-spec.p0)
        self.log_pd = torch.zeros((), device=dev)                           # log π_D (başta 1)
        self.log_pr = torch.full((R,), -float("inf"), device=dev)            # log π_r
        self.mu = torch.full((R,), ref.m_y, device=dev)                      # [R] Normal-Gamma (skaler y)
        self.kappa = torch.full((R,), spec.kappa_new, device=dev)
        self.alpha = torch.full((R,), spec.alpha_new, device=dev)
        self.beta = torch.full((R,), spec.alpha_new * ref.s2_y, device=dev)
        self.bsum = torch.zeros(C, device=dev)
        self.bcnt = torch.zeros((), device=dev)
        self.o_prev = torch.zeros(C, device=dev)                             # ar1: önceki tam blok
        self.has_prev = torch.zeros((), device=dev)
        self.y = torch.zeros((), device=dev)                                 # son skor (loglanır)
        self.a = torch.zeros((), device=dev)
        self.n_recent = torch.zeros((), device=dev)

    def reset_state(self) -> None:
        sp, ref = self.spec, self.ref
        self.log_pd.zero_()
        self.log_pr.fill_(-float("inf"))
        self.mu.fill_(ref.m_y)
        self.kappa.fill_(sp.kappa_new)
        self.alpha.fill_(sp.alpha_new)
        self.beta.fill_(sp.alpha_new * ref.s2_y)
        for t in (self.bsum, self.bcnt, self.o_prev, self.has_prev, self.y, self.a, self.n_recent):
            t.zero_()

    def _loglik(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """y: [] → log ℓ_D [], log ℓ_r [R] (tavlanmış)."""
        t, ref = self.spec.temper, self.ref
        ld = -0.5 * (LOG_2PI + math.log(ref.s2_y) + (y - ref.m_y) ** 2 / ref.s2_y)
        nu = 2.0 * self.alpha
        sc2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        lt = (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0) - 0.5 * torch.log(nu * math.pi * sc2)
              - (nu + 1.0) / 2.0 * torch.log1p((y - self.mu) ** 2 / (nu * sc2)))
        return t * ld, t * lt

    def score(self, o: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        """Blok ortalaması o: [C] → y: [] (ar1'de o_prev kullanılır)."""
        ref = self.ref
        z = o - ref.m
        if ref.ar1:
            z = z - ref.rho * (self.o_prev - ref.m)
        z = z / ref.s
        return torch.log((z.pow(2) * live).sum() / live.sum().clamp(min=1.0) + self.spec.score_eps)

    def update_score(self, y: torch.Tensor, valid: torch.Tensor) -> None:
        """Skaler gözlem y: []; valid: [] 0/1 (0 → durum değişmez). Yerinde."""
        ld, lr = self._loglik(y)
        a_d = self.log_pd + ld
        a_r = self.log_pr + lr
        tot = torch.logsumexp(torch.cat([a_d.view(1), a_r]), 0)
        new_d = torch.logaddexp(a_d + self.log_1mh, self.log_h + self.log_p0 + tot)
        grow = a_r + self.log_1mh
        new_r = torch.empty_like(self.log_pr)
        new_r[1:] = grow[:-1]
        new_r[-1] = torch.logaddexp(grow[-2], grow[-1])                      # kesme: son koşu birikir
        new_r[0] = self.log_h + self.log_1mp0 + tot
        z = torch.logsumexp(torch.cat([new_d.view(1), new_r]), 0)
        new_d, new_r = new_d - z, new_r - z
        # Normal-Gamma güncellemesi, sonra kaydırma (koşu r + 1 = r'nin sonsalı; koşu 0 = önsel)
        sp, ref = self.spec, self.ref
        k1 = self.kappa + 1.0
        mu_u = (self.kappa * self.mu + y) / k1
        beta_u = self.beta + self.kappa / (2.0 * k1) * (y - self.mu) ** 2
        alpha_u = self.alpha + 0.5
        dev = y.device
        mu_n = torch.cat([torch.full((1,), ref.m_y, device=dev), mu_u[:-1]])
        kappa_n = torch.cat([torch.full((1,), sp.kappa_new, device=dev), k1[:-1]])
        alpha_n = torch.cat([torch.full((1,), sp.alpha_new, device=dev), alpha_u[:-1]])
        beta_n = torch.cat([torch.full((1,), sp.alpha_new * ref.s2_y, device=dev), beta_u[:-1]])
        v = valid > 0
        self.log_pd.copy_(torch.where(v, new_d, self.log_pd))
        self.log_pr.copy_(torch.where(v, new_r, self.log_pr))
        self.mu.copy_(torch.where(v, mu_n, self.mu))
        self.kappa.copy_(torch.where(v, kappa_n, self.kappa))
        self.alpha.copy_(torch.where(v, alpha_n, self.alpha))
        self.beta.copy_(torch.where(v, beta_n, self.beta))
        self.a.copy_(1.0 - torch.exp(self.log_pd))
        self.n_recent.copy_(torch.exp(self.log_pr[: sp.r_new]).sum())

    def update_block(self, o: torch.Tensor, live: torch.Tensor, valid: torch.Tensor) -> None:
        """Bir blok ortalaması o: [C]; live: [C] 0/1; valid: [] 0/1. ar1'de ilk geçerli blok yalnız o_prev'i doldurur."""
        y = self.score(o, live)
        ok = valid * self.has_prev if self.ref.ar1 else valid
        self.update_score(y, ok)
        v = valid > 0
        self.y.copy_(torch.where(ok > 0, y, self.y))
        self.o_prev.copy_(torch.where(v, o, self.o_prev))
        self.has_prev.copy_(torch.where(v, torch.ones_like(self.has_prev), self.has_prev))

    def step(self, vt: torch.Tensor, mt: torch.Tensor, live: torch.Tensor, block_end: bool) -> torch.Tensor:
        """vt: [C] kaynak sinyali, mt: [] maske, live: [C]; block_end: bu adım bloğun son adımı mı (Python) → a: []."""
        self.bsum.add_(mt * vt)
        self.bcnt.add_(mt)
        if block_end:
            full = (self.bcnt >= self.spec.block - 0.5).float()               # yalnız tamamı skorlanan bloklar
            self.update_block(self.bsum / self.bcnt.clamp(min=1.0), live, full)
            self.bsum.zero_()
            self.bcnt.zero_()
        return self.a


@torch.no_grad()
def bocpd_series(o: torch.Tensor, ref: ScoreRef, spec: BocpdSpec, live: torch.Tensor | None = None) -> torch.Tensor:
    """Blok ortalamaları o: [nb, C] (hepsi geçerli) → blok başına a: [nb] (kalibrasyon ve testler için)."""
    det = BOCPD(ref, spec)
    lv = torch.ones_like(ref.m) if live is None else live
    one = torch.ones((), device=o.device)
    out = torch.zeros(o.shape[0], device=o.device)
    for k in range(o.shape[0]):
        det.update_block(o[k], lv, one)
        out[k] = det.a
    return out


def calibrate_temper(o_hold: torch.Tensor, ref: ScoreRef, cfg: dict[str, Any], source: str | None = None
                     ) -> tuple[float, list]:
    """Day-0 train'in ayrılmış diliminin blok ortalamalarında (o_hold: [nb, C]) yanlış alarm oranı (a > 0,5 olan blok
    oranı) `bocpd.calib_false_alarm`'ı geçmeyen en büyük τ_T (en duyarlı). → (τ_T, [(τ, oran), …])"""
    b = cfg["bocpd"]
    rows, best = [], None
    for t in sorted(float(v) for v in b["temper_grid"]):
        fa = float((bocpd_series(o_hold, ref, BocpdSpec.from_config(cfg, temper=t, source=source)) > 0.5).float().mean())
        rows.append((t, fa))
        if fa <= b["calib_false_alarm"]:
            best = t
    if best is None:
        best = min(float(v) for v in b["temper_grid"])
    return best, rows
