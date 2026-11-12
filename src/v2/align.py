"""Degenhart hizalaması (ARCHITECTURE.md §12.18, KR-049; Degenhart vd. 2020, Nat. Biomed. Eng.), kanal uzayında.

Day-0:     FA(x_kutu; q) → Λ_0 [C, q], μ_0 [C], Ψ_0 [C]
Yeni gün:  FA(son ≤ T_win s) → Λ_k, μ_k, Ψ_k;  kararlı kanal kümesi S, n_iter kez:
             O = argmin_{OᵀO = I} ‖Λ_0[S] − Λ_k[S]·O‖_F  (Procrustes);  d_c = ‖Λ_0[c] − Λ_k[c]·O‖;  S ← en küçük ⌈f·C_canlı⌉
           M_k = Λ_kᵀ (Λ_kΛ_kᵀ + Ψ_k)⁻¹;   x̂_0 = μ_0 + Λ_0·Oᵀ·M_k·(x − μ_k) = A·x + b
Akış:      kararlı kanal → yeniden ölçeklenmiş x;  kararsız ya da ölü → x̂_0.

Etiket kullanmaz; yeni günün FA'sı yalnızca akışın geçmiş girdisinden (KV6). FA EM ile, gradyan yok (KV2).
`Aligner` haritayı akış parçalarının arasında (CUDA Graph dışında) yeniden uydurur ve cihazdaki A, b, kararlı
maskesi ve hazır bayrağını yerinde günceller; graf bu tensörleri okur.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FA:
    lam: torch.Tensor                     # [C, q]
    mu: torch.Tensor                      # [C]
    psi: torch.Tensor                     # [C]


def bin_means(x: torch.Tensor, b: int) -> torch.Tensor:
    """x: [n, C] → [⌊n/b⌋, C] ardışık kutu ortalamaları."""
    nb = x.shape[0] // b
    return x[: nb * b].reshape(nb, b, x.shape[1]).mean(1)


@torch.no_grad()
def fa_fit(X: torch.Tensor, q: int, iters: int, psi_floor: float) -> FA:
    """Faktör analizi (EM, Rubin & Thayer). X: [n, C] → FA. float64'te; PCA ile başlatılır."""
    X = X.double()
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / max(1, X.shape[0] - 1)                                   # [C, C]
    C = S.shape[0]
    evals, evecs = torch.linalg.eigh(S)
    top = evals.flip(0)[:q].clamp(min=psi_floor)
    lam = evecs.flip(1)[:, :q] * top.sqrt()                                  # [C, q]
    psi = (torch.diagonal(S) - (lam ** 2).sum(1)).clamp(min=psi_floor)
    eye_q = torch.eye(q, dtype=S.dtype, device=S.device)
    for _ in range(iters):
        # (ΛΛᵀ + Ψ)⁻¹ Woodbury ile: Ψ⁻¹ − Ψ⁻¹Λ (I + ΛᵀΨ⁻¹Λ)⁻¹ ΛᵀΨ⁻¹
        pinv = 1.0 / psi                                                     # [C]
        lp = lam * pinv.unsqueeze(1)                                         # Ψ⁻¹Λ  [C, q]
        G = torch.linalg.inv(eye_q + lam.T @ lp)                             # [q, q]
        beta = (lp - lp @ G @ (lam.T @ lp)).T                                # Λᵀ(ΛΛᵀ + Ψ)⁻¹  [q, C]
        Ezz = eye_q - beta @ lam + beta @ S @ beta.T                         # [q, q]
        lam = S @ beta.T @ torch.linalg.inv(Ezz)
        psi = (torch.diagonal(S) - (lam * (S @ beta.T)).sum(1)).clamp(min=psi_floor)
    return FA(lam.float(), mu.float(), psi.float())


def latent_map(fa: FA) -> torch.Tensor:
    """M = Λᵀ (ΛΛᵀ + Ψ)⁻¹ : [q, C] (E[z | x] = M (x − μ))."""
    lam, psi = fa.lam.double(), fa.psi.double()
    lp = lam / psi.unsqueeze(1)
    G = torch.linalg.inv(torch.eye(lam.shape[1], dtype=lam.dtype, device=lam.device) + lam.T @ lp)
    return (lp - lp @ G @ (lam.T @ lp)).T.float()


def procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """argmin_{OᵀO = I} ‖A − B·O‖_F;  A, B: [m, q] → O: [q, q]."""
    U, _, Vh = torch.linalg.svd(B.double().T @ A.double())
    return (U @ Vh).float()


def stable_alignment(lam0: torch.Tensor, lamk: torch.Tensor, live: torch.Tensor, frac: float,
                     iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Degenhart kararlı kanal seçimi. lam0, lamk: [C, q], live: [C] 0/1 → O [q, q], kararlı maske [C] 0/1."""
    idx_live = live.nonzero().flatten()
    k = max(lam0.shape[1], math.ceil(frac * len(idx_live)))
    S = idx_live
    O = torch.eye(lam0.shape[1], device=lam0.device)
    for _ in range(iters):
        O = procrustes(lam0[S], lamk[S])
        d = (lam0 - lamk @ O).norm(dim=1)                                    # [C]
        d = torch.where(live > 0, d, torch.full_like(d, float("inf")))
        S = d.argsort()[:k]
    mask = torch.zeros_like(live)
    mask[S] = 1.0
    return O, mask


def channel_map(fa0: FA, fak: FA, O: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x̂_0 = μ_0 + Λ_0 Oᵀ M_k (x − μ_k) = A x + b → A: [C, C], b: [C]."""
    A = fa0.lam @ O.T @ latent_map(fak)                                      # [C, C]
    return A, fa0.mu - A @ fak.mu


@dataclass
class AlignSpec:
    q: int
    fa_bin: int                           # FA kutusu (adım)
    t_cal: int                            # ilk uydurmaya kadar biriken adım
    t_refit: int                          # yeniden uydurma aralığı (adım)
    t_win: int                            # FA penceresi üst sınırı (adım)
    stable_frac: float
    n_iter: int
    em_iters: int
    psi_floor: float
    a_on: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AlignSpec":
        a, dt = cfg["align"], cfg["time"]["dt_ms"]
        st = lambda s: int(round(1000.0 * s / dt))                           # noqa: E731
        return cls(int(a["q"]), int(round(a["fa_bin_ms"] / dt)), st(a["t_cal_s"]), st(a["t_refit_s"]), st(a["t_win_s"]),
                   float(a["stable_frac"]), int(a["n_iter"]), int(a["em_iters"]), float(a["psi_floor"]), float(a["a_on"]))


class Aligner:
    """Bir hizalama yuvası (mod 1: BOCPD a ≥ a_on'dan sonra; mod 2: akış başından). Graf dışında çağrılır."""

    def __init__(self, fa0: FA, spec: AlignSpec, gated: bool, device: torch.device) -> None:
        C = fa0.mu.shape[0]
        self.fa0, self.spec, self.gated = fa0, spec, gated
        self.A = torch.eye(C, device=device)                                 # [C, C]
        self.b = torch.zeros(C, device=device)                               # [C]
        self.stable = torch.ones(C, device=device)                           # [C] 0/1
        self.ready = torch.zeros((), device=device)                          # [] 0/1
        self.t_on: int | None = None
        self.last_fit: int | None = None
        self.n_fits = 0

    def reset_state(self) -> None:
        self.A.copy_(torch.eye(self.A.shape[0], device=self.A.device))
        self.b.zero_()
        self.stable.fill_(1.0)
        self.ready.zero_()
        self.t_on, self.last_fit, self.n_fits = None, None, 0

    @torch.no_grad()
    def maybe_refit(self, x: torch.Tensor, i: int, begin: int, a_now: float, live: torch.Tensor) -> bool:
        """x: [n, C] tüm akış girdisi (yalnız x[:i] okunur, nedensel); i: şu anki mutlak adım; begin: akış başı;
        a_now: BOCPD arousal'ı; live: [C] canlı kanallar. Harita güncellendiyse True."""
        sp = self.spec
        if self.t_on is None:
            if self.gated and a_now < sp.a_on:
                return False
            self.t_on = i if self.gated else begin
        if i - self.t_on < sp.t_cal or (self.last_fit is not None and i - self.last_fit < sp.t_refit):
            return False
        lo = max(self.t_on, i - sp.t_win)
        X = bin_means(x[lo:i], sp.fa_bin)
        if X.shape[0] < 2 * sp.q:
            return False
        fak = fa_fit(X, sp.q, sp.em_iters, sp.psi_floor)
        O, mask = stable_alignment(self.fa0.lam, fak.lam, live, sp.stable_frac, sp.n_iter)
        A, b = channel_map(self.fa0, fak, O)
        self.A.copy_(A)
        self.b.copy_(b)
        self.stable.copy_(mask)
        self.ready.fill_(1.0)
        self.last_fit = i
        self.n_fits += 1
        return True
