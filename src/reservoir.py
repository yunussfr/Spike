"""Sabit LIF rezervuarı (ARCHITECTURE.md §3.2).

K1: `W_in` ve `W_rec` sabit seed ile üretilir, buffer olarak tutulur (parametre değil),
hiçbir zaman güncellenmez. Bu sınıfın `update` metodu yoktur; olmamalıdır.

θ(τ): τ ızgarası üzerinde bir kez kalibre edilen sabit tablo, log-doğrusal interpolasyonla
okunur. Öğrenilmez, regülatörler doğrudan değiştiremez.

Nöron i'nin zaman sabiti τ_i = τ · m_i; m_i sabit seed ile bir kez üretilir (K1),
geometrik ortalaması 1'dir. Regülatör yalnızca ortak τ'yu verir.
"""

import math
from typing import Any

import torch
from torch import nn

from src.encoder import SpikeEncoder


def kwta(candidate: torch.Tensor, k: int) -> torch.Tensor:
    """Eşiği geçen (aday > 0) nöronlar arasından en büyük en fazla k tanesi ateşler (K7).

    candidate: [B, N] (V − θ) → s: [B, N] (0/1). Ateşleyen sayısı ≤ k;
    en az k nöron eşiği geçtiyse tam olarak k.
    """
    values, idx = candidate.topk(k, dim=-1)                     # [B, k]
    s = torch.zeros_like(candidate)
    s.scatter_(-1, idx, (values > 0).to(candidate.dtype))
    return s


def _as_column(value: float | torch.Tensor, B: int, like: torch.Tensor) -> torch.Tensor:
    """Skaler, [B] ya da [B, 1] → [B, 1]."""
    t = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return t.reshape(-1, 1).expand(B, 1) if t.numel() == 1 else t.reshape(B, 1)


class LIFReservoir(nn.Module):
    def __init__(
        self,
        n_in: int,
        N: int,
        dt: float,
        T: int,
        rho: float,
        theta_init: float,
        tau_grid: list[float],
        c_in: int,
        p_rec: float,
        rho_spec: float,
        rec_gain: float,
        tau_spread: float,
        reset: str,
        seed: int,
    ) -> None:
        super().__init__()
        if c_in > n_in:
            raise ValueError(f"c_in={c_in} girdi kanal sayısından ({n_in}) büyük olamaz")
        if reset not in ("subtract", "zero"):
            raise ValueError(f"reset 'subtract' ya da 'zero' olmalı, verilen: {reset}")
        self.n_in = n_in
        self.N = N
        self.dt = dt
        self.T = T
        self.rho = rho
        self.rec_gain = rec_gain
        self.reset = reset
        self.k = math.ceil(round(rho * N, 9))   # yuvarlama: 0.07*100 = 7.000000000000001 → 7

        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("W_in", self._init_w_in(N, n_in, c_in, gen))            # [N, n_in]
        self.register_buffer("W_rec", self._init_w_rec(N, p_rec, rho_spec, gen))     # [N, N]
        self.register_buffer("tau_mult", self._init_tau_mult(N, tau_spread, gen))    # [N]
        # θ(τ) tablosu: kalibrasyondan önce her yerde başlangıç tahmini
        self.register_buffer("theta_tau", torch.tensor(sorted(tau_grid)))           # [G]
        self.register_buffer("theta_table", torch.full((len(tau_grid),), theta_init))  # [G]
        self.register_buffer("calibrated", torch.tensor(False))

        self.V: torch.Tensor | None = None       # [B, N] zar potansiyeli
        self.s: torch.Tensor | None = None       # [B, N] son iç adımın spike'ları
        self.last_spike_count: torch.Tensor | None = None  # [B, T] iç adım başına ateşleyen sayısı

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], D_in: int, seed: int, calibrate: bool = True
    ) -> "LIFReservoir":
        """D_in: ham girdi boyutu; rezervuar girdi kanalı sayısı kodlayıcıdan hesaplanır."""
        r = cfg["reservoir"]
        res = cls(
            n_in=SpikeEncoder.n_channels(cfg, D_in), N=r["N"], dt=cfg["time"]["dt"], T=cfg["time"]["T"], rho=r["rho"],
            theta_init=r["theta_init"], tau_grid=r["calibration"]["tau_grid"], c_in=r["c_in"],
            p_rec=r["p_rec"], rho_spec=r["rho_spec"], rec_gain=r["rec_gain"],
            tau_spread=r["tau_spread"], reset=r["reset"], seed=seed,
        )
        if calibrate:
            res.calibrate_from_config(cfg, D_in, seed)
        return res

    @staticmethod
    def _init_w_in(N: int, n_in: int, c_in: int, gen: torch.Generator) -> torch.Tensor:
        """Her nöron rastgele c_in girdi kanalına N(0, 1) ağırlıkla bağlı."""
        W = torch.zeros(N, n_in)
        cols = torch.rand(N, n_in, generator=gen).argsort(dim=1)[:, :c_in]           # [N, c_in]
        W.scatter_(1, cols, torch.randn(N, c_in, generator=gen))
        return W

    @staticmethod
    def _init_w_rec(N: int, p_rec: float, rho_spec: float, gen: torch.Generator) -> torch.Tensor:
        """p_rec seyreklikte N(0, 1), öz-bağlantı yok, spektral yarıçap rho_spec'e ölçekli."""
        mask = torch.rand(N, N, generator=gen) < p_rec
        mask.fill_diagonal_(False)
        W = torch.randn(N, N, generator=gen) * mask
        radius = torch.linalg.eigvals(W.double()).abs().max().item()
        if radius == 0.0:
            return W
        return W * (rho_spec / radius)

    @staticmethod
    def _init_tau_mult(N: int, tau_spread: float, gen: torch.Generator) -> torch.Tensor:
        """m_i: log m_i ~ U(−log S, log S), geometrik ortalaması tam 1. S = 1 → hepsi 1."""
        log_s = math.log(tau_spread)
        log_m = (torch.rand(N, generator=gen) * 2 - 1) * log_s                       # [N]
        return (log_m - log_m.mean()).exp()

    # --- θ(τ) -----------------------------------------------------------------

    def theta(self, tau: torch.Tensor) -> torch.Tensor:
        """θ(τ): tablodan log-doğrusal interpolasyon, ızgara dışında uç değere sabit. [...] → [...]"""
        log_grid = self.theta_tau.log()                                     # [G]
        x = tau.log().clamp(log_grid[0], log_grid[-1])
        i = torch.searchsorted(log_grid, x.contiguous()).clamp(1, len(log_grid) - 1)
        x0, x1 = log_grid[i - 1], log_grid[i]
        y0, y1 = self.theta_table[i - 1], self.theta_table[i]
        return y0 + (x - x0) / (x1 - x0) * (y1 - y0)

    @torch.no_grad()
    def calibrate(
        self,
        u_seq: torch.Tensor,             # [S, T, n_in] kalibrasyon girdisi
        g: float,
        target_rate: float,
        washout: int,
        n_iter: int,
    ) -> torch.Tensor:
        """Her ızgara τ'sunda ortalama ateşleme oranı ≈ target_rate olacak θ'yı bulur.

        Oran θ'da monoton azalan olduğu için geometrik ikiye bölme kullanılır; tüm ızgara
        τ'ları paralel (batch) koşar. Tabloyu yazar ve ölçülen oranları döndürür: [G].
        """
        taus = self.theta_tau                                               # [G]
        G = len(taus)
        tau_col = taus.reshape(G, 1)                                        # [G, 1]
        g_col = _as_column(g, G, tau_col)                                   # [G, 1]

        def mean_rate(theta: torch.Tensor) -> torch.Tensor:                 # [G] → [G]
            self.reset_state()
            total = torch.zeros(G, dtype=taus.dtype, device=taus.device)
            for t in range(len(u_seq)):
                u = u_seq[t].unsqueeze(0).expand(G, -1, -1)                 # [G, T, n_in]
                f = self._integrate(u, g_col, tau_col, theta.reshape(G, 1))  # [G, N]
                if t >= washout:
                    total += f.mean(dim=-1)
            self.reset_state()
            return total / (len(u_seq) - washout)

        lo = torch.full((G,), 1e-3, dtype=taus.dtype, device=taus.device)
        hi = self.theta_table.clone()
        for _ in range(40):                                                 # üst sınırı bul
            too_active = mean_rate(hi) > target_rate
            if not too_active.any():
                break
            lo = torch.where(too_active, hi, lo)
            hi = torch.where(too_active, hi * 4, hi)
        for _ in range(n_iter):
            mid = (lo * hi).sqrt()
            too_active = mean_rate(mid) > target_rate
            lo = torch.where(too_active, mid, lo)
            hi = torch.where(too_active, hi, mid)

        self.theta_table.copy_((lo * hi).sqrt())
        self.calibrated.fill_(True)
        return mean_rate(self.theta_table)

    def calibrate_from_config(self, cfg: dict[str, Any], D_in: int, seed: int) -> torch.Tensor:
        """Config'teki ayarlarla rastgele kodlanmış girdi üretip kalibre eder.

        Kalibrasyon girdisi seviye aralığında düzgün dağılımlıdır: x ~ U(x_alt, x_üst).
        """
        c = cfg["reservoir"]["calibration"]
        washout = math.ceil(c["washout_tau_multiple"] * max(c["tau_grid"]) / (self.T * self.dt))
        gen = torch.Generator().manual_seed(seed)
        encoder = SpikeEncoder.from_config(cfg, D_in=D_in)
        if encoder.n_out != self.n_in:
            raise ValueError(f"kodlayıcı {encoder.n_out} kanal üretiyor, rezervuar {self.n_in} bekliyor")
        lo, hi = cfg["encoder"]["level"]["range"]
        x = lo + (hi - lo) * torch.rand(washout + c["n_measure"], 1, D_in, generator=gen)  # [S, 1, D_in]
        x = x.to(self.W_in.device)
        u_seq = torch.cat([encoder(x[t]) for t in range(len(x))])           # [S, T, n_in]
        return self.calibrate(
            u_seq, g=cfg["regulators"]["defaults"]["g"],
            target_rate=c["target_rate_frac"] * self.rho, washout=washout, n_iter=c["n_iter"],
        )

    # --- dinamik -----------------------------------------------------------------

    def reset_state(self) -> None:
        """Yeni dizi/bölüm başında çağrılır (K6). Dizi içinde asla çağrılmaz."""
        self.V = None
        self.s = None
        self.last_spike_count = None

    def _integrate(
        self,
        u: torch.Tensor,                 # [B, T, n_in]
        g_col: torch.Tensor,             # [B, 1]
        tau_col: torch.Tensor,           # [B, 1] ortak τ (ms)
        theta: torch.Tensor,             # [B, 1]
    ) -> torch.Tensor:
        """T iç adımı koşar, durumu günceller. Döndürür f = Σ s / T: [B, N]."""
        B, T, _ = u.shape
        if T != self.T:
            raise ValueError(f"u'nun iç adım sayısı {T}, beklenen {self.T}")
        if self.V is None:
            self.V = torch.zeros(B, self.N, dtype=u.dtype, device=u.device)
            self.s = torch.zeros_like(self.V)
        elif self.V.shape[0] != B:
            raise ValueError("batch boyutu değişti; önce reset_state() çağır")

        alpha_i = torch.exp(-self.dt / (tau_col * self.tau_mult))           # [B, N]
        alpha_bar = torch.exp(-self.dt / tau_col)                           # [B, 1]
        c = (1 - alpha_i) / (1 - alpha_bar)                                 # [B, N] göreli denge
        drive = g_col.unsqueeze(1) * (u @ self.W_in.T)                      # [B, T, N]
        total = torch.zeros_like(self.V)                                    # [B, N]
        counts = torch.empty(B, T, dtype=u.dtype, device=u.device)
        for t in range(T):
            I = drive[:, t] + self.rec_gain * (self.s @ self.W_rec.T)       # [B, N]
            self.V = alpha_i * self.V + c * I                               # ortak (1−ᾱ) yok
            self.s = kwta(self.V - theta, self.k)                           # [B, N]
            if self.reset == "subtract":
                self.V = self.V - theta * self.s                            # yumuşak sıfırlama
            else:
                self.V = self.V * (1 - self.s)
            total += self.s
            counts[:, t] = self.s.sum(dim=-1)
        self.last_spike_count = counts
        return total / T

    @torch.no_grad()
    def forward(
        self,
        u: torch.Tensor,                 # [B, T, n_in]
        g: float | torch.Tensor,         # skaler ya da [B]
        tau: float | torch.Tensor,       # ms, skaler ya da [B]
    ) -> torch.Tensor:
        """Bir ortam adımı (T iç adım). Döndürür r = [f, v]: [B, 2N]."""
        B = u.shape[0]
        g_col = _as_column(g, B, u)                                         # [B, 1]
        tau_col = _as_column(tau, B, u)                                     # [B, 1]
        f = self._integrate(u, g_col, tau_col, self.theta(tau_col))         # [B, N]
        return torch.cat([f, self.V], dim=-1)                               # [B, 2N]
