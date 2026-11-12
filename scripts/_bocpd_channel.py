"""İlk (KR-049) BOCPD biçimi: kanal başına bağımsız Gauss Day-0 rejimi, dönüşsüz. Yalnız teşhis betiklerinin
(`diagnose_bocpd_models.py`) karşılaştırması için saklanır; üretim kodu src/v2/changepoint.py'dir (KR-050).

  Day-0 rejimi D : o_c ~ N(m_c, s²_c);  yeni rejim r: o_c ~ Normal-Gamma → Student-t;  log ℓ = τ_T Σ_c canlı_c log p(o_c)
  π'_D = π_D ℓ_D (1 − H);  π'_{r+1} = π_r ℓ_r (1 − H);  π'_0 = H (π_D ℓ_D + Σ π_r ℓ_r);  a = 1 − π_D
"""

import math
from typing import Any

import torch

LOG_2PI = math.log(2.0 * math.pi)


class ChannelBOCPD:
    def __init__(self, m: torch.Tensor, s2: torch.Tensor, spec: Any) -> None:
        """m, s2: [C] Day-0 blok istatistikleri; spec: block, hazard, r_max, temper, kappa_new, alpha_new, r_new, var_floor."""
        dev = m.device
        self.spec = spec
        self.m, self.s2 = m.clone(), s2.clamp(min=spec.var_floor).clone()
        R = spec.r_max
        self.log_1mh = math.log1p(-spec.hazard) if spec.hazard < 1.0 else -float("inf")
        self.log_h = math.log(spec.hazard) if spec.hazard > 0.0 else -float("inf")
        self.log_pd = torch.zeros((), device=dev)
        self.log_pr = torch.full((R,), -float("inf"), device=dev)
        self.mu = m.unsqueeze(0).repeat(R, 1).clone()
        self.kappa = torch.full((R,), spec.kappa_new, device=dev)
        self.alpha = torch.full((R,), spec.alpha_new, device=dev)
        self.beta = (spec.alpha_new * self.s2).unsqueeze(0).repeat(R, 1).clone()
        self.a = torch.zeros((), device=dev)

    def _loglik(self, o: torch.Tensor, live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.spec.temper
        ld = -0.5 * (LOG_2PI + torch.log(self.s2) + (o - self.m) ** 2 / self.s2)
        nu = 2.0 * self.alpha
        sc2 = self.beta * (self.kappa + 1.0).unsqueeze(1) / (self.alpha * self.kappa).unsqueeze(1)
        lt = (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)).unsqueeze(1) \
            - 0.5 * torch.log(nu.unsqueeze(1) * math.pi * sc2) \
            - ((nu + 1.0) / 2.0).unsqueeze(1) * torch.log1p((o - self.mu) ** 2 / (nu.unsqueeze(1) * sc2))
        return t * (ld * live).sum(), t * (lt * live).sum(1)

    def update_block(self, o: torch.Tensor, live: torch.Tensor, valid: torch.Tensor) -> None:
        ld, lr = self._loglik(o, live)
        a_d = self.log_pd + ld
        a_r = self.log_pr + lr
        new_d = a_d + self.log_1mh
        grow = a_r + self.log_1mh
        new_r = torch.empty_like(self.log_pr)
        new_r[1:] = grow[:-1]
        new_r[-1] = torch.logaddexp(grow[-2], grow[-1])
        new_r[0] = self.log_h + torch.logsumexp(torch.cat([a_d.view(1), a_r]), 0)
        z = torch.logsumexp(torch.cat([new_d.view(1), new_r]), 0)
        new_d, new_r = new_d - z, new_r - z
        k1 = self.kappa + 1.0
        mu_u = (self.kappa.unsqueeze(1) * self.mu + o) / k1.unsqueeze(1)
        beta_u = self.beta + (self.kappa / (2.0 * k1)).unsqueeze(1) * (o - self.mu) ** 2
        alpha_u = self.alpha + 0.5
        sp = self.spec
        mu_n = torch.cat([self.m.unsqueeze(0), mu_u[:-1]])
        kappa_n = torch.cat([torch.full((1,), sp.kappa_new, device=o.device), k1[:-1]])
        alpha_n = torch.cat([torch.full((1,), sp.alpha_new, device=o.device), alpha_u[:-1]])
        beta_n = torch.cat([(sp.alpha_new * self.s2).unsqueeze(0), beta_u[:-1]])
        v = valid > 0
        self.log_pd = torch.where(v, new_d, self.log_pd)
        self.log_pr = torch.where(v, new_r, self.log_pr)
        self.mu = torch.where(v, mu_n, self.mu)
        self.kappa = torch.where(v, kappa_n, self.kappa)
        self.alpha = torch.where(v, alpha_n, self.alpha)
        self.beta = torch.where(v, beta_n, self.beta)
        self.a = 1.0 - torch.exp(self.log_pd)
