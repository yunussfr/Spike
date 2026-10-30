"""Batch'li okuma blokları: M bağımsız ajanın (ör. 5 seed) okuma bloklarını tek tensörde yürütür.

Neden: adım adım akışta her işlem çok küçük (1 × 2N vektör); GPU tek bir ajanla dolmaz ve
çekirdek başlatma maliyeti baskın olur (KR-015). M ajanın aynı adımları aynı anda yürütülürse
GPU dolar. Matematik `src/agent.py`'deki tekil `Agent` ile birebir aynıdır (adım 4–9); her
ajanın kendi ağırlıkları, normalizasyon istatistikleri ve hedef ölçekleri vardır, ajanlar
birbirini hiçbir şekilde etkilemez. Eşdeğerlik `tests/test_batched.py` ile doğrulanır.

Batch'li ajanlar tekil ajanlardan kurulur (`BatchedAgent.from_agents`): başlangıç ağırlıkları
(ör. W_z'nin seed'li başlangıcı) aynen kopyalanır. Yalnızca `normalized` kural desteklenir.
Rezervuar burada yoktur: r dışarıdan verilir (regülatörler kapalıyken r öğrenmeden bağımsızdır, KR-016).

Şekiller: M ajan, F = 2N, D_out çıktı, D_in girdi, k duygu boyutu.
"""

from typing import Any

import torch

from src.agent import Agent


class BatchedStandardizer:
    """M ajan için OnlineStandardizer: her ajanın kendi μ, σ² istatistiği. [M, F] → [M, F+1]"""

    def __init__(self, mus: list[torch.Tensor], vars_: list[torch.Tensor], n_seen: int,
                 rate: float, eps: float, bias: float, device: torch.device) -> None:
        self.mu = torch.stack(mus).to(device)                                # [M, F]
        self.var = torch.stack(vars_).to(device)                             # [M, F]
        self.n_seen, self.rate, self.eps, self.bias = n_seen, rate, eps, bias
        self.frozen = False

    @torch.no_grad()
    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            self.n_seen += 1
            rate = max(self.rate, 1.0 / self.n_seen)
            self.mu = (1 - rate) * self.mu + rate * phi
            self.var = (1 - rate) * self.var + rate * (phi - self.mu) ** 2
        z = (phi - self.mu) / (self.var + self.eps).sqrt()                   # [M, F]
        return torch.cat([z, torch.full_like(z[:, :1], self.bias)], dim=-1)  # [M, F+1]

    @classmethod
    def from_singles(cls, norms: list[Any], device: torch.device) -> "BatchedStandardizer":
        n0 = norms[0]
        return cls([n.mu for n in norms], [n.var for n in norms], n0.n_seen, n0.rate, n0.eps, n0.bias, device)


def _nlms(err: torch.Tensor, phi: torch.Tensor, eps: float) -> torch.Tensor:
    """Ajan başına normalize LMS: err [M, D], phi [M, F] → ΔW [M, D, F] (η hariç)."""
    return err.unsqueeze(-1) * phi.unsqueeze(1) / (eps + (phi ** 2).sum(-1)).view(-1, 1, 1)


class BatchedAgent:
    def __init__(self, agents: list[Agent], device: torch.device) -> None:
        a0 = agents[0]
        for a in agents:
            for block in (a.decision, a.cerebellum):
                if block is not None and block.rule != "normalized":
                    raise NotImplementedError("batch'li ajan yalnızca normalized kuralı destekler")
        self.M = len(agents)
        self.device = device
        self.eta = a0.eta
        self.freeze_norm_eval = a0.freeze_norm_eval
        self.eps_d = a0.decision.eps
        self.dec_norm = BatchedStandardizer.from_singles([a.decision.norm for a in agents], device)
        self.W_a = torch.stack([a.decision.W_a for a in agents]).to(device)          # [M, D_out, F+1]
        self.has_cereb = a0.cerebellum is not None
        if self.has_cereb:
            self.eps_c = a0.cerebellum.eps
            self.cer_norm = BatchedStandardizer.from_singles([a.cerebellum.norm for a in agents], device)
            self.W_c = torch.stack([a.cerebellum.W_c for a in agents]).to(device)    # [M, D_in, F+A+1]
        self.has_affect = a0.affect is not None
        if self.has_affect:
            af = a0.affect
            self.k, self.lr_z, self.lr_head, self.eps_z = af.k, af.lr_z, af.lr_head, af.eps
            self.preact_l2 = af.preact_l2
            self.head_w = af.head_w.to(device)                                       # [3]
            self.aff_norm = BatchedStandardizer.from_singles([a.affect.norm for a in agents], device)
            self.W_z = torch.stack([a.affect.W_z for a in agents]).to(device)        # [M, k, F+1]
            self.W_h = torch.stack([a.affect.W_h for a in agents]).to(device)        # [M, 3, k+1]
            self.t_mu = torch.stack([a.affect.t_mu for a in agents]).to(device)      # [M, 3]
            self.t_var = torch.stack([a.affect.t_var for a in agents]).to(device)    # [M, 3]
            self.t_seen, self.t_rate = af.t_seen, af.t_rate
            self.a_short = a0.affect_targets.a_short
            self.a_long = a0.affect_targets.a_long
            self.m_short: torch.Tensor | None = None                                 # [M]
            self.m_long: torch.Tensor | None = None
            self.aff_frozen = False
        self.r_t = self.y_hat = self.x_hat = None

    # --- adım 4–5 -------------------------------------------------------------------

    @torch.no_grad()
    def act(self, r_t: torch.Tensor) -> torch.Tensor:
        """r_t: [M, F] (her ajanın kendi r'si) → ŷ: [M, D_out]."""
        self.r_t = r_t
        self.phi_d = self.dec_norm(r_t)                                              # [M, F+1]
        self.y_hat = torch.einsum("mf,mdf->md", self.phi_d, self.W_a)
        if self.has_cereb:
            self.phi_c = self.cer_norm(torch.cat([r_t, self.y_hat], dim=-1))         # a(t) = ŷ(t)
            self.x_hat = torch.einsum("mf,mdf->md", self.phi_c, self.W_c)            # [M, D_in]
        return self.y_hat

    # --- adım 6–9 -------------------------------------------------------------------

    @torch.no_grad()
    def learn(self, y_t: torch.Tensor, x_next: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """y_t: [M, D_out], x_next: [M, D_in] ya da None."""
        delta = y_t - self.y_hat
        self.W_a += self.eta * _nlms(delta, self.phi_d, self.eps_d)                  # yalnızca δ → karar (K3)
        out = {"delta": delta}
        if self.has_cereb and x_next is not None:
            e = x_next - self.x_hat
            self.W_c += self.eta * _nlms(e, self.phi_c, self.eps_c)                  # yalnızca e → beyincik (K3)
            out["e"] = e
            out.update(self._affect_step(delta, e, learn=True))
        return out

    @torch.no_grad()
    def errors(self, y_t: torch.Tensor, x_next: torch.Tensor | None) -> dict[str, torch.Tensor]:
        out = {"delta": y_t - self.y_hat}
        if self.has_cereb and x_next is not None:
            out["e"] = x_next - self.x_hat
            out.update(self._affect_step(out["delta"], out["e"], learn=False))
        return out

    @torch.no_grad()
    def _affect_step(self, delta: torch.Tensor, e: torch.Tensor, learn: bool) -> dict[str, torch.Tensor]:
        if not self.has_affect:
            return {}
        # forward: z = tanh(W_z · φ̃(r)), başlar
        self.aff_norm.frozen = self.aff_frozen
        phi = self.aff_norm(self.r_t)                                                # [M, F+1]
        a = torch.einsum("mf,mkf->mk", phi, self.W_z)                                # [M, k]
        z = torch.tanh(a)
        zb = torch.cat([z, torch.ones_like(z[:, :1])], dim=-1)                       # [M, k+1]
        heads = torch.einsum("mk,mhk->mh", zb, self.W_h)                             # [M, 3]
        raw = heads * (self.t_var + self.eps_z).sqrt() + self.t_mu
        # hedefler: δ ve e yalnızca okunur
        q = -(delta ** 2).sum(-1)
        u = (e ** 2).sum(-1)
        if self.m_short is None:
            self.m_short, self.m_long = u.clone(), u.clone()
        else:
            self.m_short = (1 - self.a_short) * self.m_short + self.a_short * u
            self.m_long = (1 - self.a_long) * self.m_long + self.a_long * u
        target = torch.stack([q, u, self.m_short - self.m_long], dim=-1)             # [M, 3]
        if not self.aff_frozen:
            self.t_seen += 1
            rate = max(self.t_rate, 1.0 / self.t_seen)
            self.t_mu = (1 - rate) * self.t_mu + rate * target
            self.t_var = (1 - rate) * self.t_var + rate * (target - self.t_mu) ** 2
        err = (target - self.t_mu) / (self.t_var + self.eps_z).sqrt() - heads        # [M, 3]
        if not self.aff_frozen:
            back = torch.einsum("mh,mhk->mk", err * self.head_w, self.W_h)[:, : self.k] * (1 - z ** 2)
            back = back - self.preact_l2 * a
            self.W_h += self.lr_head * _nlms(err, zb, self.eps_z)
            self.W_z += self.lr_z * _nlms(back, phi, self.eps_z)
        return {"z": z, "heads": raw, "targets": target, "head_err": err}

    def set_eval(self, frozen: bool) -> None:
        self.dec_norm.frozen = frozen and self.freeze_norm_eval
        if self.has_cereb:
            self.cer_norm.frozen = frozen and self.freeze_norm_eval
        if self.has_affect:
            self.aff_frozen = frozen

    @classmethod
    def from_agents(cls, agents: list[Agent], device: torch.device) -> "BatchedAgent":
        return cls(agents, device)


def gpu_chunks(n_rows: int, chunk: int) -> list[slice]:
    """r'yi GPU'ya parça parça taşımak için satır dilimleri."""
    return [slice(i, min(i + chunk, n_rows)) for i in range(0, n_rows, chunk)]

