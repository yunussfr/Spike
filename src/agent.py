"""Tüm blokları bağlayan ana döngü (ARCHITECTURE.md §6).

Faz 3: adım 1–8, Faz 4: adım 9. Regülatörler kapalı (K5): g, τ, η, β varsayılanda sabit.
  1. u = encoder(x_t)
  2. g, τ, η, β = varsayılanlar
  3. r_t = reservoir(u, g, τ)
  4. ŷ_t = decision(r_t, β)
  5. x̂_{t+1} = cerebellum(r_t, ŷ_t)            (yalnızca B ve üstü)
  6. y_t ve x_{t+1} gelir
  7. δ_t = y_t − ŷ_t        → decision.update(δ_t, η)
  8. e_{t+1} = x_{t+1} − x̂_{t+1} → cerebellum.update(e_{t+1}, η)
  9. z_t = affect(r_t) → affect.update(q, u, s)    (affect.enabled; q, u, s δ ve e'den okunur)

Bir adım üç çağrıya bölünür: `perceive` (1–3), `act` (4–5), `learn` (6–9). Regülatörler
sabitken r_t öğrenmeden bağımsızdır; deneyler r'yi bir kez üretip `act`/`learn`'e verebilir.
K3: δ yalnızca karara, e yalnızca beyinciğe gider; duygu uzayı ikisini yalnızca okur ve z hiçbir
bloğa geri verilmez (Faz 5'e kadar). K6: x_{t+1}, x̂_{t+1} üretildikten sonra gelir.
"""

from typing import Any

import torch

from src.affect import AffectSpace, AffectTargets
from src.cerebellum import Cerebellum
from src.decision import DecisionReadout
from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir

CONFIGS = ("A", "B")                                                         # C: Faz 5


class Agent:
    def __init__(self, cfg: dict[str, Any], D_in: int, D_out: int, seed: int,
                 reservoir: LIFReservoir | None = None, device: torch.device | str = "cpu") -> None:
        config = cfg["agent"]["config"]
        if config not in CONFIGS:
            raise ValueError(f"agent.config {CONFIGS} içinden olmalı, verilen: {config}")
        if cfg["regulators"]["enabled"]:
            raise NotImplementedError("regülatörler Faz 5'te; şimdilik regulators.enabled: false (K5)")
        self.cfg = cfg
        self.config = config
        self.freeze_norm_eval = cfg["agent"]["freeze_norm_eval"]
        defaults = cfg["regulators"]["defaults"]
        self.g, self.tau, self.eta, self.beta = defaults["g"], defaults["tau"], defaults["eta"], defaults["beta"]

        self.encoder = SpikeEncoder.from_config(cfg, D_in=D_in)
        self.reservoir = (reservoir if reservoir is not None
                          else LIFReservoir.from_config(cfg, D_in=D_in, seed=seed).to(device))
        F = 2 * self.reservoir.N
        self.decision = DecisionReadout.from_config(cfg, n_features=F, D_out=D_out, device=device)
        self.cerebellum = (Cerebellum.from_config(cfg, n_features=F, D_a=D_out, D_pred=D_in, device=device)
                           if config in ("B",) else None)
        self.affect = self.affect_targets = None
        if cfg["affect"]["enabled"]:
            if self.cerebellum is None:
                raise ValueError("duygu uzayı u = ‖e‖² için beyincik gerektirir (agent.config: B)")
            self.affect = AffectSpace.from_config(cfg, n_features=F, seed=seed, device=device)
            self.affect_targets = AffectTargets(cfg["affect"]["ema_short"], cfg["affect"]["ema_long"])
        self.r_t: torch.Tensor | None = None                                 # [B, 2N]
        self.y_hat: torch.Tensor | None = None                               # [B, D_out]
        self.x_hat: torch.Tensor | None = None                               # [B, D_in] x̂_{t+1}

    # --- bir adımın parçaları -------------------------------------------------------

    @torch.no_grad()
    def perceive(self, x_t: torch.Tensor) -> torch.Tensor:
        """Adım 1–3. x_t: [B, D_in] → r_t: [B, 2N]."""
        u = self.encoder(x_t)                                                # [B, T, C]
        return self.reservoir(u, g=self.g, tau=self.tau)                     # [B, 2N]

    @torch.no_grad()
    def act(self, r_t: torch.Tensor) -> torch.Tensor:
        """Adım 4–5. r_t: [B, 2N] → ŷ_t: [B, D_out]. x̂_{t+1} içeride saklanır."""
        self.r_t = r_t
        self.y_hat = self.decision(r_t, self.beta)                           # [B, D_out]
        if self.cerebellum is not None:
            self.x_hat = self.cerebellum(r_t, self.y_hat)                    # [B, D_in], a(t) = ŷ(t)
        return self.y_hat

    @torch.no_grad()
    def learn(self, y_t: torch.Tensor, x_next: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Adım 6–8. y_t: [B, D_out], x_next: [B, D_in] (dizi sonunda None). δ ve e'yi döndürür."""
        if self.y_hat is None:
            raise RuntimeError("learn'den önce act çağrılmalı")
        delta = y_t - self.y_hat                                             # [B, D_out]
        self.decision.update(delta, self.eta)                                # yalnızca δ → karar (K3)
        out = {"delta": delta}
        if self.cerebellum is not None and x_next is not None:
            e = x_next - self.x_hat                                          # [B, D_in]
            self.cerebellum.update(e, self.eta)                              # yalnızca e → beyincik (K3)
            out["e"] = e
            out.update(self._affect_step(delta, e))
        return out

    @torch.no_grad()
    def _affect_step(self, delta: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """Adım 9. δ ve e yalnızca okunur. Donukken (değerlendirme) yalnızca z ve baş hataları hesaplanır."""
        if self.affect is None:
            return {}
        z, heads = self.affect(self.r_t)                                     # [B, k], {q̂, û, ŝ}
        targets = self.affect_targets(delta, e)                              # {q, u, s}: [B]
        err = self.affect.update(targets["q"], targets["u"], targets["s"])   # [B, 3] standart ölçek
        return {"z": z, "heads": heads, "targets": targets, "head_err": err}

    @torch.no_grad()
    def errors(self, y_t: torch.Tensor, x_next: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Öğrenme kapalıyken δ ve e (değerlendirme için)."""
        out = {"delta": y_t - self.y_hat}
        if self.cerebellum is not None and x_next is not None:
            out["e"] = x_next - self.x_hat
            out.update(self._affect_step(out["delta"], out["e"]))
        return out

    def set_eval(self, frozen: bool) -> None:
        """Değerlendirme modu: normalizasyon istatistikleri de donar (agent.freeze_norm_eval)."""
        for block in (self.decision, self.cerebellum):
            if block is not None and block.norm is not None:
                block.norm.frozen = frozen and self.freeze_norm_eval
        if self.affect is not None:
            self.affect.frozen = frozen                                      # forward kendi norm'unu da dondurur

    def reset_state(self) -> None:
        """Yeni dizi başında (K6). Ağırlıklar kalır."""
        self.encoder.reset_state()
        self.reservoir.reset_state()
        self.decision.reset_state()
        if self.cerebellum is not None:
            self.cerebellum.reset_state()
        if self.affect is not None:
            self.affect.reset_state()
            self.affect_targets.reset_state()
        self.r_t = self.y_hat = self.x_hat = None
