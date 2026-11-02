"""Tüm blokları bağlayan ana döngü (ARCHITECTURE.md §6).

Faz 3: adım 1–8, Faz 4: adım 9, Faz 5: adım 2 ve 10 (konfigürasyon C).
  1. u = encoder(x_t)
  2. g, τ, η, β = regulators(E_{t−1})          (ilk adımda ve K5 kapalıyken varsayılanlar; dinamik kapalıyken z_{t−1})
  3. r_t = reservoir(u, g, τ)
  4. ŷ_t = decision(r_t, β)
  5. x̂_{t+1} = cerebellum(r_t, ŷ_t)            (yalnızca B ve üstü)
  6. y_t ve x_{t+1} gelir
  7. δ_t = y_t − ŷ_t        → decision.update(δ_t, η)
  8. e_{t+1} = x_{t+1} − x̂_{t+1} → cerebellum.update(e_{t+1}, η)
  9. z_t = affect(r_t) → affect.update(q, u, s)    (affect.enabled; q, u, s δ ve e'den okunur)
     E_t = dynamics(z_t)                             (affect.dynamics.enabled; E/M/R, §3.5)
 10. regulators.update(q_t − q_gölge,t)              (her adım; M adımlık pencere sonunda perturbasyon kararı)
     Gölge: regülatörsüz eş ajan (A, kendi rezervuar kopyası), aynı veriyi görür (§3.5 eşli karşılaştırma).

Bir adım üç çağrıya bölünür: `perceive` (1–3), `act` (4–5), `learn` (6–9). Regülatörler
sabitken r_t öğrenmeden bağımsızdır; deneyler r'yi bir kez üretip `act`/`learn`'e verebilir.
C'de (regülatörler açık) bu geçerli değildir: g ve τ z_{t−1}'e bağlıdır, `perceive` her adım çağrılmalı.
K3: δ yalnızca karara, e yalnızca beyinciğe gider; duygu uzayı ikisini yalnızca okur. z davranışa
yalnızca regülatörler üzerinden etki eder. K6: x_{t+1}, x̂_{t+1} üretildikten sonra gelir.
g ve τ örnek başınadır ([B]); η ve β batch ortalamasıdır (güncellemeler batch üzerinden ortalanır).
"""

import copy
from typing import Any

import torch

from src.affect import HEADS, AffectDynamics, AffectSpace, AffectTargets, Regulators
from src.cerebellum import Cerebellum
from src.decision import DecisionReadout
from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir

CONFIGS = ("A", "B", "C")                                                    # C: B + duygu uzayı + regülatörler


class Agent:
    def __init__(self, cfg: dict[str, Any], D_in: int, D_out: int, seed: int,
                 reservoir: LIFReservoir | None = None, device: torch.device | str = "cpu") -> None:
        config = cfg["agent"]["config"]
        if config not in CONFIGS:
            raise ValueError(f"agent.config {CONFIGS} içinden olmalı, verilen: {config}")
        if cfg["regulators"]["enabled"] and config != "C":
            raise ValueError("regulators.enabled yalnızca agent.config: C ile kullanılabilir")
        if config == "C" and not cfg["affect"]["enabled"]:
            raise ValueError("C duygu uzayı gerektirir (affect.enabled: true)")
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
                           if config in ("B", "C") else None)
        self.affect = self.affect_targets = None
        if cfg["affect"]["enabled"]:
            if self.cerebellum is None:
                raise ValueError("duygu uzayı u = ‖e‖² için beyincik gerektirir (agent.config: B)")
            self.affect = AffectSpace.from_config(cfg, n_features=F, seed=seed, device=device)
            self.affect_targets = AffectTargets(cfg["affect"]["ema_short"], cfg["affect"]["ema_long"])
        self.dynamics = (AffectDynamics.from_config(cfg)
                         if self.affect is not None and cfg["affect"]["dynamics"]["enabled"] else None)
        self.regulators = (Regulators.from_config(cfg, seed=seed, device=device)
                           if config == "C" else None)                       # C'de enabled: false → K5 (sabit)
        self.z_prev: torch.Tensor | None = None                              # [B, k] regülatör girdisi (E ya da z)
        self.shadow: Agent | None = None                                     # eşli perturbasyon için gölge (A)
        if (self.regulators is not None and self.regulators.enabled
                and self.regulators.mode == "perturb" and cfg["regulators"]["perturbation"]["paired"]):
            cs = copy.deepcopy(cfg)
            cs["agent"]["config"] = "A"
            cs["affect"]["enabled"] = False
            cs["regulators"]["enabled"] = False
            self.shadow = Agent(cs, D_in=D_in, D_out=D_out, seed=seed,
                                reservoir=copy.deepcopy(self.reservoir), device=device)   # K1: sabit kopya
        self._r_shadow: torch.Tensor | None = None                           # [B, 2N]
        self._q_shadow: torch.Tensor | None = None                           # [B]
        self.r_t: torch.Tensor | None = None                                 # [B, 2N]
        self.y_hat: torch.Tensor | None = None                               # [B, D_out]
        self.x_hat: torch.Tensor | None = None                               # [B, D_in] x̂_{t+1}

    # --- bir adımın parçaları -------------------------------------------------------

    @torch.no_grad()
    def perceive(self, x_t: torch.Tensor) -> torch.Tensor:
        """Adım 1–3. x_t: [B, D_in] → r_t: [B, 2N]."""
        u = self.encoder(x_t)                                                # [B, T, C]
        self._regulate(len(x_t))                                             # adım 2
        if self.shadow is not None:
            self._r_shadow = self.shadow.perceive(x_t)                       # varsayılan g, τ
        return self.reservoir(u, g=self.g, tau=self.tau)                     # [B, 2N]

    @torch.no_grad()
    def _regulate(self, B: int) -> None:
        """Adım 2: g, τ, η, β = regulators(z_{t−1}). g, τ: [B]; η, β: batch ortalaması (skaler)."""
        if self.regulators is None or not self.regulators.enabled:
            return
        if self.regulators.mode == "hand":
            self.regulators.set_hand_mapping(self.affect.W_h[HEADS.index("u"), : self.affect.k])
        v = self.regulators(self.z_prev, B=B)
        self.g, self.tau = v["g"], v["tau"]                                  # [B]
        self.eta, self.beta = float(v["eta"].mean()), float(v["beta"].mean())

    @torch.no_grad()
    def act(self, r_t: torch.Tensor) -> torch.Tensor:
        """Adım 4–5. r_t: [B, 2N] → ŷ_t: [B, D_out]. x̂_{t+1} içeride saklanır."""
        self.r_t = r_t
        if self.shadow is not None:
            self.shadow.act(self._r_shadow)
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
        if self.shadow is not None:                                          # gölge kendi δ'sıyla öğrenir
            d_s = self.shadow.learn(y_t, None)["delta"]
            self._q_shadow = -(d_s ** 2).sum(dim=-1)                         # [B]
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
        out = {"z": z, "heads": heads, "targets": targets, "head_err": err}
        if self.dynamics is not None:
            out.update(self.dynamics(z))                                     # {E, M, R}: [B, k]
            self.z_prev = out["E"]                                           # sonraki adımın regülatör girdisi
        else:
            self.z_prev = z
        if self.regulators is not None:
            q = targets["q"] if self._q_shadow is None else targets["q"] - self._q_shadow
            trial = self.regulators.update(q)                                # adım 10 (donukken etkisiz)
            if trial is not None:
                out["trial"] = trial
        return out

    @torch.no_grad()
    def errors(self, y_t: torch.Tensor, x_next: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Öğrenme kapalıyken δ ve e (değerlendirme için)."""
        out = {"delta": y_t - self.y_hat}
        self._q_shadow = None                                                # donuk: perturbasyon yok
        if self.cerebellum is not None and x_next is not None:
            out["e"] = x_next - self.x_hat
            out.update(self._affect_step(out["delta"], out["e"]))
        return out

    def set_eval(self, frozen: bool) -> None:
        """Değerlendirme modu: ağırlıklar donar. Girdi normalizasyonu `agent.freeze_norm_eval` true ise donar,
        false ise nedensel EMA ile etiketsiz güncellenmeye devam eder (ARCHITECTURE.md §8, KR-030)."""
        for block in (self.decision, self.cerebellum):
            if block is not None and block.norm is not None:
                block.norm.frozen = frozen and self.freeze_norm_eval
        if self.affect is not None:
            self.affect.frozen = frozen                                      # ağırlıklar ve hedef istatistikleri
            self.affect.freeze_norm = self.freeze_norm_eval                  # r normu (freeze_norm_eval)
        if self.regulators is not None:
            if frozen:
                self.regulators.abort_trial()                                # yarım deneme geri alınır
            self.regulators.frozen = frozen
        if self.shadow is not None:
            self.shadow.set_eval(frozen)

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
        if self.regulators is not None:
            self.regulators.reset_state()
        if self.dynamics is not None:
            self.dynamics.reset_state()
        if self.shadow is not None:
            self.shadow.reset_state()
        self._r_shadow = self._q_shadow = None
        self.r_t = self.y_hat = self.x_hat = self.z_prev = None
