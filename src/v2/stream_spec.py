"""Akış koşulu ve parametre tanımları (ARCHITECTURE.md §12.9, §12.13–§12.18). `online_v5.py`'den ayrıldı (KR-049).

Condition: bir akış koşulu (E koşul aynı girdi akışında birlikte yürür). V7 alanları (KR-049): `sensor` (persist |
bocpd), `dead` (ölü kanal dedektörü), `align` (0 kapalı, 1 BOCPD kapılı, 2 akış başından Degenhart hizalaması).
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from src.v2.align import Aligner
from src.v2.changepoint import BOCPD
from src.v2.dead_channel import DeadChannelDetector
from src.v2.intent import Intent


@dataclass
class DynSpec:
    """Dinamik duygu parametreleri (KR-047); EMA adımları adım başına (λ = 1 − e^(−Δt/τ))."""
    w_u: float                      # arousal karışımında başlık uyuşmazlığının ağırlığı
    lam_a: float                    # arousal E_a adımı
    lam_f: float                    # valens hızlı EMA
    lam_m: float                    # valens orta EMA (λ_m < λ_f)
    lam_d: float                    # dominans EMA
    rho_v: float                    # rahatlamada η kısma: η·(1 − ρ_v·v⁺)
    alpha_b: float                  # frende ek çapa adımı
    lam_c: float                    # konsolidasyon adımı
    delta_m: float                  # m = clamp(‖W − W_anc‖ / (δ_m·‖W_0‖), 0, 1)
    gas: bool = True                # η'yı arousal'la aç
    brake: bool = True
    loosen: bool = True
    consolidate: bool = True
    a_mode: int = 0                 # kontrolü süren arousal: 0 iç (s_int + u), 1 sensör (kalıcı sürpriz), 2 ikisinin maks.,
                                    # 3 tonik / fazik iş bölümü (§12.20: η, g ← iç; α, λ_norm ← sensör)
    gas_dom: bool = False           # gaz a⁺·d⁻ (dominans düşükken)
    ph_norm: int = 0                # iç (fazik) sürprizin normalizasyonu: 0 ‖e‖²/ε²_0 (§12.6), 1 kopya oranı (§12.20)
    ph_on: bool = True              # False → iç arousal 0 (fazik kol kapalı, ablasyon)


@dataclass
class DynBase:
    """Day-0 tabanları (donuk model, Day-0 train sonundaki ayrılmış dilim): başlık uyuşmazlığı u_0, okuma hatası q_0;
    kopya oranı için (§12.20) donuk beyincikle r_0 = ort. e² / ort. Δx̄² ve d_0 = ort. Δx̄² (ölçülmediyse None)."""
    u0: float
    q0: float
    r0: float | None = None
    d0: float | None = None


@dataclass
class Condition:
    """Bir akış koşulu (§12.9). affect = False → η, η_in, λ_norm sabit (η_0, η_in,0, λ_calm), g = 1 (KV5).

    renorm = True → §12.14 kanal yeniden ölçeklemesi; λ_norm = λ_calm + (λ_surp − λ_calm)·max(0, a) (affect açıksa).
    *_seq verilirse (v2-C) skorlanan adımlarda η, η_in, λ_norm bu dizilerden gelir.
    """
    name: str
    eta0: float = 0.0
    eta_in0: float = 0.0
    affect: bool = False
    gate: bool = False
    alpha_eta: float = 0.0
    renorm: bool = False
    lam_calm: float = 0.0
    lam_surp: float = 0.0
    eta_seq: torch.Tensor | None = None        # [L]
    eta_in_seq: torch.Tensor | None = None     # [L]
    lam_seq: torch.Tensor | None = None        # [L]
    cer_raw: bool = False                      # beyincik hedefi ham x̄ (yeniden ölçeklenmemiş)
    cer_eta: float | None = None               # beyincik NLMS adımı (None → config; 0 → W_cer donuk)
    cer_src: int = -1                          # φ kaynağı: −1 kendi yolu; i → i. koşulun izleri ve v̂'si
    fixed_slow: bool = False                   # E_slow Day-0 tabanında sabit
    a_fast: float | None = None                # E_fast adımı (None → config)
    drift_w: float = 0.0                       # sürpriz karışımında girdi sapmasının ağırlığı
    drift_max: bool = False                    # sürpriz = max(beyincik, sapma) ("biri alarm verirse"); w yok sayılır
    gate_lo: float = 0.0                       # λ_norm kapısı h(a) = clamp((a − lo)/(hi − lo), 0, 1)
    gate_hi: float = 1.0
    lam_return: float = 0.0                    # (1 − h)·λ_ret ile m, v Day-0 referansına (m_0, v_0) çekilir
    dyn: DynSpec | None = None                 # dinamik duygu (KR-047)
    alpha_anchor: float | None = None          # sabit çapa adımı (None → akışın anchor_decay'i)
    alpha_seq: torch.Tensor | None = None      # [L] v2-C: α(t) dış dizisi (dyn gerekli)
    cons_seq: torch.Tensor | None = None       # [L] v2-C: konsolidasyon c(t) dış dizisi (dyn gerekli)
    sensor: str = "persist"                    # V7: "persist" (KR-046) | "bocpd" (§12.16: a = 1 − P(rejim = Day-0))
    dead: bool = False                         # V7: ölü kanal dedektörü (§12.17)
    align: int = 0                             # V7: Degenhart (§12.18) — 0 kapalı, 1 BOCPD kapılı, 2 akış başından


@dataclass
class V5Result:
    pred: torch.Tensor                    # [E, L, D] standart ölçek
    rate: torch.Tensor                    # [E, N]
    dw_rel: torch.Tensor                  # [E] ‖W − W_0‖ / ‖W_0‖ (okuma)
    din_rel: torch.Tensor                 # [E] ‖W_in − W_in,0‖ / ‖W_in,0‖ akış sonu
    din_max: torch.Tensor                 # [E] akış boyunca en büyük göreli sapma
    a: torch.Tensor                       # [E, L] arousal
    eta: torch.Tensor                     # [E, L]
    eta_in: torch.Tensor                  # [E, L]
    lam: torch.Tensor                     # [E, L] yeniden ölçekleme adımı λ_norm
    extra: dict = field(default_factory=dict)


@dataclass
class Plasticity:
    alpha_in: float
    delta_max: float
    kappa_psi: float
    kappa_x: float
    eps: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Plasticity":
        p, dt = cfg["plasticity"], cfg["time"]["dt_ms"]
        return cls(p["alpha_in"], p["delta_max"], math.exp(-dt / p["psi_tau_ms"]), math.exp(-dt / p["x_tau_ms"]),
                   p["eps"])


@dataclass
class V7Inputs:
    """V7 paylaşılan girdi uyumu bileşenleri (KR-049); hepsi koşullar arasında ortaktır, ham x_norm'dan beslenir.

    intent verilirse etiket kaynağı sönümlü ReFIT'tir (§12.15) ve etiket gecikmesi 0 olmalıdır."""
    bocpd: BOCPD | None = None
    dead: DeadChannelDetector | None = None
    aligners: dict[int, Aligner] = field(default_factory=dict)   # mod (1, 2) → yuva
    intent: Intent | None = None
    align_check: int = 500                     # adım adım yolda hizalama kontrol aralığı (graf yolunda parça sınırı)
    w_b: torch.Tensor | None = None            # [C, P] BOCPD cer_mon kaynağının donuk ridge'i (§12.16, KR-050)
