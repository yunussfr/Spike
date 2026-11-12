"""Faz V5 akışı: beyincik + duygu uzayı + duygu-kapılı yeniden ölçekleme / W_in plastisitesi
(ARCHITECTURE.md §12.5, §12.6, §12.13, §12.14; KR-044, KR-045).

E koşul aynı girdi akışında birlikte yürütülür; V4'ten (`online.py`) farkı: girdi ölçeklemesi ve W_in koşul başına
değişebildiği için omurga durumu da koşul başınadır (omurga batch'i = E). Her adım (prequential):
  0. koşul başına etiketsiz yeniden ölçekleme x → x'[e] (adım λ_norm[e], a_{t−1}'den); x̄[e] ← α_sm·x̄ + (1 − α_sm)·x'
  1. s = omurga(x'[e], W_in[e], ϑ[e]);  izler; r̃ = r ⊙ g(a_{t−1});  v̂_t = (1/K) Σ_k W^(k)·r̃^(k) + b^(k)   → kaydedilir
  2. beyincik: e_cer = x̄_t − x̂_t;  W_cer NLMS;  duygu: a_t;  η_t, η_in,t, λ_norm, g
  3. etiket y_{t−d}: e_v = y_{t−d} − v̂_{t−d}
     okuma:  W ← W − α·(W − W_0) + η_t·e_v·r̃_{t−d}ᵀ / (‖r̃_{t−d}‖² + ε)
     W_in:   W_in ← W_in − α_in·(W_in − W_in,0) + η_in,t·(W̄ᵀe_v ⊙ ψ̄_{t−d})·x̃_{t−d}ᵀ / (‖x̃_{t−d}‖² + ε), norm topu
  4. ψ̄, x̃ izleri; φ_t = [r̄_t (kapılanmamış), v̂_{t−1}];  x̂_{t+1} = W_cer·φ_t;  (ops.) homeostaz
Burn-in / dolgu adımlarında (maske 0) hiçbir öğrenme, duygu ilerlemesi ve oran sayımı yoktur.
Akış sonunda modelin hiçbir parametresi / buffer'ı değişmemiş olur (W_in, ϑ kopyalarla çalışılır).

Kalıcı sürpriz (KR-046, koşul başına seçenekler):
  - `cer_raw`: beyincik hedefi ham x̄ (Day-0 z-skorlu, yeniden ölçeklenmemiş) → ölçek düzelince hata sönmez
  - `cer_eta = 0`: W_cer donuk (yeni güne alışmaz);  `cer_src = i`: φ, i. koşulun (donuk A) yolundan (izleme yolu)
  - `fixed_slow`: E_slow Day-0 tabanında (1) sabit
  - `drift_w`: sürpriz ε² = (1 − w)·mean e_cer²/ε²_0 + w·δ(t)/δ_0  (δ: etiketsiz kanal sapması, `DriftMonitor`);
    `drift_max`: ε² = max(mean e_cer²/ε²_0, δ/δ_0) (iki algılayıcıdan biri yeter)
  - `gate_lo / gate_hi`: λ_norm = λ_calm + (λ_surp − λ_calm)·h,  h = clamp((a − lo)/(hi − lo), 0, 1)
  - `lam_return`: sakinken m, v ← m, v + λ_ret·(1 − h)·(m_0, v_0 − m, v) (kısa açılmanın kaydırması kalıcı olmaz)
Varsayılanlarla (False, None, −1, False, None, 0, 0, 1, 0) davranış KR-045 akışıyla aynıdır.

Dinamik duygu (KR-047, `dyn` verilen koşul; `AffectDynamics`):
  - sensör (yeniden ölçekleme kapısı): tüm koşullar için ortak izleme yolu — koşul 0'ın (donuk A) izleri, Day-0 W_cer,
    ham hedef — ve girdi sapması; koşulun `aff` durumu (a_fast, sabit taban, eşikli kapı, dönüş) bununla beslenir.
  - koşulun kendi beyinciği iç yoldur (`cer_raw = False`, `cer_src = −1`): s_int ve başlık uyuşmazlığı u → arousal a_int.
  - valens / dominans gecikmeli okuma hatasından; η, α_anchor ve konsolidasyon c bunlardan; çapa akış içinde W_anc
    (W_0'dan başlar, W_anc ← W_anc + λ_c·c·(W − W_anc)); g_fast = 1 + α_gate·a_int.
  - `alpha_anchor`: koşul başına sabit α (eşlenmiş kontroller); `alpha_seq` / `cons_seq`: v2-C dış dizileri.
"""

import torch

from src.v2.affect import AffectDynamics, AffectSpaceV2
from src.v2.cerebellum import CerebellumV2
from src.v2.drift import Drift, DriftMonitor, drift_baseline, drift_series  # noqa: F401 (dışa aktarım)
from src.v2.model import V2Model
from src.v2.online import Controls
from src.v2.intent import refit_label
from src.v2.stream_spec import Condition, DynBase, DynSpec, Plasticity, V5Result, V7Inputs  # noqa: F401 (dışa aktarım)


class _StreamV5:
    def __init__(self, model: V2Model, conds: list[Condition], cer: CerebellumV2, aff: AffectSpaceV2,
                 plast: Plasticity, alpha_sm: float, anchor_decay: float, eps: float, homeostasis: bool,
                 delay: int, ctrl: Controls, L: int, device: torch.device, drift: Drift | None = None,
                 dyn_base: DynBase | None = None, v7: V7Inputs | None = None) -> None:
        ro, bb = model.readout, model.backbone
        self.bb, self.ro, self.cer, self.aff, self.pl = bb, ro, cer, aff, plast
        E = len(conds)
        self.E = E
        f = lambda vals: torch.tensor(vals, device=device, dtype=torch.float32)   # noqa: E731
        self.enabled = f([float(c.affect) for c in conds])
        self.gate_on = f([float(c.gate) for c in conds])
        self.eta0 = f([c.eta0 for c in conds])
        self.alpha_eta = f([c.alpha_eta for c in conds])
        self.eta_in0 = f([c.eta_in0 for c in conds])
        self.has_seq = f([float(c.eta_seq is not None) for c in conds])
        self.rn = f([float(c.renorm) for c in conds]).view(E, 1)            # [E, 1]
        self.lam_calm = f([c.lam_calm for c in conds])
        self.lam_surp = f([c.lam_surp for c in conds])
        self.W0 = ro.w_anchor                                                # [K, D, N]
        self.W = self.W0.unsqueeze(0).repeat(E, 1, 1, 1).clone()             # [E, K, D, N]
        self.b = ro.b_anchor.unsqueeze(0)                                    # [1, K, D] donuk
        self.Win0 = bb.w_in.weight.detach()                                  # [N, C]
        self.Win = self.Win0.unsqueeze(0).repeat(E, 1, 1).clone()            # [E, N, C]
        self.win_norm = self.Win0.norm()
        self.din_max = torch.zeros(E, device=device)
        self.theta = bb.theta.unsqueeze(0).repeat(E, 1).clone()              # [E, N]
        self.rate_ema = (ctrl.homeo_target if ctrl.homeo_target is not None else bb.rate_ema).unsqueeze(0).repeat(E, 1).clone()
        self.target = ctrl.homeo_target if ctrl.homeo_target is not None else bb.target_rate
        self.hlr = ctrl.homeo_lr if ctrl.homeo_lr is not None else bb.homeo_lr
        self.alpha, self.eps, self.homeo, self.d, self.alpha_sm = anchor_decay, eps, homeostasis, delay, alpha_sm
        N, C, K = bb.N, bb.C, ro.K
        if any(c.renorm for c in conds) and ctrl.renorm_m0 is None:
            raise ValueError("yeniden ölçekleme koşulu için Controls.renorm_m0 / renorm_v0 (Day-0 train) gerekli")
        self.m0 = ctrl.renorm_m0 if ctrl.renorm_m0 is not None else torch.zeros(C, device=device)
        self.v0 = ctrl.renorm_v0 if ctrl.renorm_v0 is not None else torch.ones(C, device=device)
        self.reps = ctrl.renorm_eps
        self.rm = self.m0.unsqueeze(0).repeat(E, 1).clone()                 # [E, C]
        self.rv = self.v0.unsqueeze(0).repeat(E, 1).clone()
        self.lam_t = self.lam_calm.clone()                                   # bu adımın λ_norm'u (a_{t−1}'den)
        D = self.W0.shape[1]
        self.half = bb.half
        self.rate = torch.zeros(E, N, device=device)
        self.x_bar = torch.zeros(E, C, device=device)
        self.x_tr = torch.zeros(E, C, device=device)                         # x̃
        self.psi_tr = torch.zeros(E, N, device=device)                       # ψ̄
        self.v_prev = torch.zeros(E, D, device=device)
        self.g_fast = torch.ones(E, device=device)
        self.eta_t = self.eta0.clone()
        self.eta_in_t = self.eta_in0.clone()
        self.seq_eta = torch.zeros(E, L, device=device)
        self.seq_eta_in = torch.zeros(E, L, device=device)
        self.seq_lam = torch.zeros(E, L, device=device)
        for e, c in enumerate(conds):
            if c.eta_seq is not None:
                self.seq_eta[e] = c.eta_seq
                self.seq_eta_in[e] = c.eta_in_seq if c.eta_in_seq is not None else 0.0
                self.seq_lam[e] = c.lam_seq if c.lam_seq is not None else c.lam_calm
        dd = max(delay, 1)
        self.buf_r = torch.zeros(dd, E, K, N, device=device)                 # r̃_{t−d}
        self.buf_v = torch.zeros(dd, E, D, device=device)
        self.buf_y = torch.zeros(dd, D, device=device)
        self.buf_m = torch.zeros(dd, device=device)
        self.buf_psi = torch.zeros(dd, E, N, device=device)
        self.buf_x = torch.zeros(dd, E, C, device=device)
        # kalıcı sürpriz seçenekleri (KR-046)
        if cer.eps2_0 <= 0:
            raise ValueError("beyincik Day-0 hata tabanı ε²_0 pozitif olmalı")
        if any(c.cer_src >= E for c in conds) or any(c.gate_hi <= c.gate_lo for c in conds):
            raise ValueError("cer_src < E ve gate_lo < gate_hi olmalı")
        self.e2_0 = cer.eps2_0
        self.cer_raw = f([float(c.cer_raw) for c in conds]).view(E, 1)      # [E, 1]
        self.cer_eta = f([cer.eta if c.cer_eta is None else c.cer_eta for c in conds])
        self.src = torch.tensor([e if c.cer_src < 0 else c.cer_src for e, c in enumerate(conds)], device=device,
                                dtype=torch.long)
        self.drift_w = f([c.drift_w for c in conds])
        self.dmax = f([float(c.drift_max) for c in conds])
        self.gate_lo = f([c.gate_lo for c in conds])
        self.gate_hi = f([c.gate_hi for c in conds])
        self.lam_ret = f([c.lam_return for c in conds])
        self.has_ret = any(c.lam_return > 0 for c in conds)
        self.h_t = torch.zeros(E, device=device)                             # kapı h(a_{t−1}); burn-in'de 0
        self.x_bar_raw = torch.zeros(C, device=device)                       # ham x̄ (koşullar arasında ortak)
        self.s_drift = torch.zeros((), device=device)
        self.drift = None
        if any(c.drift_w > 0 or c.drift_max for c in conds):
            if drift is None or ctrl.renorm_m0 is None:
                raise ValueError("sapma terimi için Drift ve Controls.renorm_m0 / renorm_v0 (Day-0 train) gerekli")
            self.drift, self.delta0 = DriftMonitor(self.m0, self.v0, drift), drift.delta0
        self.plastic = any(c.eta_in0 > 0 or (c.eta_in_seq is not None and bool(c.eta_in_seq.abs().sum() > 0))
                           for c in conds)                                   # W_in kuralı yoksa ortak Linear yol
        # dinamik duygu (KR-047)
        self.has_dyn = any(c.dyn is not None for c in conds)
        if any((c.alpha_seq is not None or c.cons_seq is not None) and c.dyn is None for c in conds):
            raise ValueError("alpha_seq / cons_seq yalnızca dyn koşulunda")
        if self.has_dyn:
            c0 = conds[0]
            if dyn_base is None or dyn_base.u0 <= 0 or dyn_base.q0 <= 0:
                raise ValueError("dinamik duygu için pozitif Day-0 tabanları (DynBase) gerekli")
            if c0.dyn is not None or c0.renorm or c0.eta0 != 0 or c0.eta_in0 != 0 or c0.eta_seq is not None:
                raise ValueError("dinamik duygu: koşul 0 donuk A olmalı (ortak izleme yolu)")
        self.has_avec = self.has_dyn or any(c.alpha_anchor is not None for c in conds)
        self.alpha_c = f([anchor_decay if c.alpha_anchor is None else c.alpha_anchor for c in conds])   # [E]
        dz = DynSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False, False, False, False)   # dyn'siz: nötr
        ds = [c.dyn or dz for c in conds]
        self.is_dyn = f([float(c.dyn is not None) for c in conds])
        self.rho_v, self.alpha_b = f([s.rho_v for s in ds]), f([s.alpha_b for s in ds])
        self.lam_c, self.delta_m = f([s.lam_c for s in ds]), f([s.delta_m for s in ds])
        self.f_gas, self.f_brake = f([float(s.gas) for s in ds]), f([float(s.brake) for s in ds])
        self.f_loosen, self.f_cons = f([float(s.loosen) for s in ds]), f([float(s.consolidate) for s in ds])
        if any(s.a_mode not in (0, 1, 2, 3) for s in ds):
            raise ValueError("DynSpec.a_mode 0 (iç), 1 (sensör), 2 (maks.) ya da 3 (tonik / fazik) olmalı")
        if any(s.ph_norm not in (0, 1) for s in ds):
            raise ValueError("DynSpec.ph_norm 0 (‖e‖²/ε²_0) ya da 1 (kopya oranı) olmalı")
        self.a_mode, self.f_gas_dom = f([float(s.a_mode) for s in ds]), f([float(s.gas_dom) for s in ds])
        self.dyn = AffectDynamics(f([s.w_u for s in ds]), f([s.lam_a for s in ds]), f([s.lam_f for s in ds]),
                                  f([s.lam_m for s in ds]), f([s.lam_d for s in ds]), aff.eps)
        self.u0, self.q0 = (dyn_base.u0, dyn_base.q0) if dyn_base is not None else (1.0, 1.0)
        # fazik kol (§12.20, KR-050): kopya oranı Day-0 tabanı r_0, d_0 ister
        ph_copy = f([float(s.ph_norm == 1) for s in ds])
        if bool(ph_copy.sum() > 0) and (dyn_base is None or not dyn_base.r0 or not dyn_base.d0):
            raise ValueError("kopya oranı (ph_norm = 1) için DynBase.r0 ve d0 (Day-0 tabanı) gerekli")
        r0 = dyn_base.r0 if dyn_base is not None and dyn_base.r0 else 1.0
        d0 = dyn_base.d0 if dyn_base is not None and dyn_base.d0 else 1.0
        self.dyn.set_internal(ph_copy, torch.where(ph_copy > 0, torch.full_like(ph_copy, r0),
                                                   torch.full_like(ph_copy, cer.eps2_0)), torch.full_like(ph_copy, d0))
        self.ph_on = f([float(s.ph_on) for s in ds])
        self.tgt_prev = torch.zeros(E, C, device=device)                     # x̄'(t − 1) (kopya tahmini)
        self.tgt_valid = torch.zeros((), device=device)
        self.W_anc = self.W.clone()                                          # [E, K, D, N] akış içi çapa
        self.w0_norm = self.W0.norm()
        self.w_mon = cer.w0                                                  # [C, P] ortak izleme yolu (donuk)
        self.phi_mon = torch.zeros(cer.w0.shape[1], device=device)           # [P]
        self.x_hat_mon = torch.zeros(C, device=device)                       # [C]
        self.valid_mon = torch.zeros((), device=device)
        self.has_lseq = f([float(c.lam_seq is not None) for c in conds])
        self.has_aseq = f([float(c.alpha_seq is not None) for c in conds])
        self.has_cseq = f([float(c.cons_seq is not None) for c in conds])
        self.seq_alpha = torch.zeros(E, L, device=device)
        self.seq_cons = torch.zeros(E, L, device=device)
        for e, c in enumerate(conds):
            if c.alpha_seq is not None:
                self.seq_alpha[e] = c.alpha_seq
            if c.cons_seq is not None:
                self.seq_cons[e] = c.cons_seq
            if c.lam_seq is not None:
                self.seq_lam[e] = c.lam_seq
        self.zE = torch.zeros(E, device=device)                              # dyn'siz akışta boş çıktılar
        # V7 (KR-049): BOCPD sensörü, ölü kanal dedektörü, Degenhart hizalaması, sönümlü ReFIT etiketi
        v7 = v7 or V7Inputs()
        self.v7 = v7
        self.bocpd, self.dead_det = v7.bocpd, v7.dead
        self.is_bocpd = f([float(c.sensor == "bocpd") for c in conds])
        if any(c.sensor not in ("persist", "bocpd") for c in conds):
            raise ValueError("Condition.sensor 'persist' ya da 'bocpd' olmalı")
        if self.bocpd is None and any(c.sensor == "bocpd" for c in conds):
            raise ValueError("sensor = 'bocpd' koşulu için V7Inputs.bocpd gerekli")
        self.dead_on = f([float(c.dead) for c in conds]).view(E, 1)          # [E, 1]
        if self.dead_det is None and any(c.dead for c in conds):
            raise ValueError("dead = True koşulu için V7Inputs.dead gerekli")
        if any(c.align not in (0, 1, 2) for c in conds) or any(c.align and c.align not in v7.aligners for c in conds):
            raise ValueError("Condition.align 0, 1 ya da 2 olmalı ve yuvası V7Inputs.aligners'ta bulunmalı")
        if any(c.align == 1 for c in conds) and self.bocpd is None:
            raise ValueError("align = 1 (BOCPD kapılı) için V7Inputs.bocpd gerekli")
        self.has_align = any(c.align for c in conds)
        self.align_on = f([float(c.align > 0) for c in conds]).view(E, 1)    # [E, 1]
        self.slot = torch.tensor([max(c.align - 1, 0) for c in conds], device=device, dtype=torch.long)   # [E]
        self.A_s = torch.eye(C, device=device).unsqueeze(0).repeat(2, 1, 1).clone()   # [2, C, C] yuvalar (mod 1, 2)
        self.b_s = torch.zeros(2, C, device=device)
        self.st_s = torch.ones(2, C, device=device)
        self.rd_s = torch.zeros(2, device=device)
        for k, al in v7.aligners.items():                                    # Aligner tensörleri yuva görünümleri olur
            self.A_s[k - 1].copy_(al.A)
            self.b_s[k - 1].copy_(al.b)
            self.st_s[k - 1].copy_(al.stable)
            self.rd_s[k - 1].copy_(al.ready)
            al.A, al.b, al.stable, al.ready = self.A_s[k - 1], self.b_s[k - 1], self.st_s[k - 1], self.rd_s[k - 1]
        self.refit = v7.intent is not None
        if self.refit and delay != 0:
            raise ValueError("sönümlü ReFIT etiketi gecikmesizdir (label_delay = 0, §12.15)")
        if self.refit:
            self.y_mu, self.y_sd = v7.intent.mu, v7.intent.sd                # [D]
        self.zC = torch.zeros(C, device=device)
        self.oneC = torch.ones(C, device=device)
        self.a_b = torch.zeros((), device=device)
        # BOCPD kaynağı (§12.16, KR-050): cer_mon → izleme yolunda (koşul 0 = donuk A) donuk W_b'nin artığı
        self.b_cer = self.bocpd is not None and self.bocpd.spec.source == "cer_mon"
        if self.b_cer:
            c0 = conds[0]
            if v7.w_b is None:
                raise ValueError("bocpd.source = cer_mon için V7Inputs.w_b gerekli")
            if c0.dyn is not None or c0.renorm or c0.eta0 != 0 or c0.eta_in0 != 0 or c0.eta_seq is not None or c0.align:
                raise ValueError("bocpd.source = cer_mon: koşul 0 donuk A olmalı (izleme yolu)")
        self.w_b = v7.w_b
        self.x_hat_b = torch.zeros(C, device=device)
        self.has_mon = self.has_dyn or self.b_cer
        cer.reset_state(E)
        aff.reset_state(E, 1.0, device, a_fast=f([aff.a_f if c.a_fast is None else c.a_fast for c in conds]),
                        slow_on=f([0.0 if c.fixed_slow else 1.0 for c in conds]))   # ε² Day-0 tabanına bölünür

    def step(self, xt: torch.Tensor, yt: torch.Tensor, mt: torch.Tensor, j: int, seq_eta: torch.Tensor,
             seq_eta_in: torch.Tensor, seq_lam: torch.Tensor, seq_alpha: torch.Tensor,
             seq_cons: torch.Tensor, block_end: bool = False) -> tuple[torch.Tensor, ...]:
        """xt: [1, C], yt: [D] (ReFIT'te [D + 1] = [σ·û, m_int]), mt: [], seq_*: [E] o adımın dış dizisi;
        block_end: BOCPD bloğunun son adımı (Python) → v̂ [E, D], a, η, η_in, λ, s_cer, s_drift, a_int, valens, dominans,
        α, c, u, fren, s_sens, a_BOCPD, ölü kanal sayısı, ort. e_cer², ort. Δx̄'² [E]"""
        bb, ro, cer, aff, pl = self.bb, self.ro, self.cer, self.aff, self.pl
        E = self.E
        # V7 paylaşılan girdi bileşenleri (ham x_norm'dan; x(t) bu adımda biliniyor, nedensel)
        dead = self.dead_det.step(xt[0]) if self.dead_det is not None else self.zC   # [C] 0/1
        dead_e =self.dead_on * dead.unsqueeze(0)                            # [E, C] koşulun gördüğü ölü kanallar
        # 0. koşul başına etiketsiz yeniden ölçekleme (§12.14), adım λ_norm a_{t−1}'den; ölü kanal donar (§12.17)
        lam_used = self.lam_t.clone()
        dr = (lam_used * mt * self.rn[:, 0]).unsqueeze(1) * (1.0 - dead_e)   # [E, C]
        diff = xt - self.rm                                                  # [E, C]
        self.rm.add_(dr * diff)
        self.rv.add_(dr * (diff * diff - self.rv))
        if self.has_ret:                                                     # sakinken Day-0 referansına dönüş
            ret = (self.lam_ret * (1.0 - self.h_t) * mt * self.rn[:, 0]).unsqueeze(1)   # [E, 1]
            self.rm.add_(ret * (self.m0 - self.rm))
            self.rv.add_(ret * (self.v0 - self.rv))
        x_rn =self.m0 + (xt - self.rm) * torch.sqrt(self.v0 / (self.rv + self.reps))
        xe = torch.where(self.rn > 0, x_rn, xt.expand(E, -1))               # [E, C]
        xe = torch.where(dead_e > 0, self.m0.expand(E, -1), xe)              # ölü → Day-0 ortalaması (nötr)
        if self.has_align:                                                   # §12.18: kararsız / ölü kanal yeniden kurulur
            xa = torch.einsum("scd,d->sc", self.A_s, xt[0]) + self.b_s       # [2, C]
            use = (self.align_on * self.rd_s[self.slot].unsqueeze(1)
                   * torch.maximum(1.0 - self.st_s[self.slot], dead.unsqueeze(0)))   # [E, C]
            xe = torch.where(use > 0, xa[self.slot], xe)
        self.x_bar.mul_(self.alpha_sm).add_((1.0 - self.alpha_sm) * xe)
        self.x_bar_raw.mul_(self.alpha_sm).add_((1.0 - self.alpha_sm) * xt[0])   # [C] ham hedef (KR-046)
        if self.bocpd is not None:                                           # kaynak: x_norm ya da izleme yolu artığı
            vb = (self.x_bar_raw - self.x_hat_b) * self.valid_mon if self.b_cer else xt[0]
            self.a_b.copy_(self.bocpd.step(vb, mt, self.oneC - dead, block_end))
        if self.drift is not None:
            self.s_drift.copy_(self.drift.step(xt[0], mt) / self.delta0)     # δ(t)/δ_0, x(t)'ye kadar
        # 1. tahmin
        s = (bb.step(xe, w_in=self.Win, theta=self.theta) if self.plastic
             else bb.step(xe, theta=self.theta))                             # [E, N]
        ro.traces = ro.kappa * ro.traces + (1.0 - ro.kappa) * s.unsqueeze(0)   # [K, E, N]
        gvec = torch.cat([self.g_fast.unsqueeze(1).expand(E, self.half),
                          (2.0 - self.g_fast).unsqueeze(1).expand(E, bb.N - self.half)], 1)   # [E, N]
        r = ro.traces.transpose(0, 1) * gvec.unsqueeze(1)                    # [E, K, N] kapılı
        vk = torch.einsum("ekdn,ekn->ekd", self.W, r) + self.b               # [E, K, D] başlık başına
        v = vk.mean(1)                                                       # [E, D]
        # 2. beyincik + duygu (x̄_t artık biliniyor; x̂_t φ_{t−1}'den)
        tgt = torch.where(self.cer_raw > 0, self.x_bar_raw.expand(E, -1), self.x_bar)   # [E, C]
        e_cer = cer.error(tgt)                                               # [E, C]
        cer.update(e_cer, mt, self.cer_eta)
        e2 = e_cer.pow(2).mean(1)                                            # [E]
        s_cer = e2 / self.e2_0                                               # [E] Day-0 tabanına göre
        if self.has_dyn:                                                     # ortak sensör: A yolu, donuk W_cer, ham x̄
            s_mon = ((self.x_bar_raw - self.x_hat_mon) * self.valid_mon).pow(2).mean() / self.e2_0   # []
            s_sens = torch.where(self.is_dyn > 0, s_mon, s_cer)              # [E]
        else:
            s_sens = s_cer
        mix = torch.where(self.dmax > 0, torch.maximum(s_sens, self.s_drift),
                          (1.0 - self.drift_w) * s_sens + self.drift_w * self.s_drift)   # [E]
        aff.update(mix, mt)
        if self.bocpd is not None:                                           # V7: sensör arousal'ı = BOCPD a (§12.16)
            aff.a.copy_(torch.where(self.is_bocpd > 0, self.a_b.expand(E), aff.a))
        g_fast, eta, eta_in = aff.forward(self.enabled, self.gate_on, self.eta0, self.alpha_eta, self.eta_in0)
        if self.has_dyn:                                                     # KR-047: a_int, v, d → η, α, c, g
            u = vk.var(1, unbiased=False).mean(1) / self.u0                  # [E] başlık uyuşmazlığı / u_0
            d1 = (tgt - self.tgt_prev).pow(2).mean(1) * self.tgt_valid       # [E] kopya tahmininin hatası
            self.tgt_prev.copy_(tgt)
            self.tgt_valid.fill_(1.0)
            a_in = self.dyn.update_arousal(e2, u, mt, d1) * self.ph_on       # iç (fazik) arousal (§12.6 / §12.20)
            a_int = torch.where(self.a_mode == 0, a_in,                      # η ve g'yi süren arousal
                                torch.where(self.a_mode == 1, aff.a,
                                            torch.where(self.a_mode == 2, torch.maximum(a_in, aff.a), a_in)))
            a_alp = torch.where(self.a_mode == 3, aff.a, a_int)              # α'yı süren (iş bölümünde tonik)
            m_l = ((self.W - self.W_anc).flatten(1).norm(dim=1) / (self.delta_m * self.w0_norm)).clamp(0.0, 1.0)
            v_used, d_used = self.dyn.v.clone(), self.dyn.d.clone()          # e_v(t − 1)'e kadar (bu adımda kullanılan)
            eta_d, alpha_d, brk, cns = AffectDynamics.routed_controls(
                a_int, a_alp, v_used, d_used, m_l, self.eta0, self.alpha_eta, self.rho_v, self.alpha_c, self.alpha_b,
                self.f_gas, self.f_brake, self.f_loosen, self.f_cons, self.f_gas_dom)
            dmask = self.is_dyn > 0
            eta = torch.where(dmask, eta_d, eta)
            g_fast = torch.where(dmask, 1.0 + aff.alpha_gate * a_int * self.gate_on, g_fast)
            alpha_t = torch.where(self.has_aseq > 0, seq_alpha, torch.where(dmask, alpha_d, self.alpha_c))
            cons_t = torch.where(self.has_cseq > 0, seq_cons, cns * self.is_dyn)
            brk = brk * self.is_dyn
        else:
            alpha_t = self.alpha_c
        self.eta_t.copy_(torch.where(self.has_seq > 0, seq_eta, eta))
        self.eta_in_t.copy_(torch.where(self.has_seq > 0, seq_eta_in, eta_in))
        h = ((aff.a - self.gate_lo) / (self.gate_hi - self.gate_lo)).clamp(0.0, 1.0)   # [E]
        lam = self.lam_calm + (self.lam_surp - self.lam_calm) * h * self.enabled
        self.lam_t.copy_(torch.where(self.has_lseq > 0, seq_lam, lam))     # bir sonraki adımın λ_norm'u
        self.h_t.copy_(h * self.enabled)                                     # bir sonraki adımın kapısı (dönüş için)
        self.g_fast.copy_(g_fast)                                            # bir sonraki adımın kapısı
        # 3. gecikmeli etiket
        if self.d > 0:
            r_old, m_old = self.buf_r[j].clone(), self.buf_m[j].clone()
            psi_old, x_old = self.buf_psi[j].clone(), self.buf_x[j].clone()
            err = (self.buf_y[j] - self.buf_v[j]) * m_old                    # [E, D]
        elif self.refit:                                                     # §12.15: gecikmesiz sönümlü ReFIT
            D = v.shape[1]
            y_int = refit_label(v, yt[:D].unsqueeze(0), self.y_mu, self.y_sd)   # [E, D]
            r_old, m_old, err = r, mt * yt[D], (y_int - v) * (mt * yt[D])
            psi_old, x_old = None, None
        else:
            r_old, m_old, err = r, mt, (yt.unsqueeze(0) - v) * mt
            psi_old, x_old = None, None
        st = r_old / (r_old.pow(2).sum(-1, keepdim=True) + self.eps)         # [E, K, N]
        W_bar = self.W.mean(1) if self.plastic else None                     # [E, D, N] geri besleme (güncelleme öncesi)
        if self.has_avec:                                                    # koşul başına α(t); dyn'de akış içi çapa
            self.W.sub_((alpha_t * m_old).view(E, 1, 1, 1) * (self.W - (self.W_anc if self.has_dyn else self.W0)))
        else:
            self.W.sub_((self.alpha * m_old) * (self.W - self.W0))
        self.W.add_(self.eta_t.view(E, 1, 1, 1) * torch.einsum("ed,ekn->ekdn", err, st))
        if self.has_dyn:                                                     # konsolidasyon, sonra valens / dominans
            self.W_anc.add_((self.lam_c * cons_t * m_old).view(E, 1, 1, 1) * (self.W - self.W_anc))
            self.dyn.update_valence(err.pow(2).mean(1) / self.q0, m_old)    # e_v(t) → bir sonraki adımın v, d'si
        if self.plastic:
            # ψ̄, x̃ izleri (t anı) — d = 0 ise güncelleme bu adımın izleriyle
            psi = 1.0 / (1.0 + bb.surrogate_scale * (bb.v - self.theta).abs()) ** 2   # [E, N]
            self.psi_tr.mul_(pl.kappa_psi).add_((1.0 - pl.kappa_psi) * psi)
            self.x_tr.mul_(pl.kappa_x).add_((1.0 - pl.kappa_x) * xe)
            if self.d == 0:
                psi_old, x_old = self.psi_tr, self.x_tr
            L_sig = torch.einsum("ed,edn->en", err, W_bar) * psi_old         # [E, N]
            upd = torch.einsum("en,ec->enc", L_sig, x_old / (x_old.pow(2).sum(1, keepdim=True) + pl.eps))
            self.Win.sub_((pl.alpha_in * m_old) * (self.Win - self.Win0))
            self.Win.add_(self.eta_in_t.view(E, 1, 1) * upd)
            dev = self.Win - self.Win0
            nrm = dev.flatten(1).norm(dim=1)                                 # [E]
            scale = (pl.delta_max * self.win_norm / (nrm + 1e-12)).clamp(max=1.0)
            self.Win.copy_(self.Win0 + dev * scale.view(E, 1, 1))
            self.din_max.copy_(torch.maximum(self.din_max, (nrm * scale) / self.win_norm))
        if self.d > 0:
            self.buf_r[j] = r
            self.buf_v[j] = v
            self.buf_y[j] = yt
            self.buf_m[j] = mt
            if self.plastic:
                self.buf_psi[j] = self.psi_tr
                self.buf_x[j] = self.x_tr
        # 4. beyincik tahmini x̂_{t+1} (kapılanmamış izler, efference copy v̂_{t−1}; cer_src → izleme yolu)
        cer.forward(ro.traces.mean(0)[self.src], self.v_prev[self.src])
        if self.has_mon:                                                     # ortak izleme yolu tahmini x̂_{t+1} (koşul 0)
            self.phi_mon.copy_(torch.cat([ro.traces.mean(0)[0], self.v_prev[0]]))
            self.x_hat_mon.copy_(self.w_mon @ self.phi_mon)
            if self.b_cer:
                self.x_hat_b.copy_(self.w_b @ self.phi_mon)
            self.valid_mon.fill_(1.0)
        self.v_prev.copy_(v)
        self.rate.add_(mt * s)
        if self.homeo:
            dh = (1.0 - bb.rate_decay) * mt
            self.rate_ema.mul_(1.0 - dh).add_(dh * s)
            self.theta.add_(self.hlr * mt * (self.rate_ema - self.target)).clamp_(bb.theta_min, bb.theta_max)
        if self.has_dyn:
            dyn_out = (a_int.clone(), v_used, d_used, alpha_t.clone(), cons_t, u, brk, s_sens)
            d1_out = d1
        else:
            z = self.zE
            dyn_out = (z, z, z, alpha_t.clone(), z, z, z, s_sens)
            d1_out = z
        return (v, aff.a.clone(), self.eta_t.clone(), self.eta_in_t.clone(), lam_used, s_cer,
                self.s_drift.expand(E).clone()) + dyn_out + (self.a_b.expand(E).clone(), dead_e.sum(1), e2, d1_out)


def _persistent(st: _StreamV5) -> list[torch.Tensor]:
    """Graf parçaları arasında taşınan, yeniden atanan (yerinde güncellenmeyen) durum."""
    return [st.bb.v, st.bb.i_syn, st.bb.s, st.ro.traces]


def _set_persistent(st: _StreamV5, xs: list[torch.Tensor]) -> None:
    st.bb.v, st.bb.i_syn, st.bb.s, st.ro.traces = xs


@torch.no_grad()
def stream_adapt_v5(model: V2Model, x: torch.Tensor, y: torch.Tensor, start: int, stop: int, burnin: int,
                    conds: list[Condition], cer: CerebellumV2, aff: AffectSpaceV2, plast: Plasticity, alpha_sm: float,
                    anchor_decay: float, eps: float, homeostasis: bool, chunk: int = 0, label_delay: int = 0,
                    controls: Controls | None = None, drift: Drift | None = None,
                    dyn_base: DynBase | None = None, v7: V7Inputs | None = None) -> V5Result:
    """x: [n, C] x_norm, y: [n, D] standart hedef (cihazda). [start, stop) skorlanır; akış max(0, start − burnin)'de başlar.

    v7: BOCPD, ölü kanal, hizalama, ReFIT (KR-049). intent verilirse y yerine niyet girdisi kullanılır (y okunmaz);
    BOCPD blokları akış başından sayılır; hizalama haritası parça sınırlarında (graf dışında) yenilenir.
    chunk > 0 ve CUDA → CUDA Graph parçaları (chunk, label_delay'in ve BOCPD bloğunun katı); aksi halde adım adım.
    drift: sapma terimi (`drift_w > 0` olan koşul varsa gerekli, KR-046).
    dyn_base: dinamik duygu Day-0 tabanları (`dyn` koşulu varsa gerekli, KR-047).
    """
    model.eval()
    if not bool(model.readout.anchored):
        raise RuntimeError("çapa kaydedilmemiş: önce readout.save_anchor() (Aşama 2)")
    if model.backbone.mode != "plif":
        raise ValueError("V5 akışı yalnızca plif omurgasıyla (W_in plastisitesi, §12.13)")
    d = label_delay
    if chunk > 0 and d > 0 and chunk % d:
        raise ValueError(f"graph_chunk ({chunk}) etiket gecikmesinin ({d}) katı olmalı")
    dev = x.device
    begin = max(0, start - burnin)
    total, Ls = stop - begin, stop - start
    v7 = v7 or V7Inputs()
    blk = v7.bocpd.spec.block if v7.bocpd is not None else 0
    if chunk > 0 and blk > 0 and chunk % blk:
        raise ValueError(f"graph_chunk ({chunk}) BOCPD blok uzunluğunun ({blk}) katı olmalı")
    if v7.intent is not None:                                                # ReFIT: adım girdisi [σ·û, m_int]
        y = torch.cat([v7.intent.dd, v7.intent.m.unsqueeze(1)], 1)          # [n, D + 1]
    E, D = len(conds), y.shape[1] - (1 if v7.intent is not None else 0)
    for al in v7.aligners.values():
        al.reset_state()
    if v7.bocpd is not None:
        v7.bocpd.reset_state()
    if v7.dead is not None:
        v7.dead.reset_state()
    st = _StreamV5(model, conds, cer, aff, plast, alpha_sm, anchor_decay, eps, homeostasis, d,
                   controls or Controls(), Ls, dev, drift, dyn_base, v7)
    model.reset_state(E)
    # step çıktıları: v̂ [E, D], a, η, η_in, λ_norm, s_cer, s_drift, a_int, valens, dominans, α, c, u, fren, s_sens,
    # a_BOCPD, ölü kanal sayısı, ort. e_cer², ort. Δx̄'² [E]
    outs = [torch.empty(E, total, D, device=dev)] + [torch.zeros(E, total, device=dev) for _ in range(18)]
    Dy = y.shape[1]
    mask = torch.zeros(total, device=dev)
    mask[start - begin:] = 1.0
    seqs = [torch.zeros(E, total, device=dev) for _ in range(5)]            # η, η_in, λ, α, c (burn-in 0)
    for s_, src in zip(seqs, (st.seq_eta, st.seq_eta_in, st.seq_lam, st.seq_alpha, st.seq_cons)):
        s_[:, start - begin:] = src

    def j_of(i: int) -> int:
        return i % d if d else 0

    def bend(i: int) -> bool:                                                # BOCPD blok sonu (akış başından)
        return blk > 0 and (i + 1) % blk == 0

    def realign(i: int) -> None:                                             # graf dışında; i: akış içi indeks
        if not v7.aligners:
            return
        a_now = float(st.a_b) if v7.bocpd is not None else 0.0
        live = st.oneC - (v7.dead.dead if v7.dead is not None else st.zC)
        for al in v7.aligners.values():
            al.maybe_refit(x, begin + i, begin, a_now, live)

    if chunk > 0 and dev.type == "cuda":
        C = x.shape[1]
        sx = torch.zeros(chunk, C, device=dev)
        sy = torch.zeros(chunk, Dy, device=dev)
        sm = torch.zeros(chunk, device=dev)
        ss = [torch.zeros(E, chunk, device=dev) for _ in seqs]
        init = [t.clone() for t in _persistent(st)]
        _set_persistent(st, init)
        snap = _snapshot(st)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                        # ısınma; ardından tüm durum geri alınır
            st.step(sx[0:1], sy[0], sm[0], 0, *(s_[:, 0] for s_ in ss), block_end=bend(0))
        torch.cuda.current_stream().wait_stream(side)
        _restore(st, snap)
        _set_persistent(st, init)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            res = [st.step(sx[i: i + 1], sy[i], sm[i], j_of(i), *(s_[:, i] for s_ in ss), block_end=bend(i))
                   for i in range(chunk)]
            stk = [torch.stack([r_[k] for r_ in res], 1) for k in range(len(outs))]   # [E, chunk, ...]
        final = _persistent(st)
        n_full = total // chunk * chunk
        for c in range(0, n_full, chunk):
            realign(c)
            sx.copy_(x[begin + c: begin + c + chunk])
            sy.copy_(y[begin + c: begin + c + chunk])
            sm.copy_(mask[c: c + chunk])
            for b_, s_ in zip(ss, seqs):
                b_.copy_(s_[:, c: c + chunk])
            g.replay()
            for o, s_ in zip(outs, stk):
                o[:, c: c + chunk] = s_
            for a_, b_ in zip(init, final):
                a_.copy_(b_)
        torch.cuda.synchronize()
        del g, stk, res
        _set_persistent(st, init)
        rest = range(n_full, total)
    else:
        rest = range(total)
    for i in rest:
        if i % (chunk if chunk > 0 else v7.align_check) == 0:
            realign(i)
        r_ = st.step(x[begin + i: begin + i + 1], y[begin + i], mask[i], j_of(i), *(s_[:, i] for s_ in seqs),
                     block_end=bend(i))
        for o, v_ in zip(outs, r_):
            o[:, i] = v_

    W, W0 = st.W, st.W0
    dw = (W - W0).flatten(1).norm(dim=1) / W0.flatten().norm()
    din = (st.Win - st.Win0).flatten(1).norm(dim=1) / st.win_norm
    anc = (st.W_anc - W0).flatten(1).norm(dim=1) / W0.flatten().norm()
    model.reset_state(1)
    sl = slice(start - begin, None)
    (pred, a_o, eta_o, ein_o, lam_o, scer_o, sdr_o, aint_o, val_o, dom_o, alp_o, cons_o, u_o, brk_o,
     ssens_o, ab_o, nd_o, e2_o, d1_o) = (o[:, sl] for o in outs)
    return V5Result(pred, st.rate / Ls, dw, din, st.din_max.clone(), a_o, eta_o, ein_o, lam_o,
                    {"theta": st.theta.clone(), "s_cer": scer_o, "s_drift": sdr_o, "a_int": aint_o, "val": val_o,
                     "dom": dom_o, "alpha": alp_o, "cons": cons_o, "u": u_o, "brake": brk_o, "s_sens": ssens_o,
                     "anc_rel": anc, "a_bocpd": ab_o, "n_dead": nd_o, "e2": e2_o, "d1": d1_o,
                     "align_fits": {k: al.n_fits for k, al in v7.aligners.items()}})

_MUTABLE = ("W", "Win", "din_max", "theta", "rate_ema", "rate", "x_bar", "x_tr", "psi_tr", "v_prev", "g_fast",
            "eta_t", "eta_in_t", "lam_t", "rm", "rv", "buf_r", "buf_v", "buf_y", "buf_m", "buf_psi", "buf_x",
            "x_bar_raw", "s_drift", "h_t", "W_anc", "phi_mon", "x_hat_mon", "valid_mon", "a_b", "x_hat_b", "tgt_prev",
            "tgt_valid")
_BOCPD_STATE = ("log_pd", "log_pr", "mu", "kappa", "alpha", "beta", "bsum", "bcnt", "a", "n_recent", "o_prev",
                "has_prev", "y")


def _snapshot(st: _StreamV5) -> dict[str, torch.Tensor]:
    snap = {k: getattr(st, k).clone() for k in _MUTABLE}
    for k in ("W", "phi", "x_hat", "valid"):
        snap["cer_" + k] = getattr(st.cer, k).clone()
    for k in ("e_fast", "e_slow", "a"):
        snap["aff_" + k] = getattr(st.aff, k).clone()
    for k in ("e_a", "q_f", "q_m", "q_d", "a", "v", "d", "n_e", "k_e", "u_e"):
        snap["dyn_" + k] = getattr(st.dyn, k).clone()
    if st.drift is not None:
        snap["drift_m"], snap["drift_v"] = st.drift.m.clone(), st.drift.v.clone()
    if st.bocpd is not None:
        for k in _BOCPD_STATE:
            snap["bocpd_" + k] = getattr(st.bocpd, k).clone()
    if st.dead_det is not None:
        snap["dead_z"], snap["dead_dead"] = st.dead_det.z.clone(), st.dead_det.dead.clone()
    return snap


def _restore(st: _StreamV5, snap: dict[str, torch.Tensor]) -> None:
    for k, v in snap.items():
        if k.startswith("cer_"):
            getattr(st.cer, k[4:]).copy_(v)
        elif k.startswith("aff_"):
            getattr(st.aff, k[4:]).copy_(v)
        elif k.startswith("dyn_"):
            getattr(st.dyn, k[4:]).copy_(v)
        elif k.startswith("drift_"):
            getattr(st.drift, k[6:]).copy_(v)
        elif k.startswith("bocpd_"):
            getattr(st.bocpd, k[6:]).copy_(v)
        elif k.startswith("dead_"):
            getattr(st.dead_det, k[5:]).copy_(v)
        else:
            getattr(st, k).copy_(v)


@torch.no_grad()
def collect_cerebellum_data(model: V2Model, x: torch.Tensor, start: int, stop: int, burnin: int,
                            alpha_sm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Donuk model, [start, stop) aralığında (öncesinde burn-in): φ(t) = [r̄(t), v̂(t−1)] ve hedef x̄(t+1).

    Beyincik ridge başlatması için (Aşama 2b, yalnızca train dilimi). → phi: [T−1, N + D], target: [T−1, C]"""
    model.eval()
    begin = max(0, start - burnin)
    model.reset_state(1)
    C = x.shape[1]
    D = model.readout.weight.shape[1]
    x_bar = torch.zeros(C, device=x.device)
    v_prev = torch.zeros(1, D, device=x.device)
    phis, bars = [], []
    for t in range(begin, stop):
        x_bar = alpha_sm * x_bar + (1.0 - alpha_sm) * x[t]
        s = model.backbone.step(x[t: t + 1])
        v, r_bar = model.readout.step(s)                                     # g = 1 → kapılanmamış
        if t >= start:
            phis.append(torch.cat([r_bar, v_prev], 1)[0])
            bars.append(x_bar.clone())
        v_prev = v
    model.reset_state(1)
    phi = torch.stack(phis)                                                  # φ(t)
    bar = torch.stack(bars)                                                  # x̄(t)
    return phi[:-1], bar[1:]


@torch.no_grad()
def collect_dyn_baseline(model: V2Model, x: torch.Tensor, y: torch.Tensor, start: int, stop: int,
                         burnin: int) -> DynBase:
    """Donuk model (çapa ağırlıkları W_0, b_0), [start, stop) aralığında (öncesinde burn-in) Day-0 tabanları (KR-047):
    u_0 = ort. Var_k(v̂^(k)) (başlık uyuşmazlığı, D boyunca ortalama), q_0 = ort. (y − v̂)² (okuma hatası).

    x: [n, C] x_norm, y: [n, D] standart hedef. Yalnızca train diliminde çağrılır (KV6)."""
    model.eval()
    ro = model.readout
    begin = max(0, start - burnin)
    model.reset_state(1)
    us, qs = [], []
    for t in range(begin, stop):
        s = model.backbone.step(x[t: t + 1])                                 # [1, N]
        ro.traces = ro.kappa * ro.traces + (1.0 - ro.kappa) * s.unsqueeze(0)   # [K, 1, N]
        vk = torch.einsum("kbn,kdn->bkd", ro.traces, ro.w_anchor) + ro.b_anchor   # [1, K, D]
        if t >= start:
            us.append(vk.var(1, unbiased=False).mean())
            qs.append((y[t] - vk.mean(1)[0]).pow(2).mean())
    model.reset_state(1)
    return DynBase(float(torch.stack(us).mean()), float(torch.stack(qs).mean()))


def dyn_baseline_stream(model: V2Model, x: torch.Tensor, y: torch.Tensor, start: int, stop: int, burnin: int,
                        cer: CerebellumV2, aff: AffectSpaceV2, plast: Plasticity, alpha_sm: float, eps: float,
                        chunk: int = 0) -> DynBase:
    """`collect_dyn_baseline` ile aynı tabanlar, akışla (CUDA Graph parçaları; uzun dilimde hızlı): donuk A ve η = 0'lı,
    kapısız, yeniden ölçeklemesiz dyn yoklayıcı (W = W_0, g = 1, donuk beyincik). x: [n, C], y: [n, D] → DynBase.

    Kopya oranı tabanı (§12.20): r_0 = ort. e_cer² / ort. (x̄(t) − x̄(t−1))², d_0 = ort. (x̄(t) − x̄(t−1))²."""
    probe = Condition("U", cer_eta=0.0,
                      dyn=DynSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False, False, False, False))
    res = stream_adapt_v5(model, x, y, start, stop, burnin, [Condition("A"), probe], cer, aff, plast, alpha_sm, 0.0, eps,
                          False, chunk=chunk, dyn_base=DynBase(1.0, 1.0))
    d0 = float(res.extra["d1"][1].double().mean())
    return DynBase(float(res.extra["u"][1].double().mean()), float((y[start:stop] - res.pred[0]).double().pow(2).mean()),
                   float(res.extra["e2"][1].double().mean()) / d0, d0)


@torch.no_grad()
def collect_streams(model: V2Model, x: torch.Tensor, a: int, b: int, n_streams: int, burnin: int, alpha_sm: float,
                    proj: torch.Tensor | None = None, keep_r: bool = False
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Donuk model, [a, b) aralığı `n_streams` ardışık akışla (her biri kendi burn-in'iyle); kapılanmamış izler (g = 1).

    → x̄(t) [L, C] (x(t) dahil), v̂(t−1) [L, D], r̄(t)·projᵀ [L, K] (proj: [K, N]), istenirse r̄(t) [L, N].
    φ(t) = [r̄(t), v̂(t−1)] (`collect_cerebellum_data` ile aynı); akışlar uç uca eklenir."""
    model.eval()
    L = b - a
    seg = -(-L // n_streams)
    starts = [a + i * seg for i in range(n_streams) if a + i * seg < b]
    lens = [min(seg, b - s) for s in starts]
    begin = [max(0, s - burnin) for s in starts]
    pre = [s - g for s, g in zip(starts, begin)]
    B, T = len(starts), max(p + n for p, n in zip(pre, lens))
    C, N, dev = x.shape[1], model.backbone.N, x.device
    D = model.readout.weight.shape[1]
    xb = torch.zeros(B, T, C, device=dev)
    for i in range(B):
        xb[i, : pre[i] + lens[i]] = x[begin[i]: starts[i] + lens[i]]
    model.reset_state(B)
    xbar = torch.zeros(B, C, device=dev)
    vprev = torch.zeros(B, D, device=dev)
    o_bar = torch.empty(B, T, C, device=dev)
    o_vp = torch.empty(B, T, D, device=dev)
    o_p = torch.empty(B, T, proj.shape[0], device=dev) if proj is not None else None
    o_r = torch.empty(B, T, N, device=dev) if keep_r else None
    for t in range(T):
        xt = xb[:, t]
        xbar = alpha_sm * xbar + (1.0 - alpha_sm) * xt
        s = model.backbone.step(xt)                                           # [B, N]
        v, rb = model.readout.step(s)                                         # [B, D], [B, N]
        o_bar[:, t] = xbar
        o_vp[:, t] = vprev
        if o_p is not None:
            o_p[:, t] = rb @ proj.T
        if o_r is not None:
            o_r[:, t] = rb
        vprev = v
    model.reset_state(1)

    def cut(o: torch.Tensor | None) -> torch.Tensor | None:
        return None if o is None else torch.cat([o[i, pre[i]: pre[i] + lens[i]] for i in range(B)])

    return cut(o_bar), cut(o_vp), cut(o_p), cut(o_r)                         # type: ignore[return-value]


@torch.no_grad()
def monitor_residual_blocks(model: V2Model, x: torch.Tensor, w_b: torch.Tensor, a: int, b: int, block: int,
                            n_streams: int, burnin: int, alpha_sm: float, piece: int) -> torch.Tensor:
    """İzleme yolu artığı x̄(t) − W_b·φ(t−1)'in [a, b) aralığındaki blok ortalamaları → [nb, C] (BOCPD cer_mon referansı).

    w_b: [C, N + D]. Bellek için `piece` adımlık parçalarla (blok katı); parça sınırında bir adımlık önce-akış alınır."""
    N = model.backbone.N
    piece = max(block, piece // block * block)
    out = []
    for p0 in range(a, b, piece):
        p1 = min(b, p0 + piece)
        bar, vp, P, _ = collect_streams(model, x, p0 - 1, p1, n_streams, burnin, alpha_sm, proj=w_b[:, :N])
        pred = P + vp @ w_b[:, N:].T                                          # W_b·φ(t)
        res = bar[1:] - pred[:-1]                                             # [p1 − p0, C]
        nb = res.shape[0] // block
        out.append(res[: nb * block].reshape(nb, block, -1).mean(1))
    return torch.cat(out)
