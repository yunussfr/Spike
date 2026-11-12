"""Faz V7 (KR-049): değişim noktası tespitiyle kapılanan girdi uyumu + sönümlü ReFIT (ARCHITECTURE.md §12.9, §12.15–19).

Protokoller:
  test  — `primates` (NeuroBench oturumları); senaryolar crossday / within / electrode (V6 ile aynı tanım).
  tune  — `tune_primates` (test dışı oturumlar; Day-0 modeli aynı): crossday (Day-0 → ayar oturumları) ve electrode
          (ayar oturumu + değerlendirme diliminin ortasında kanal kaybı, Day-0 modeli). Test oturumlarına dokunmaz.
  ext   — (V8, KR-051) `ext_primates` (EK test oturumları); tune ile aynı senaryolar, ayrı tablo.
Omurga kolu (V8, §12.9): `--ckpt-stage day0` (S, yalnız Day-0) | `multiday` (M, ÖN + Day-0); satırlarda `backbone`.
`--electrode-crossday`: test protokolünde elektrot kaybı da günler arası (Day-0 / multiday) modelle koşar (V8).
Geçişler (aynı akış, aynı model):
  1. ReFIT: A, R5, G7 (Degenhart akış başından), D7 (BOCPD + ölü kanal + Degenhart + dinamik duygu), D7 ablasyonları
  2. ReFIT, eşlenmiş: B7 (D7'nin η̄, ᾱ'sı), P7 (D7'nin girdi tarafı + η̄, ᾱ), C7 (D7'nin η, α, c dizileri karıştırılmış)
  3. gecikmeli hız etiketi (KR-041): A, D7_delay
  4. (--runs_v6 verilirse) V6 checkpoint'iyle, ReFIT: A, D7@v6 (augmentation'ın katkısı)
  5. BOCPD öteki kaynaklarla (`v7.alt_sources`, §12.16): A, D7@<kaynak>, D7_split@<kaynak>
Geçiş 1'de ayrıca beyincik ailesi (`v7.cer_family`, §12.20, KR-050): D7_split (tonik / fazik iş bölümü), D7_ph0 (fazik
kol kapalı), D7_N0 (fazik kol eski normalizasyonla).
Tüm tabanlar (beyincik, BOCPD skor referansı / τ_T / W_b, ölü kanal p, Day-0 FA, u_0, q_0, r_0) yalnızca referans
modelin train diliminden (KV6).

Kullanım:
    uv run python experiments/v2_v7.py --config configs/v2_v7.yaml --runs runs/<v7 ön-eğitim indy> runs/<… loco> \\
        --protocol tune --split val            # ayar: ayar seed'leri
    uv run python experiments/v2_v7.py --config configs/v2_v7.yaml --runs ... --runs_v6 runs/<v3 indy> runs/<v3 loco> \\
        --protocol test --split test           # rapor: rapor seed'leri
    uv run python experiments/v2_v7.py --config configs/v2_v8.yaml --runs runs/<v8 ön-eğitim indy> --primates indy \\
        --ckpt-stage multiday --protocol tune --split val    # V8: kol M (kol S için --ckpt-stage day0)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.v2_online import find_ckpt  # noqa: E402
from experiments.v2_v6 import Ctx, Ref, drop_channels, recovery, summary  # noqa: E402
from src.data.primate import load_task_state  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.affect import AffectSpaceV2  # noqa: E402
from src.v2.align import Aligner, AlignSpec, bin_means, fa_fit  # noqa: E402
from src.v2.cerebellum import CerebellumV2  # noqa: E402
from src.v2.changepoint import BOCPD, BocpdSpec, ScoreRef, block_means, calibrate_temper, fit_score_ref  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.dead_channel import DeadChannelDetector, spike_prob  # noqa: E402
from src.v2.intent import Intent, label_quality, make_intent, refit_label  # noqa: E402
from src.v2.metrics import r2_neurobench  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.online_v5 import (Condition, V5Result, V7Inputs, collect_streams, monitor_residual_blocks,  # noqa: E402
                              stream_adapt_v5)
from src.v2.stream_spec import DynBase  # noqa: E402
from src.v2.train import Prepared, prepare_session  # noqa: E402

TRACE_KEYS = ("a_bocpd", "n_dead", "a_int", "val", "dom", "alpha")


@dataclass
class Ref7:
    """V6 tabanları + V7 bileşenlerinin Day-0 (ya da oturum içi) train referansları."""
    base: Ref
    bocpd: dict[str, tuple[ScoreRef, float, torch.Tensor | None]]   # kaynak → (skor referansı, τ_T, W_b)
    calib: dict[str, list]                            # kaynak → [(τ_T, yanlış alarm oranı)]
    z0: torch.Tensor                                  # [C]
    p_spike: torch.Tensor                             # [C]
    fa0: Any
    q0_refit: float


class Ctx7(Ctx):
    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        super().__init__(cfg, split)
        self.v7, self.bc = cfg["v7"], cfg["bocpd"]
        self.aspec = AlignSpec.from_config(cfg)

    def make_ref7(self, m: V2Model, ref: Prepared) -> Ref7:
        cfg, sl = self.cfg, ref.session.split("train")
        base = self.make_ref(m, ref)
        xt = ref.x[sl]
        bocpd, calib = {}, {}
        for src in self.sources():
            r_, t_, w_, c_ = self.bocpd_ref(m, ref, src)
            bocpd[src], calib[src] = (r_, t_, w_), c_
        z0 = ref.z0
        p = spike_prob(xt, z0, cfg["dead"]["tol"])
        fa0 = fa_fit(bin_means(xt, self.aspec.fa_bin), self.aspec.q, self.aspec.em_iters, self.aspec.psi_floor)
        # q_0 (ReFIT): donuk modelin baz dilimindeki niyet hatası (V6'daki q_0 ile aynı dilim, etiket kaynağı farklı)
        bs = self.v6["baseline_split"]
        h = cfg["cerebellum"]["holdout_steps"]
        bsl = slice(sl.stop - h, sl.stop) if bs == "train" else ref.session.split(bs)
        it = self.intent(ref)
        res = self.stream(m, ref, [Condition("A")], base, None, bsl, it)
        dd, mi = it.dd[bsl], it.m[bsl]
        err = (refit_label(res.pred[0], dd, it.mu, it.sd) - res.pred[0]) * mi.unsqueeze(1)
        q0 = float(err.pow(2).mean(1).sum() / mi.sum().clamp(min=1.0))
        return Ref7(base, bocpd, calib, z0, p, fa0, q0)

    def sources(self) -> list[str]:
        return list(dict.fromkeys([self.bc["source"], *self.v7["alt_sources"]]))

    def bocpd_ref(self, m: V2Model, ref: Prepared, src: str) -> tuple[ScoreRef, float, torch.Tensor | None, list]:
        """BOCPD skor referansı (§12.16, KR-050), yalnız Day-0 train: m, s (ρ), m_y, s²_y ilk %80'den; τ_T son %20'de.
        cer_mon: W_b train başındaki [başlangıç + burn-in, + fit_steps) dilimde ridge; artık bu dilimden sonra."""
        cfg, bc, sl = self.cfg, self.bc, ref.session.split("train")
        spec0 = BocpdSpec.from_config(cfg, source=src)
        blk = spec0.block
        n80 = sl.start + int(round((1.0 - bc["calib_frac"]) * (sl.stop - sl.start)))
        w_b = None
        if src == "cer_mon":
            cc = cfg["cerebellum"]
            f0 = sl.start + self.burnin
            f1 = f0 + cc["fit_steps"]
            if f1 >= n80:
                raise ValueError("cer_mon: Day-0 train fit dilimi + kalibrasyon payı için çok kısa")
            bar, vp, _, r = collect_streams(m, ref.x, f0, f1, int(bc["mon_fit_streams"]), self.burnin, self.alpha_sm,
                                            keep_r=True)
            w_b, _ = CerebellumV2.fit(torch.cat([r, vp], 1)[:-1], bar[1:], cc["ridge"])
            del bar, vp, r
            kw = dict(block=blk, n_streams=int(bc["mon_streams"]), burnin=self.burnin, alpha_sm=self.alpha_sm,
                      piece=int(bc["mon_piece_steps"]))
            o_fit = monitor_residual_blocks(m, ref.x, w_b, f1, n80, **kw)
            o_hold = monitor_residual_blocks(m, ref.x, w_b, n80, sl.stop, **kw)
        else:
            o_fit = block_means(ref.x[sl.start: n80], blk)
            o_hold = block_means(ref.x[n80: sl.stop], blk)
        sref = fit_score_ref(o_fit, src, float(bc["score_eps"]), spec0.var_floor)
        temper, calib = calibrate_temper(o_hold, sref, cfg, src)
        return sref, temper, w_b, calib

    def intent(self, pr: Prepared) -> Intent:
        pos, tgt = load_task_state(self.cfg, pr.session.name)
        if len(pos) != len(pr.x):
            raise ValueError(f"{pr.session.name}: görev durumu ({len(pos)}) ve girdi ({len(pr.x)}) uzunluğu farklı")
        return make_intent(self.cfg, pos, tgt, pr.y_mu, pr.y_sd, pr.x.device)

    def v7_inputs(self, r7: Ref7, it: Intent | None, dev: torch.device, source: str | None = None) -> V7Inputs:
        src = source or self.bc["source"]
        sref, temper, w_b = r7.bocpd[src]
        return V7Inputs(bocpd=BOCPD(sref.to(dev), BocpdSpec.from_config(self.cfg, temper=temper, source=src)), w_b=w_b,
                        dead=DeadChannelDetector(r7.z0, r7.p_spike, self.cfg["dead"]["alpha"], self.cfg["dead"]["tol"]),
                        aligners={1: Aligner(r7.fa0, self.aspec, True, dev), 2: Aligner(r7.fa0, self.aspec, False, dev)},
                        intent=it, align_check=max(1, self.kw["chunk"]))

    def stream(self, m: V2Model, pr: Prepared, conds: list[Condition], ref: Ref, v7: V7Inputs | None, sl: slice,
               it: Intent | None = None, delay: int = 0, dyn_base: DynBase | None = None) -> V5Result:
        v7 = v7 if v7 is not None else V7Inputs(intent=it)
        return stream_adapt_v5(m, pr.x, pr.y, sl.start, sl.stop, self.burnin, conds,
                               CerebellumV2(ref.w0, ref.e2, self.cfg["cerebellum"]), AffectSpaceV2(self.af), self.plast,
                               self.alpha_sm, self.kw["anchor_decay"], self.kw["eps"], self.cfg["v5"]["homeostasis"],
                               chunk=self.kw["chunk"], label_delay=delay, controls=ref.ctrl, drift=ref.drift,
                               dyn_base=dyn_base or ref.base, v7=v7)

    # --- koşullar ---------------------------------------------------------------------------------------------------
    def bocpd_kw(self) -> dict[str, Any]:
        return dict(sensor="bocpd", gate_lo=float(self.bc["gate_lo"]), gate_hi=float(self.bc["gate_hi"]))

    def d7(self, name: str, over: dict[str, Any] | None = None, **kw: Any) -> Condition:
        return self.dyn_cond(name, self.v7["base_variant"], over=over,
                             **{**self.bocpd_kw(), "dead": True, "align": 1, **kw})

    def fam(self, name: str) -> dict[str, Any]:
        """Beyincik ailesi üyesinin dyn ayarları (§12.20): v7.cer_family[name] + fazik kol (plastik beyincik η_cer, w_u)."""
        ph = self.v7["phasic"]
        return {**self.v7["cer_family"][name], "cer_eta": float(ph["cer_eta"]), "w_u": float(ph["w_u"])}

    def pass1(self) -> list[Condition]:      # type: ignore[override]
        e = self.eta0
        out = [Condition("A"), Condition("R5", eta0=e, renorm=True, lam_calm=self.lam(5.0)),
               Condition("G7", eta0=e, align=2), self.d7("D7")]
        abl = {"D7_nodead": dict(dead=False), "D7_noalign": dict(align=0)}
        return (out + [self.d7(a, **abl[a]) for a in self.v7["ablations"]]
                + [self.d7(n, over=self.fam(n)) for n in self.v7["cer_family"]])

    def pass_sources(self) -> list[tuple[str, list[Condition]]]:
        """Geçiş 5: BOCPD öteki kaynaklarla; A, D7@kaynak, D7_split@kaynak (D7_split ailede varsa)."""
        out = []
        for src in self.sources()[1:]:
            cs = [Condition("A"), self.d7(f"D7@{src}")]
            if "D7_split" in self.v7["cer_family"]:
                cs.append(self.d7(f"D7_split@{src}", over=self.fam("D7_split")))
            out.append((src, cs))
        return out

    def pass2(self, conds: list[Condition], res: V5Result, seed: int) -> list[Condition]:   # type: ignore[override]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        i = [c.name for c in conds].index("D7")
        eta, alpha, cons = res.eta[i], res.extra["alpha"][i], res.extra["cons"][i]
        e_bar, a_bar = float(eta.mean()), float(alpha.mean())
        perm = torch.randperm(eta.shape[0], generator=g).to(eta.device)
        p7 = {**self.sensor_kw(), **self.bocpd_kw(), "dead": True, "align": 1, "alpha_anchor": a_bar}
        return [Condition("A"), Condition("B7", eta0=e_bar, alpha_anchor=a_bar),
                Condition("P7", eta0=e_bar, cer_raw=True, cer_eta=0.0, cer_src=0, **p7),
                self.d7("C7", eta_seq=eta[perm].clone(), alpha_seq=alpha[perm].clone(), cons_seq=cons[perm].clone())]


def first_time(a: torch.Tensor, th: float, dt: float, at: int = 0) -> float:
    """a: [L] → a ≥ th olduğu ilk an (s, `at`'tan sonra); hiç yoksa −1."""
    idx = (a[at:] >= th).nonzero()
    return float(idx[0, 0]) * dt / 1000.0 if len(idx) else -1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="V7 (augmentation'lı) checkpoint koşuları")
    p.add_argument("--runs_v6", nargs="*", default=[], help="V6 checkpoint koşuları (D7@v6 geçişi)")
    p.add_argument("--protocol", choices=["test", "tune", "ext"], default="tune")
    p.add_argument("--ckpt-stage", choices=["day0", "multiday"], default="day0", help="günler arası model (V8 kolu S / M)")
    p.add_argument("--electrode-crossday", action="store_true", help="test protokolünde elektrot kaybı günler arası modelle")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scenarios", nargs="*", default=["crossday", "within", "electrode"])
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    if a.protocol == "test" and a.split == "test" and a.seeds is None:
        seeds = cfg["experiment"]["seeds"]
    else:
        seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds"]
    stage = a.ckpt_stage
    backbone = {"day0": "S", "multiday": "M"}[stage]
    cfg["experiment"]["name"] = (f"{cfg['experiment']['name']}_{a.protocol}_{a.split}"
                                 + ("" if stage == "day0" else f"_{stage}"))     # V7: v2_v7_<protokol>_<bölme>
    device = resolve_device(cfg["experiment"]["device"])
    runs, runs6 = [Path(r) for r in a.runs], [Path(r) for r in a.runs_v6]
    crossday_proto = a.protocol in ("tune", "ext")                           # Day-k'ler test dışı ya da EK
    if crossday_proto:
        a.scenarios = [s for s in a.scenarios if s != "within"]              # ayar / EK oturumlarının ince ayar modeli yok
    el_cross = crossday_proto or a.electrode_crossday
    logger = RunLogger(cfg, seeds, a.config)
    logger.save_json("inputs.json", {"runs": [str(r) for r in runs], "runs_v6": [str(r) for r in runs6],
                                     "split": a.split, "protocol": a.protocol, "scenarios": a.scenarios,
                                     "ckpt_stage": stage, "backbone": backbone, "electrode_crossday": el_cross})
    ctx = Ctx7(cfg, a.split)
    dt, v6 = ctx.dt, ctx.v6
    rc, el = v6["recovery"], v6["electrode"]
    w_rec, s_rec = ms_to_steps(1000.0 * rc["window_s"], dt), ms_to_steps(1000.0 * rc["step_s"], dt)
    d_delay = ms_to_steps(ctx.v7["delay_ms"], dt)
    every = int(v6["trace_every"])
    rows: list[dict[str, Any]] = []
    refs_log: list[dict[str, Any]] = []

    def record(res: V5Result, conds: list[Condition], pr: Prepared, it: Intent, meta: dict[str, Any], t0: float,
               t_drop: int | None, store: dict[str, Any], label: str, ck: str, src: str | None = None) -> None:
        sl = pr.session.split(a.split)
        y = (pr.y[sl] * pr.y_sd + pr.y_mu).cpu()
        h = res.rate.shape[1] // 2
        lq = label_quality(y, it.dd[sl].cpu(), it.m[sl].cpu())
        for e, c in enumerate(conds):
            if c.name == "A" and any(r["cond"] == "A" and r["session"] == pr.session.name and r["seed"] == meta["seed"]
                                     and r["scenario"] == meta["scenario"] for r in rows):
                continue
            pred = (res.pred[e] * pr.y_sd + pr.y_mu).cpu()
            ex = res.extra
            ab = ex["a_bocpd"][e].cpu()
            row = {**meta, "backbone": backbone, "session": pr.session.name, "cond": c.name, "r2": r2_neurobench(pred, y), "label": label,
                   "ckpt": ck, "bocpd_source": src or ctx.bc["source"],
                   "a_int_pos_mean": float(ex["a_int"][e].clamp(min=0.0).mean()),
                   "dw_rel": float(res.dw_rel[e]), "anc_rel": float(ex["anc_rel"][e]),
                   "rate_fast": float(res.rate[e, :h].mean()), "rate_slow": float(res.rate[e, h:].mean()),
                   "eta_mean": float(res.eta[e].mean()), "alpha_mean": float(ex["alpha"][e].mean()),
                   "lam_mean": float(res.lam[e].mean()), "a_bocpd_mean": float(ab.mean()),
                   "t_detect_s": first_time(ab, 0.5, dt), "n_dead_end": float(ex["n_dead"][e, -1]),
                   "align_fits": json.dumps(ex["align_fits"]), "label_quality_r2": r2_neurobench(lq, y),
                   "dyn": c.dyn is not None}
            if t_drop is not None:
                rec_s, cens, ref_r2, st, rr = recovery(pred, y, t_drop, w_rec, s_rec, rc["tol"], dt)
                row.update(r2_pre=r2_neurobench(pred[:t_drop], y[:t_drop]), r2_post=r2_neurobench(pred[t_drop:], y[t_drop:]),
                           rec_s=rec_s, rec_censored=cens, r2_win_ref=ref_r2,
                           t_detect_drop_s=first_time(ab, 0.5, dt, t_drop),
                           t_dead_s=first_time(ex["n_dead"][e].cpu(), 1.0, dt, t_drop))
                store.setdefault("rolling", {})[c.name] = (st, rr)
            row["sec"] = time.time() - t0
            rows.append(row)
            logger.log(len(rows), **row)
            store.setdefault("trace", {})[c.name] = {"a": res.a[e, ::every].cpu(), "eta": res.eta[e, ::every].cpu(),
                                                     "lam": res.lam[e, ::every].cpu(),
                                                     **{k: ex[k][e, ::every].cpu() for k in TRACE_KEYS}}
        done = [r for r in rows if r["session"] == pr.session.name and r["seed"] == meta["seed"]
                and r["scenario"] == meta["scenario"]]
        print(json.dumps({**meta, "session": pr.session.name, "sec": round(time.time() - t0),
                          "r2": {r["cond"]: round(r["r2"], 3) for r in done}}, ensure_ascii=False), flush=True)

    def evaluate(m: V2Model, m6: V2Model | None, pr: Prepared, r7: Ref7, r76: Ref7 | None, t_drop: int | None,
                 **meta: Any) -> None:
        sl = pr.session.split(a.split)
        it = ctx.intent(pr)
        b0 = r7.base.base
        dyn7 = DynBase(b0.u0, r7.q0_refit, b0.r0, b0.d0)
        store: dict[str, Any] = {"t_drop": t_drop}
        t0 = time.time()
        c1 = ctx.pass1()
        res = ctx.stream(m, pr, c1, r7.base, ctx.v7_inputs(r7, it, pr.x.device), sl, dyn_base=dyn7)
        record(res, c1, pr, it, meta, t0, t_drop, store, "refit", "v7")
        c2 = ctx.pass2(c1, res, meta["seed"])
        del res
        t0 = time.time()
        record(ctx.stream(m, pr, c2, r7.base, ctx.v7_inputs(r7, it, pr.x.device), sl, dyn_base=dyn7), c2, pr, it, meta,
               t0, t_drop, store, "refit", "v7")
        t0 = time.time()                                                     # gecikmeli hız etiketi (KR-041)
        c3 = [Condition("A"), ctx.d7("D7_delay")]
        v7d = ctx.v7_inputs(r7, None, pr.x.device)
        record(ctx.stream(m, pr, c3, r7.base, v7d, sl, delay=d_delay), c3, pr, it, meta, t0, t_drop, store, "delayed", "v7")
        if m6 is not None and r76 is not None:                               # augmentation'sız V6 checkpoint'i
            t0 = time.time()
            c4 = [Condition("A6"), ctx.d7("D7@v6")]
            res6 = ctx.stream(m6, pr, c4, r76.base, ctx.v7_inputs(r76, it, pr.x.device), sl,
                              dyn_base=DynBase(r76.base.base.u0, r76.q0_refit, r76.base.base.r0, r76.base.base.d0))
            record(res6, c4, pr, it, meta, t0, t_drop, store, "refit", "v6")
        for src, c5 in ctx.pass_sources():                                   # BOCPD öteki kaynaklarla (§12.16)
            t0 = time.time()
            record(ctx.stream(m, pr, c5, r7.base, ctx.v7_inputs(r7, it, pr.x.device, src), sl, dyn_base=dyn7), c5, pr,
                   it, meta, t0, t_drop, store, "refit", "v7", src)
        torch.save(store, logger.path(f"trace_{meta['scenario']}_{pr.session.name}_seed{meta['seed']}.pt"))

    def load(rs: list[Path], name: str, C: int, D: int) -> V2Model | None:
        ck = find_ckpt(rs, name) if rs else None
        if ck is None:
            return None
        m = V2Model(cfg, C, D).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        return m

    def log_ref(r7: Ref7, **meta: Any) -> None:
        refs_log.append({**meta, "temper": {k: v[1] for k, v in r7.bocpd.items()}, "calib": r7.calib, "q0_refit": r7.q0_refit,
                         "n_never_spike": int((r7.p_spike == 0).sum())})
        logger.save_json("refs.json", refs_log)

    for primate in a.primates or list(cfg["primates"]):
        day0, dayk = crossday_sessions(cfg, primate, protocol=a.protocol)
        names = [day0] + dayk
        sessions = {n: load_v2_session(cfg, n) for n in names}
        C, D = sessions[day0].D_in, sessions[day0].y.shape[1]
        for seed in seeds:
            m = load(runs, f"{stage}_{primate}_seed{seed}.pt", C, D)
            m6 = load(runs6, f"day0_{primate}_seed{seed}.pt", C, D)
            if m is not None and ({"crossday", "electrode"} & set(a.scenarios)):
                pr0 = prepare_session(cfg, sessions[day0], device)
                r7 = ctx.make_ref7(m, pr0)
                r76 = ctx.make_ref7(m6, pr0) if m6 is not None else None
                log_ref(r7, primate=primate, seed=seed, model=stage)
                for n in names:
                    pr = pr0 if n == day0 else prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                    if "crossday" in a.scenarios and not (crossday_proto and n == day0):
                        evaluate(m, m6, pr, r7, r76, None, primate=primate, seed=seed, scenario="crossday",
                                 day=names.index(n), protocol=a.protocol)
                    if "electrode" in a.scenarios and el_cross and n != day0:
                        sl = sessions[n].split(a.split)
                        at = sl.start + int(el["at_frac"] * (sl.stop - sl.start))
                        s_e, ch = drop_channels(sessions[n], at, el["n_channels"], 1000 * seed + names.index(n))
                        pr_e = prepare_session(cfg, s_e, device, stats_from=sessions[day0])
                        evaluate(m, m6, pr_e, r7, r76, at - sl.start, primate=primate, seed=seed, scenario="electrode",
                                 day=names.index(n), protocol=a.protocol, channels=ch)
                        del pr_e
                    if pr is not pr0:
                        del pr
                del pr0
            del m, m6
            if a.protocol == "test" and ({"within", "electrode"} & set(a.scenarios)):
                for n in names:
                    m = load(runs, f"finetune_{n}_seed{seed}.pt", C, D)
                    if m is None:
                        continue
                    m6 = load(runs6, f"finetune_{n}_seed{seed}.pt", C, D)
                    pr = prepare_session(cfg, sessions[n], device)
                    r7 = ctx.make_ref7(m, pr)
                    r76 = ctx.make_ref7(m6, pr) if m6 is not None else None
                    log_ref(r7, primate=primate, seed=seed, model=f"finetune_{n}")
                    if "within" in a.scenarios:
                        evaluate(m, m6, pr, r7, r76, None, primate=primate, seed=seed, scenario="within",
                                 day=names.index(n), protocol=a.protocol)
                    if "electrode" in a.scenarios and not el_cross:
                        sl = sessions[n].split(a.split)
                        at = sl.start + int(el["at_frac"] * (sl.stop - sl.start))
                        s_e, ch = drop_channels(sessions[n], at, el["n_channels"], 1000 * seed + names.index(n))
                        pr_e = prepare_session(cfg, s_e, device)
                        evaluate(m, m6, pr_e, r7, r76, at - sl.start, primate=primate, seed=seed, scenario="electrode",
                                 day=names.index(n), protocol=a.protocol, channels=ch)
                        del pr_e
                    del pr, m, m6
            logger.save_json("results.json", rows)
            torch.cuda.empty_cache() if device.type == "cuda" else None

    summary(rows, logger, a.split, seeds)


if __name__ == "__main__":
    main()
