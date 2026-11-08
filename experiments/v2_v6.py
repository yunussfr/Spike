"""Faz V5 kapanışı (val ayarı) ve Faz V6 (asıl deney): dinamik duygu (KR-047) ve kontrolleri.

Senaryolar:
  crossday  — `day0_<primat>_seed<s>.pt`; tüm oturumlar Day-0 istatistikleriyle; tabanlar (beyincik, sapma, u_0, q_0) Day-0 train
  within    — `finetune_<oturum>_seed<s>.pt`; tabanlar oturumun kendi train diliminden
  electrode — within ile aynı model; değerlendirme diliminin `at_frac` noktasında seed'e bağlı n kanalın ham sayımı 0 (v2-E)
Koşullar (geçiş 1, tek akışta): v6.fixed (A, B, R5, Pm, Pm3, O) + v6.variants (dinamik duygu D ve ablasyonları).
Geçiş 2 (v6.matched'teki her varyant v için, aynı akış): Bm_v (v2-B: η̄, ᾱ eşlenmiş), P_v (v2-P: Pm'in sensörlü yeniden
ölçeklemesi + η̄, ᾱ), C_v (v2-C: v'nin η(t), α(t), c(t) dizileri zamanda karıştırılmış; yeniden ölçekleme D ile aynı sensörden).
Ayar yalnızca `--split val` + ayar seed'leri (varsayılan); test: `--split test` (varsayılan rapor seed'leri).

Kullanım:
    uv run python experiments/v2_v6.py --config configs/v2_v6.yaml --runs runs/<v3 indy> runs/<v3 loco> --split val
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import r2_neurobench  # noqa: E402
from experiments.v2_online import find_ckpt  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.affect import AffectSpaceV2  # noqa: E402
from src.v2.cerebellum import CerebellumV2  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.online import Controls, stream_kwargs  # noqa: E402
from src.v2.online_v5 import (Condition, Drift, DynBase, DynSpec, Plasticity, V5Result,  # noqa: E402
                              collect_cerebellum_data, drift_baseline, dyn_baseline_stream, stream_adapt_v5)
from src.v2.train import Prepared, prepare_session  # noqa: E402

TRACE_KEYS = ("a_int", "val", "dom", "alpha", "cons", "brake", "u", "s_sens", "s_drift", "s_cer")


@dataclass
class Ref:
    """Bir modelin Day-0 (ya da oturum içi) train tabanları; yalnızca train girdisi / etiketi (KV6)."""
    w0: torch.Tensor                      # [C, P] beyincik ridge
    e2: float                             # beyincik Day-0 hata tabanı
    ctrl: Controls                        # yeniden ölçekleme referansı (m_0, v_0)
    drift: Drift                          # sapma ölçüsü ve tolerans zarfı δ_0
    base: DynBase                         # u_0, q_0


class Ctx:
    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        self.cfg, self.split = cfg, split
        self.dt = cfg["time"]["dt_ms"]
        self.burnin = ms_to_steps(cfg["online"]["burnin_ms"], self.dt)
        self.alpha_sm = math.exp(-self.dt / cfg["encoder"]["smooth_tau_ms"])
        self.plast = Plasticity.from_config(cfg)
        self.kw = stream_kwargs(cfg)
        af = dict(cfg["affect"])
        for k in ("fast", "slow"):
            if f"tau_{k}_ms" in af:
                af[f"a_{k}"] = 1.0 - math.exp(-self.dt / af[f"tau_{k}_ms"])
        self.af, self.eta0 = af, af["eta_0"]
        self.pc, self.v6 = cfg["persist"], cfg["v6"]

    def lam(self, tau_s: float) -> float:
        """τ (s) → adım başına EMA katsayısı; τ = ∞ → 0."""
        return 1.0 - math.exp(-self.dt / (1000.0 * tau_s))

    # --- tabanlar ---------------------------------------------------------------------------------------------------
    def make_ref(self, m: V2Model, ref: Prepared) -> Ref:
        cfg, sl = self.cfg, ref.session.split("train")
        n, h = cfg["cerebellum"]["fit_steps"], cfg["cerebellum"]["holdout_steps"]
        phi, tgt = collect_cerebellum_data(m, ref.x, max(sl.start, sl.stop - n - h), sl.stop, self.burnin, self.alpha_sm)
        w, _ = CerebellumV2.fit(phi[:-h], tgt[:-h], cfg["cerebellum"]["ridge"])
        e2 = float(((tgt[-h:] - phi[-h:] @ w.T) ** 2).mean())
        xt = ref.x[sl]
        ctrl = Controls(renorm_m0=xt.mean(0), renorm_v0=xt.var(0), renorm_eps=cfg["renorm"]["eps"])
        spec = Drift(lam=self.lam(self.pc["drift"]["tau_s"]), var_floor=self.pc["drift"]["var_floor"])
        spec.delta0 = drift_baseline(ref.x, sl.start, sl.stop, ctrl.renorm_m0, ctrl.renorm_v0, spec,
                                     stat=self.pc["drift"]["baseline"])
        bs = self.v6["baseline_split"]                                       # u_0, q_0: train sonu ya da Day-0 val
        bsl = slice(sl.stop - h, sl.stop) if bs == "train" else ref.session.split(bs)
        base = dyn_baseline_stream(m, ref.x, ref.y, bsl.start, bsl.stop, self.burnin,
                                   CerebellumV2(w, e2, cfg["cerebellum"]), AffectSpaceV2(self.af), self.plast,
                                   self.alpha_sm, self.kw["eps"], chunk=self.kw["chunk"])
        return Ref(w, e2, ctrl, spec, base)

    # --- koşullar ---------------------------------------------------------------------------------------------------
    def sensor_kw(self) -> dict[str, Any]:
        """Kalıcı sürpriz sensörü (KR-046 Pm): izleme yolu + girdi sapması (max), sabit taban, eşikli kapı, dönüş."""
        pc = self.pc
        return dict(renorm=True, lam_calm=self.lam(pc["tau_calm_s"]), lam_surp=self.lam(pc["tau_surp_s"]), affect=True,
                    fixed_slow=True, a_fast=self.lam(pc["tau_fast_s"]), drift_max=True, gate_lo=pc["gate_lo"],
                    gate_hi=pc["gate_hi"], lam_return=self.lam(pc["tau_return_s"]))

    def persist_cond(self, name: str, eta0: float, **kw: Any) -> Condition:
        """Pm: sensör kendi koşul durumunda (izleme yolu cer_src = 0, ham hedef, donuk W_cer); okuma sabit η."""
        return Condition(name, eta0=eta0, cer_raw=True, cer_eta=0.0, cer_src=0, **self.sensor_kw(), **kw)

    def dyn_params(self, name: str) -> dict[str, Any]:
        return {**self.v6["dyn"], **self.v6["variants"][name]}

    def dyn_cond(self, name: str, variant: str, **kw: Any) -> Condition:
        p = self.dyn_params(variant)
        spec = DynSpec(w_u=float(p["w_u"]), lam_a=self.lam(p["tau_arousal_s"]), lam_f=self.lam(p["tau_vfast_s"]),
                       lam_m=self.lam(p["tau_vmid_s"]), lam_d=self.lam(p["tau_dom_s"]), rho_v=float(p["rho_v"]),
                       alpha_b=self.lam(p["tau_brake_s"]), lam_c=self.lam(p["tau_cons_s"]), delta_m=float(p["delta_m"]),
                       gas=bool(p.get("gas", True)), brake=bool(p.get("brake", True)),
                       loosen=bool(p.get("loosen", True)), consolidate=bool(p.get("consolidate", True)),
                       a_mode={"internal": 0, "sensor": 1, "max": 2}[p["arousal"]], gas_dom=bool(p["gas_dom"]))
        base = dict(eta0=self.eta0 * float(p["eta_scale"]), gate=bool(p.get("gate", True)),
                    alpha_eta=float(p.get("alpha_eta", self.af["alpha_eta"])), cer_raw=False, cer_eta=float(p["cer_eta"]),
                    cer_src=-1, dyn=spec, **self.sensor_kw())
        return Condition(name, **{**base, **kw})

    def pass1(self, scenario: str, day: int) -> list[Condition]:
        eta0, out = self.eta0, []
        for c in self.v6["fixed"]:
            if c == "A":
                out.append(Condition("A"))
            elif c == "B":
                out.append(Condition("B", eta0=eta0))
            elif c.startswith("R"):
                out.append(Condition(c, eta0=eta0, renorm=True, lam_calm=self.lam(float(c[1:]))))
            elif c == "Pm":
                out.append(self.persist_cond("Pm", eta0))
            elif c == "Pm3":
                out.append(self.persist_cond("Pm3", eta0 * self.v6["pm_eta_scale"]))
            elif c in ("Bg", "Pg"):                                          # tüm akışlar üzerinden eşlenmiş sabit η, α
                gm = self.v6["global_matched"]                               # (ayar koşusundaki D'nin ortalaması)
                out.append(Condition("Bg", eta0=gm["eta"], alpha_anchor=gm["alpha"]) if c == "Bg"
                           else self.persist_cond("Pg", gm["eta"], alpha_anchor=gm["alpha"]))
            elif c == "O":                                                   # gün bilgisiyle üst sınır (yalnız crossday)
                on = scenario == "crossday" and day > 0
                out.append(Condition("O", eta0=eta0, renorm=True, lam_calm=self.lam(self.pc["tau_surp_s"]) if on else 0.0))
            else:
                raise ValueError(f"bilinmeyen sabit koşul: {c}")
        if out[0].name != "A":
            raise ValueError("v6.fixed A ile başlamalı (ortak izleme yolu)")
        return out + [self.dyn_cond(v, v) for v in self.v6["variants"]]

    def pass2(self, conds: list[Condition], res: V5Result, seed: int) -> list[Condition]:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        names, out = [c.name for c in conds], [Condition("A")]
        for v in self.v6["matched"]:
            i = names.index(v)
            eta, alpha, cons = res.eta[i], res.extra["alpha"][i], res.extra["cons"][i]
            e_bar, a_bar = float(eta.mean()), float(alpha.mean())
            perm = torch.randperm(eta.shape[0], generator=g).to(eta.device)
            out += [Condition(f"Bm_{v}", eta0=e_bar, alpha_anchor=a_bar),
                    self.persist_cond(f"P_{v}", e_bar, alpha_anchor=a_bar),
                    self.dyn_cond(f"C_{v}", v, eta_seq=eta[perm].clone(), alpha_seq=alpha[perm].clone(),
                                  cons_seq=cons[perm].clone())]
        return out


# --- metrikler -------------------------------------------------------------------------------------------------------
def rolling_r2(pred: torch.Tensor, y: torch.Tensor, w: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Kayan pencere R² (sütun başına 1 − SSE/SST, ortalama; NeuroBench tanımı). pred, y: [n, D] → başlangıçlar, R²."""
    p, t = pred.double().cpu(), y.double().cpu()
    z = torch.zeros(1, t.shape[1], dtype=torch.float64)
    cs = lambda a: torch.cat([z, a.cumsum(0)])                               # noqa: E731
    s_e, s_y, s_yy = cs((t - p) ** 2), cs(t), cs(t * t)
    st = torch.arange(0, len(t) - w + 1, step)
    sse = s_e[st + w] - s_e[st]
    sy = s_y[st + w] - s_y[st]
    sst = (s_yy[st + w] - s_yy[st]) - sy * sy / w
    return st.numpy(), (1.0 - sse / sst.clamp(min=1e-12)).mean(1).numpy()


def recovery(pred: torch.Tensor, y: torch.Tensor, t_drop: int, w: int, step: int, tol: float,
             dt: float) -> tuple[float, bool, float, np.ndarray, np.ndarray]:
    """Kayıptan sonraki ilk pencerenin (tamamı kayıptan sonra) R²'si ≥ kayıp öncesi pencere medyanı − tol olduğu an.
    → (toparlanma s, sansürlü mü, öncesi medyan, pencere başlangıçları, R²)"""
    st, r = rolling_r2(pred, y, w, step)
    pre = st + w <= t_drop
    ref = float(np.median(r[pre])) if pre.any() else float("nan")
    post = np.nonzero(st >= t_drop)[0]
    ok = post[r[post] >= ref - tol] if len(post) else post
    if len(ok):
        return float((st[ok[0]] + w - t_drop) * dt / 1000.0), False, ref, st, r
    return float((len(y) - t_drop) * dt / 1000.0), True, ref, st, r


def drop_channels(s: Any, at: int, n: int, seed: int) -> tuple[Any, list[int]]:
    """Session kopyası: `at` adımından sonra seed'e bağlı n kanalın ham sayımı 0 (elektrot kaybı)."""
    g = torch.Generator().manual_seed(int(seed))
    ch = torch.randperm(s.x.shape[1], generator=g)[:n].sort().values.tolist()
    x = s.x.clone()
    x[at:, ch] = 0
    return replace(s, x=x), ch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scenarios", nargs="*", default=["crossday", "within", "electrode"])
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = f"v2_v6_{a.split}"
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds" if a.split == "val" else "seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    runs = [Path(r) for r in a.runs]
    logger = RunLogger(cfg, seeds, a.config)
    logger.save_json("inputs.json", {"runs": [str(r) for r in runs], "split": a.split, "scenarios": a.scenarios})
    ctx = Ctx(cfg, a.split)
    v6, dt = ctx.v6, ctx.dt
    rc = v6["recovery"]
    w_rec, s_rec = ms_to_steps(1000.0 * rc["window_s"], dt), ms_to_steps(1000.0 * rc["step_s"], dt)
    every = int(v6["trace_every"])
    rows: list[dict[str, Any]] = []

    def run(m: V2Model, pr: Prepared, conds: list[Condition], ref: Ref) -> V5Result:
        sl = pr.session.split(a.split)
        return stream_adapt_v5(m, pr.x, pr.y, sl.start, sl.stop, ctx.burnin, conds, CerebellumV2(ref.w0, ref.e2, cfg["cerebellum"]),
                               AffectSpaceV2(ctx.af), ctx.plast, ctx.alpha_sm, ctx.kw["anchor_decay"], ctx.kw["eps"],
                               cfg["v5"]["homeostasis"], chunk=ctx.kw["chunk"], label_delay=ctx.kw["label_delay"],
                               controls=ref.ctrl, drift=ref.drift, dyn_base=ref.base)

    def record(res: V5Result, conds: list[Condition], pr: Prepared, meta: dict[str, Any], t0: float,
               t_drop: int | None, skip_a: bool, store: dict[str, Any]) -> None:
        sl = pr.session.split(a.split)
        y = (pr.y[sl] * pr.y_sd + pr.y_mu).cpu()
        h = res.rate.shape[1] // 2
        for e, c in enumerate(conds):
            if skip_a and c.name == "A":
                continue
            pred = (res.pred[e] * pr.y_sd + pr.y_mu).cpu()
            lam_e = res.lam[e]
            if c.renorm and c.lam_surp > c.lam_calm and c.eta_seq is None:  # sensör kapısının açık olduğu adımlar
                op = lam_e >= 0.5 * (c.lam_calm + c.lam_surp)
                open_frac = float(op.float().mean())
                t_open = float(op.nonzero()[0, 0]) * dt / 1000.0 if bool(op.any()) else -1.0
            else:
                open_frac, t_open = float("nan"), float("nan")
            ex = res.extra
            row = {**meta, "session": pr.session.name, "cond": c.name, "r2": r2_neurobench(pred, y),
                   "dw_rel": float(res.dw_rel[e]), "anc_rel": float(ex["anc_rel"][e]),
                   "rate_fast": float(res.rate[e, :h].mean()), "rate_slow": float(res.rate[e, h:].mean()),
                   "eta_mean": float(res.eta[e].mean()), "alpha_mean": float(ex["alpha"][e].mean()),
                   "lam_mean": float(lam_e.mean()) if c.renorm else 0.0, "open_frac": open_frac, "t_open_s": t_open,
                   "a_pos": float(res.a[e].clamp(min=0).mean()), "s_sens_mean": float(ex["s_sens"][e].mean()),
                   "s_drift_mean": float(ex["s_drift"][e].mean()), "dyn": c.dyn is not None}
            if c.dyn is not None:
                for k in ("a_int", "val", "dom", "cons", "brake", "u"):
                    row[f"{k}_mean"] = float(ex[k][e].mean())
                row["a_int_pos"] = float(ex["a_int"][e].clamp(min=0).mean())
            if t_drop is not None:
                rec_s, cens, ref_r2, st, rr = recovery(pred, y, t_drop, w_rec, s_rec, rc["tol"], dt)
                row.update(r2_pre=r2_neurobench(pred[:t_drop], y[:t_drop]), r2_post=r2_neurobench(pred[t_drop:], y[t_drop:]),
                           rec_s=rec_s, rec_censored=cens, r2_win_ref=ref_r2)
                store.setdefault("rolling", {})[c.name] = (st, rr)
            row["sec"] = time.time() - t0
            rows.append(row)
            logger.log(len(rows), **row)
            if c.dyn is not None or c.renorm:
                store.setdefault("trace", {})[c.name] = {
                    "a": res.a[e, ::every].cpu(), "eta": res.eta[e, ::every].cpu(), "lam": lam_e[::every].cpu(),
                    **{k: ex[k][e, ::every].cpu() for k in TRACE_KEYS}}
        done = [r for r in rows if r["session"] == pr.session.name and r["seed"] == meta["seed"]
                and r["scenario"] == meta["scenario"]]
        print(json.dumps({**meta, "session": pr.session.name, "sec": round(time.time() - t0),
                          "r2": {r["cond"]: round(r["r2"], 3) for r in done}}, ensure_ascii=False), flush=True)

    def stream(m: V2Model, pr: Prepared, ref: Ref, t_drop: int | None = None, **meta: Any) -> None:
        conds = ctx.pass1(meta["scenario"], meta["day"])
        t0 = time.time()
        store: dict[str, Any] = {"t_drop": t_drop}
        res = run(m, pr, conds, ref)
        record(res, conds, pr, meta, t0, t_drop, False, store)
        if v6["matched"]:
            conds2 = ctx.pass2(conds, res, meta["seed"])
            del res
            t0 = time.time()
            record(run(m, pr, conds2, ref), conds2, pr, meta, t0, t_drop, True, store)
        torch.save(store, logger.path(f"trace_{meta['scenario']}_{pr.session.name}_seed{meta['seed']}.pt"))

    def load(ck: Path, C: int, D: int) -> V2Model:
        m = V2Model(cfg, C, D).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        return m

    el = v6["electrode"]
    for primate in a.primates or list(cfg["primates"]):
        names = list(cfg["primates"][primate])
        day0, _ = crossday_sessions(cfg, primate)
        sessions = {n: load_v2_session(cfg, n) for n in names}
        C, D = sessions[names[0]].D_in, sessions[names[0]].y.shape[1]
        for seed in seeds:
            if "crossday" in a.scenarios:
                ck = find_ckpt(runs, f"day0_{primate}_seed{seed}.pt")
                if ck is not None:
                    m = load(ck, C, D)
                    pr0 = prepare_session(cfg, sessions[day0], device)
                    ref = ctx.make_ref(m, pr0)
                    for n in names:
                        pr = pr0 if n == day0 else prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                        stream(m, pr, ref, primate=primate, seed=seed, scenario="crossday", day=names.index(n))
                        del pr
                    del pr0, m
            if {"within", "electrode"} & set(a.scenarios):
                for n in names:
                    ck = find_ckpt(runs, f"finetune_{n}_seed{seed}.pt")
                    if ck is None:
                        continue
                    m = load(ck, C, D)
                    pr = prepare_session(cfg, sessions[n], device)
                    ref = ctx.make_ref(m, pr)
                    if "within" in a.scenarios:
                        stream(m, pr, ref, primate=primate, seed=seed, scenario="within", day=names.index(n))
                    if "electrode" in a.scenarios:
                        sl = sessions[n].split(a.split)
                        at = sl.start + int(el["at_frac"] * (sl.stop - sl.start))
                        s_e, ch = drop_channels(sessions[n], at, el["n_channels"], 1000 * seed + names.index(n))
                        pr_e = prepare_session(cfg, s_e, device)             # kodlayıcı train'den: kayıptan etkilenmez
                        stream(m, pr_e, ref, t_drop=at - sl.start, primate=primate, seed=seed, scenario="electrode",
                               day=names.index(n), channels=ch)
                        del pr_e
                    del pr, m
            logger.save_json("results.json", rows)
            torch.cuda.empty_cache() if device.type == "cuda" else None

    summary(rows, logger, a.split, seeds)


def summary(rows: list[dict[str, Any]], logger: RunLogger, split: str, seeds: list[int]) -> None:
    """Kısa özet tablo (ayrıntı ve Wilcoxon: scripts/make_tables_v2.py)."""
    order = list(dict.fromkeys(r["cond"] for r in rows))
    groups = {"tüm günler": lambda r: r["scenario"] == "crossday", "Day-0": lambda r: r["scenario"] == "crossday" and r["day"] == 0,
              "Day-k": lambda r: r["scenario"] == "crossday" and r["day"] > 0, "oturum içi": lambda r: r["scenario"] == "within",
              "elektrot (sonrası)": lambda r: r["scenario"] == "electrode"}
    lines = ["| Koşul | " + " | ".join(groups) + " | toparlanma s | ‖W − W_0‖ Day-k | η ort. Day-k |",
             "|" + "---|" * (len(groups) + 4)]
    for c in order:
        cells = []
        for k, g in groups.items():
            v = [r["r2_post" if k.startswith("elektrot") else "r2"] for r in rows if r["cond"] == c and g(r)]
            cells.append(f"{np.mean(v):.3f}" if v else "—")
        rec = [r["rec_s"] for r in rows if r["cond"] == c and r["scenario"] == "electrode"]
        dk = [r for r in rows if r["cond"] == c and groups["Day-k"](r)]
        cells += [f"{np.mean(rec):.1f}" if rec else "—", f"{np.mean([r['dw_rel'] for r in dk]):.3f}" if dk else "—",
                  f"{np.mean([r['eta_mean'] for r in dk]):.2e}" if dk else "—"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", f"{split} R² (NeuroBench), seed {seeds}. Elektrot sütunu kayıp sonrası R²."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
