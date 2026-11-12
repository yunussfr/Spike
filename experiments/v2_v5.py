"""Faz V5: V3 checkpoint'leri üzerinde beyincik + duygu uzayı + duygu-kapılı yeniden ölçekleme (KR-045, §12.14)
ve (ops.) duygu-kapılı W_in plastisitesi (KR-044, §12.13).

Senaryolar (V4 ile aynı checkpoint'ler):
  crossday — `day0_<primat>_seed<s>.pt`; tüm oturumlar Day-0 istatistikleriyle kodlanır; beyincik Day-0 train'de
  within   — `finetune_<oturum>_seed<s>.pt`; beyincik oturumun kendi train diliminde başlatılır
Geçiş 1 (aynı akışta): v2-A (η = 0), v2-B (η_0), R_τ (v2-B + sabit τ'lu yeniden ölçekleme),
  G_τ (v2-D: duygu-kapılı yeniden ölçekleme τ_calm → τ_surp = τ, η(t), g), ops. W_e / D_e (W_in kuralı, η_in = e).
Geçiş 2 (`v5.matched`): her G_τ için λ̄_norm / η̄ ile eşlenmiş Rm_τ (v2-R) ve G_τ dizilerinin zamanda karıştırıldığı
  Cg_τ (v2-C); her D_e için Bm_e, Wm_e, C_e.
Kalıcı sürpriz (`persist.variants`, KR-046): ham hedefli donuk beyincik + sabit Day-0 tabanı + girdi sapması
  (δ/δ_0) + eşikli kapı; λ_norm a < lo'da τ_calm (donuk), a ≥ hi'de τ_surp. O: gün bilgisini kullanan üst sınır.
Ayar yalnızca `--split val` + ayar seed'leri; test koşusu seçilen değerlerle (`--set renorm.tau_surprise_s=[...]`).

Kullanım:
    uv run python experiments/v2_v5.py --config configs/v2_v5.yaml --runs runs/<v3 indy> runs/<v3 loco> --split val
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v2.metrics import r2_neurobench  # noqa: E402
from experiments.v2_online import find_ckpt  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.affect import AffectSpaceV2  # noqa: E402
from src.v2.cerebellum import CerebellumV2  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.model import V2Model, stream_predict  # noqa: E402
from src.v2.online import Controls, stream_kwargs  # noqa: E402
from src.v2.online_v5 import (Condition, Drift, Plasticity, collect_cerebellum_data, drift_baseline,  # noqa: E402
                              stream_adapt_v5)
from src.v2.train import Prepared, prepare_session  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scenarios", nargs="*", default=["crossday", "within"])
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--late_s", type=float, default=0.0,
                   help="ek değerlendirme: dilimin ilk late_s saniyesi atılarak R² (r2_late; tespit gecikmesi analizi)")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = f"v2_v5_{a.split}"
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    runs = [Path(r) for r in a.runs]
    logger = RunLogger(cfg, seeds, a.config)
    logger.save_json("inputs.json", {"runs": [str(r) for r in runs], "split": a.split})
    kw = stream_kwargs(cfg)
    dt = cfg["time"]["dt_ms"]
    burnin = ms_to_steps(cfg["online"]["burnin_ms"], dt)
    alpha_sm = math.exp(-dt / cfg["encoder"]["smooth_tau_ms"])
    plast = Plasticity.from_config(cfg)
    af, v5, tr = dict(cfg["affect"]), cfg["v5"], cfg["train"]
    for k in ("fast", "slow"):                                               # zaman sabiti ms ile verildiyse adım katsayısı
        if f"tau_{k}_ms" in af:
            af[f"a_{k}"] = 1.0 - math.exp(-dt / af[f"tau_{k}_ms"])
    grid = cfg["plasticity"]["eta_in_grid"]
    eta0 = af["eta_0"]
    rnc = cfg["renorm"]
    lam_of = lambda tau_s: 1.0 - math.exp(-dt / (1000.0 * tau_s))         # noqa: E731
    gated = [(float(c_), float(s_)) for c_, s_ in rnc["gated"]]            # (τ_calm, τ_surp) çiftleri, s
    gname = lambda c_, s_: f"G{c_:g}/{s_:g}"                                # noqa: E731
    pc = cfg.get("persist", {"variants": {}, "oracle": False})
    rows: list[dict[str, Any]] = []

    def drift_spec(ref: Prepared, ctrl: Controls) -> Drift | None:
        """δ_0: Day-0 train dilimi boyunca sapma ölçüsünün `persist.drift.baseline` istatistiği (max: tolerans zarfı,
        KR-046); akıştaki gibi m_0, v_0'dan başlar. Yalnızca train girdisi, etiket yok."""
        if not pc["variants"]:
            return None
        sl = ref.session.split("train")
        spec = Drift(lam=lam_of(pc["drift"]["tau_s"]), var_floor=pc["drift"]["var_floor"])
        spec.delta0 = drift_baseline(ref.x, sl.start, sl.stop, ctrl.renorm_m0, ctrl.renorm_v0, spec,
                                     stat=pc["drift"]["baseline"])
        return spec

    def persist_conds(scenario: str, day: int) -> list[Condition]:
        out = []
        for name, v in pc["variants"].items():
            full = bool(v.get("full", False))
            g_mod, eta_mod = bool(v.get("g", full)), bool(v.get("eta", full))   # full = g_fast/g_slow + η(t)
            lo, hi = v.get("gate", (pc["gate_lo"], pc["gate_hi"]))
            out.append(Condition(name, eta0=eta0 * float(v.get("eta_scale", 1.0)), renorm=True,
                                 lam_calm=lam_of(pc["tau_calm_s"]), lam_surp=lam_of(pc["tau_surp_s"]), affect=True,
                                 gate=g_mod, alpha_eta=af["alpha_eta"] if eta_mod else 0.0, cer_raw=True, cer_eta=0.0,
                                 cer_src=0 if v["src"] == "monitor" else -1, fixed_slow=True,
                                 a_fast=1.0 - math.exp(-dt / (1000.0 * pc["tau_fast_s"])),
                                 drift_w=float(v.get("w", 0.0)), drift_max=v.get("combine") == "max",
                                 gate_lo=float(lo), gate_hi=float(hi),
                                 lam_return=lam_of(pc["tau_return_s"]) if v.get("ret", True) else 0.0))
        if pc["oracle"]:                                                     # gün bilgisini kullanan üst sınır
            out.append(Condition("O", eta0=eta0, renorm=True,
                                 lam_calm=lam_of(pc["tau_surp_s"]) if scenario == "crossday" and day > 0 else 0.0))
        return out

    def controls(m: V2Model, ref: Prepared) -> Controls:
        sl = ref.session.split("train")
        xt = ref.x[sl]
        c = Controls(renorm_m0=xt.mean(0), renorm_v0=xt.var(0), renorm_eps=rnc["eps"])   # §12.14 referansı (train)
        if v5.get("homeo_target", "fixed") == "day0":
            _, c.homeo_target = stream_predict(m, ref.x, sl.start, sl.stop, tr["eval_streams"],
                                               ms_to_steps(tr["eval_burnin_ms"], dt), tr["eval_graph_chunk"])
        return c

    def cerebellum(m: V2Model, ref: Prepared) -> tuple[torch.Tensor, float]:
        """Train diliminin sonundan fit_steps adımla ridge; E_slow başlangıcı onu izleyen holdout_steps adımın
        (yine train) hatası — örneklem içi hata Day-0 val'in ~1,5–1,9 katı altında kalıyordu."""
        sl = ref.session.split("train")
        n, h = cfg["cerebellum"]["fit_steps"], cfg["cerebellum"]["holdout_steps"]
        phi, tgt = collect_cerebellum_data(m, ref.x, max(sl.start, sl.stop - n - h), sl.stop, burnin, alpha_sm)
        w, _ = CerebellumV2.fit(phi[:-h], tgt[:-h], cfg["cerebellum"]["ridge"])
        e2 = float(((tgt[-h:] - phi[-h:] @ w.T) ** 2).mean())
        return w, e2

    def run(m: V2Model, pr: Prepared, conds: list[Condition], w0: torch.Tensor, e2: float, ctrl: Controls,
            dspec: Drift | None):
        sl = pr.session.split(a.split)
        return stream_adapt_v5(m, pr.x, pr.y, sl.start, sl.stop, burnin, conds, CerebellumV2(w0, e2, cfg["cerebellum"]),
                               AffectSpaceV2(af), plast, alpha_sm, kw["anchor_decay"], kw["eps"], v5["homeostasis"],
                               chunk=kw["chunk"], label_delay=kw["label_delay"], controls=ctrl, drift=dspec)

    def record(res: Any, conds: list[Condition], pr: Prepared, meta: dict[str, Any], t0: float) -> None:
        sl = pr.session.split(a.split)
        y = (pr.y[sl] * pr.y_sd + pr.y_mu).cpu()
        late = ms_to_steps(1000.0 * a.late_s, dt) if a.late_s > 0 else 0
        h = res.rate.shape[1] // 2
        for e, c in enumerate(conds):
            lam_e = res.lam[e]
            if c.renorm and c.lam_surp > c.lam_calm and c.eta_seq is None:  # kapının açık olduğu adımlar (h ≥ 0,5)
                op = lam_e >= 0.5 * (c.lam_calm + c.lam_surp)
                open_frac = float(op.float().mean())
                t_open = float(op.nonzero()[0, 0]) * dt / 1000.0 if bool(op.any()) else -1.0
            else:
                open_frac, t_open = float("nan"), float("nan")
            row = {**meta, "session": pr.session.name, "cond": c.name,
                   "r2": r2_neurobench((res.pred[e] * pr.y_sd + pr.y_mu).cpu(), y),
                   "r2_late": r2_neurobench((res.pred[e, late:] * pr.y_sd + pr.y_mu).cpu(), y[late:]) if late else float("nan"),
                   "din_rel": float(res.din_rel[e]), "din_max": float(res.din_max[e]), "dw_rel": float(res.dw_rel[e]),
                   "a_mean": float(res.a[e].mean()), "a_pos": float(res.a[e].clamp(min=0).mean()),
                   "eta_mean": float(res.eta[e].mean()), "eta_in_mean": float(res.eta_in[e].mean()),
                   "lam_mean": float(res.lam[e].mean()) if c.renorm else 0.0,
                   "rate_fast": float(res.rate[e, :h].mean()), "rate_slow": float(res.rate[e, h:].mean()),
                   "open_frac": open_frac, "t_open_s": t_open, "a_med": float(res.a[e].median()),
                   "s_cer_mean": float(res.extra["s_cer"][e].mean()), "s_drift_mean": float(res.extra["s_drift"][e].mean()),
                   "sec": time.time() - t0}
            rows.append(row)
            logger.log(len(rows), **row)
        print(json.dumps({**meta, "session": pr.session.name, "sec": round(time.time() - t0),
                          "r2": {r["cond"]: round(r["r2"], 3) for r in rows[-len(conds):]}}), flush=True)

    def stream(m: V2Model, pr: Prepared, w0: torch.Tensor, e2: float, ctrl: Controls, dspec: Drift | None,
               **meta: Any) -> None:
        conds = ([Condition("A"), Condition("B", eta0=eta0)]
                 + [Condition(f"R{t:g}", eta0=eta0, renorm=True, lam_calm=lam_of(t)) for t in rnc["tau_fixed_s"]]
                 + [Condition(gname(c_, s_), eta0=eta0, renorm=True, lam_calm=lam_of(c_), lam_surp=lam_of(s_), affect=True,
                              gate=True, alpha_eta=af["alpha_eta"]) for c_, s_ in gated]
                 + [Condition(f"W{e:g}", eta0=eta0, eta_in0=e) for e in grid]
                 + [Condition(f"D{e:g}", eta0=eta0, eta_in0=e, affect=True, gate=True, alpha_eta=af["alpha_eta"])
                    for e in grid]
                 + persist_conds(meta["scenario"], meta["day"]))                 # cer_src = 0 → A (izleme yolu)
        t0 = time.time()
        res = run(m, pr, conds, w0, e2, ctrl, dspec)
        record(res, conds, pr, meta, t0)
        trace = {c.name: {"a": res.a[i, ::250].cpu(), "eta_in": res.eta_in[i, ::250].cpu(), "lam": res.lam[i, ::250].cpu(),
                          "s_cer": res.extra["s_cer"][i, ::250].cpu(), "s_drift": res.extra["s_drift"][i, ::250].cpu()}
                 for i, c in enumerate(conds) if c.affect}
        torch.save(trace, logger.path(f"trace_{meta['scenario']}_{pr.session.name}_seed{meta['seed']}.pt"))
        if not v5["matched"]:
            return
        g = torch.Generator(device="cpu").manual_seed(int(meta["seed"]))
        conds2 = []
        names_ = [c.name for c in conds]
        for c_, s_ in gated:
            i = names_.index(gname(c_, s_))
            perm = torch.randperm(res.eta.shape[1], generator=g).to(res.eta.device)
            conds2 += [Condition(f"Rm{c_:g}/{s_:g}", eta0=float(res.eta[i].mean()), renorm=True,
                                 lam_calm=float(res.lam[i].mean())),
                       Condition(f"Cg{c_:g}/{s_:g}", renorm=True, lam_calm=lam_of(c_), affect=True, gate=True,
                                 alpha_eta=af["alpha_eta"], eta_seq=res.eta[i][perm].clone(),
                                 eta_in_seq=torch.zeros_like(res.eta[i]), lam_seq=res.lam[i][perm].clone())]
        for e in grid:
            i = [c.name for c in conds].index(f"D{e:g}")
            perm = torch.randperm(res.eta.shape[1], generator=g).to(res.eta.device)
            conds2 += [Condition(f"Bm{e:g}", eta0=float(res.eta[i].mean())),
                       Condition(f"Wm{e:g}", eta0=float(res.eta[i].mean()), eta_in0=float(res.eta_in[i].mean())),
                       Condition(f"C{e:g}", affect=True, gate=True, alpha_eta=af["alpha_eta"],
                                 eta_seq=res.eta[i][perm].clone(), eta_in_seq=res.eta_in[i][perm].clone())]
        del res
        t0 = time.time()
        if conds2:
            record(run(m, pr, conds2, w0, e2, ctrl, dspec), conds2, pr, meta, t0)

    for primate in a.primates or list(cfg["primates"]):
        names = list(cfg["primates"][primate])
        day0, _ = crossday_sessions(cfg, primate)
        sessions = {n: load_v2_session(cfg, n) for n in names}
        C, D = sessions[names[0]].D_in, sessions[names[0]].y.shape[1]
        for seed in seeds:
            if "crossday" in a.scenarios:
                ck = find_ckpt(runs, f"day0_{primate}_seed{seed}.pt")
                if ck is not None:
                    m = V2Model(cfg, C, D).to(device)
                    m.load_state_dict(torch.load(ck, map_location=device))
                    ref = prepare_session(cfg, sessions[day0], device)
                    w0, e2 = cerebellum(m, ref)
                    ctrl = controls(m, ref)
                    dspec = drift_spec(ref, ctrl)
                    for n in names:
                        pr = ref if n == day0 else prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                        stream(m, pr, w0, e2, ctrl, dspec, primate=primate, seed=seed, scenario="crossday",
                               day=names.index(n))
                        del pr
                    del ref
            if "within" in a.scenarios:
                for n in names:
                    ck = find_ckpt(runs, f"finetune_{n}_seed{seed}.pt")
                    if ck is None:
                        continue
                    m = V2Model(cfg, C, D).to(device)
                    m.load_state_dict(torch.load(ck, map_location=device))
                    pr = prepare_session(cfg, sessions[n], device)
                    w0, e2 = cerebellum(m, pr)
                    ctrl = controls(m, pr)
                    stream(m, pr, w0, e2, ctrl, drift_spec(pr, ctrl), primate=primate, seed=seed, scenario="within",
                           day=names.index(n))
            logger.save_json("results.json", rows)
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # tablolar: oturum × koşul (ortalama ± std, seed'ler) ve günler arası Day-k özeti
    order = list(dict.fromkeys(r["cond"] for r in rows))
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["scenario"], r["session"], r["cond"]), []).append(r)
    lines = ["| Senaryo | Oturum | " + " | ".join(order) + " |", "|" + "---|" * (2 + len(order))]
    for sc, sess in dict.fromkeys((k[0], k[1]) for k in by):
        cells = []
        for c in order:
            v = np.array([r["r2"] for r in by.get((sc, sess, c), [])])
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "—")
        lines.append(f"| {sc} | {sess} | " + " | ".join(cells) + " |")
    lines += ["", "| Koşul | Day-k R² ort. (crossday) | Day-0 R² | oturum içi R² | a⁺ ort. (Day-k) | a⁺ ort. (Day-0) "
              "| λ_norm ort. (Day-k / Day-0) | kapı açık (Day-k / Day-0 / içi) | s_cer (Day-k / Day-0) "
              "| s_drift (Day-k / Day-0) | ‖ΔW_in‖ maks |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in order:
        dk = [r for r in rows if r["cond"] == c and r["scenario"] == "crossday" and r["day"] > 0]
        d0 = [r for r in rows if r["cond"] == c and r["scenario"] == "crossday" and r["day"] == 0]
        wi = [r for r in rows if r["cond"] == c and r["scenario"] == "within"]
        f = lambda rs, k: f"{np.mean([r[k] for r in rs]):.3f}" if rs else "—"   # noqa: E731
        g_ = lambda rs, k: f"{np.mean([r[k] for r in rs]):.2e}" if rs else "—"   # noqa: E731
        o_ = lambda rs: f"{np.nanmean([r['open_frac'] for r in rs]):.2f}" if rs and not all(   # noqa: E731
            np.isnan(r["open_frac"]) for r in rs) else "—"
        lines.append(f"| {c} | {f(dk, 'r2')} | {f(d0, 'r2')} | {f(wi, 'r2')} | {f(dk, 'a_pos')} | {f(d0, 'a_pos')} "
                     f"| {g_(dk, 'lam_mean')} / {g_(d0, 'lam_mean')} | {o_(dk)} / {o_(d0)} / {o_(wi)} "
                     f"| {f(dk, 's_cer_mean')} / {f(d0, 's_cer_mean')} | {f(dk, 's_drift_mean')} / {f(d0, 's_drift_mean')} "
                     f"| {f(dk + d0 + wi, 'din_max')} |")
    lines += ["", f"{a.split} R² (NeuroBench), seed {seeds}, etiket gecikmesi {cfg['online']['label_delay_ms']:g} ms, "
              f"η_0 = {eta0:g}. R_τ: sabit τ (s) yeniden ölçekleme; G_c/s: duygu-kapılı, τ_calm = c s → τ_surp = s s; "
              "Rm_c/s: G_c/s'nin λ̄ / η̄'sıyla eşlenmiş sabit; Cg_c/s: G_c/s dizileri karıştırılmış. W_e / D_e: W_in kuralı (KR-044). "
              f"Kalıcı sürpriz (KR-046): {', '.join(f'{k} = {v}' for k, v in pc['variants'].items()) or '—'}; "
              "O: Day-0'da donuk, Day-k'de τ_surp (gün bilgisiyle üst sınır). s_cer, s_drift: Day-0 tabanına göre (taban = 1)."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
