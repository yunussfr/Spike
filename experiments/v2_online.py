"""Faz V4: V3 checkpoint'leri üzerinde akışlı çevrimiçi çapalı NLMS (v2-A: η = 0, v2-B: η > 0).

Senaryolar:
  within   — oturum içi: `finetune_<oturum>_seed<s>.pt`, oturumun kendi istatistikleri, dilim val/test
  crossday — günler arası: `day0_<primat>_seed<s>.pt`; Day-0 ve Day-k oturumları Day-0 istatistikleriyle kodlanır
Her (oturum, varyant) için tüm η ızgarası tek akışta birlikte koşar. Varyantlar `controls.variants`'tan
(KR-044 etiketsiz kontroller: Day-0 hedefli homeostaz, akan kanal z-skoru); yoksa `online.homeostasis` listesi.
Day-0 hedefi ve yeniden ölçekleme istatistikleri referans oturumun (within: kendisi, crossday: Day-0) train
diliminden ölçülür (KV6). η seçimi yalnızca `--split val` ile;
test koşusunda `--etas` ile val'de seçilen değer(ler) verilir.

Kullanım:
    uv run python experiments/v2_online.py --config configs/v2_online.yaml --runs runs/<v3 indy> runs/<v3 loco> \
        --seeds 100 101 102 --split val
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import r2_neurobench  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.model import V2Model, stream_predict  # noqa: E402
from src.v2.online import Controls, stream_adapt, stream_kwargs  # noqa: E402
from src.v2.train import Prepared, prepare_session  # noqa: E402


def find_ckpt(runs: list[Path], name: str) -> Path | None:
    for r in runs:
        p = r / name
        if p.exists():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="V3 koşu klasörleri (checkpoint'ler)")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--scenarios", nargs="*", default=["within", "crossday"])
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--etas", nargs="*", type=float, default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = f"v2_online_{a.split}"
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds"]
    etas = a.etas if a.etas is not None else cfg["online"]["eta_grid"]
    if 0.0 not in etas:
        etas = [0.0] + etas                                                  # v2-A her zaman
    device = resolve_device(cfg["experiment"]["device"])
    runs = [Path(r) for r in a.runs]
    logger = RunLogger(cfg, seeds, a.config)
    logger.save_json("inputs.json", {"runs": [str(r) for r in runs], "etas": etas, "split": a.split})
    kw = stream_kwargs(cfg)
    burnin = ms_to_steps(cfg["online"]["burnin_ms"], cfg["time"]["dt_ms"])
    rows: list[dict[str, Any]] = []
    cc = cfg.get("controls", {})
    variants = cc.get("variants") or [{"name": f"homeo={h}", "homeo": h} for h in cfg["online"]["homeostasis"]]
    dt = cfg["time"]["dt_ms"]
    tr = cfg["train"]

    def reference(m: V2Model, ref: Prepared) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Referans oturumun train diliminde donuk modelin nöron oranı [N] ve x_norm ortalaması / varyansı [C]."""
        sl = ref.session.split("train")
        _, rate = stream_predict(m, ref.x, sl.start, sl.stop, tr["eval_streams"],
                                 ms_to_steps(tr["eval_burnin_ms"], dt), tr["eval_graph_chunk"])
        xt = ref.x[sl]
        return rate, xt.mean(0), xt.var(0)

    def controls(v: dict[str, Any], m: V2Model, ref_stats: tuple) -> Controls:
        rate, mu, var = ref_stats
        c = Controls(homeo_lr=v.get("homeo_lr"))
        if v.get("target", "fixed") == "day0":
            c.homeo_target = rate
        if v.get("renorm", False):
            c.renorm_m0, c.renorm_v0 = mu, var
            c.renorm_decay = float(np.exp(-dt / (1000.0 * v.get("renorm_tau_s", cc["renorm_tau_s"]))))
            c.renorm_eps = cc["renorm_eps"]
        return c

    def stream(m: V2Model, pr: Prepared, ref: Prepared, **meta: Any) -> None:
        sl = pr.session.split(a.split)
        ref_stats = reference(m, ref)
        for v in variants:
            homeo = bool(v.get("homeo", False))
            t0 = time.time()
            res = stream_adapt(m, pr.x, pr.y, sl.start, sl.stop, burnin, etas, homeostasis=homeo,
                               controls=controls(v, m, ref_stats), **kw)
            y = (pr.y[sl] * pr.y_sd + pr.y_mu).cpu()
            d = kw["label_delay"]                                            # kopyalama tabanı: v̂_t = y_{t−d} (en son gelen etiket)
            copy_r2 = r2_neurobench(pr.session.y[sl.start - d: sl.stop - d], pr.session.y[sl]) if d else float("nan")
            h = m.backbone.half
            for j, eta in enumerate(etas):
                r2 = r2_neurobench((res.pred[j] * pr.y_sd + pr.y_mu).cpu(), y)
                row = {**meta, "session": pr.session.name, "variant": v["name"], "homeostasis": homeo, "eta": eta,
                       "r2": r2, "copy_r2": copy_r2,
                       "dw_rel": float(res.dw_rel[j]), "rate_fast": float(res.rate[:h].mean()),
                       "rate_slow": float(res.rate[h:].mean()), "theta_mean": float(res.theta.mean()),
                       "sec": time.time() - t0}
                rows.append(row)
                logger.log(len(rows), **row)
            print(json.dumps({**meta, "session": pr.session.name, "variant": v["name"], "sec": round(time.time() - t0),
                              "r2": {f"{e:g}": round(r["r2"], 3) for e, r in zip(etas, rows[-len(etas):])}}), flush=True)
        logger.save_json("results.json", rows)

    for primate in a.primates or list(cfg["primates"]):
        names = list(cfg["primates"][primate])
        day0, dayk = crossday_sessions(cfg, primate)
        sessions = {n: load_v2_session(cfg, n) for n in names}
        C, D = sessions[names[0]].D_in, sessions[names[0]].y.shape[1]
        for seed in seeds:
            if "within" in a.scenarios:
                for n in names:
                    ck = find_ckpt(runs, f"finetune_{n}_seed{seed}.pt")
                    if ck is None:
                        continue
                    m = V2Model(cfg, C, D).to(device)
                    m.load_state_dict(torch.load(ck, map_location=device))
                    pr = prepare_session(cfg, sessions[n], device)
                    stream(m, pr, pr, primate=primate, seed=seed, scenario="within",
                           day=0 if n == day0 else names.index(n))
            if "crossday" in a.scenarios:
                ck = find_ckpt(runs, f"day0_{primate}_seed{seed}.pt")
                if ck is not None:
                    m = V2Model(cfg, C, D).to(device)
                    m.load_state_dict(torch.load(ck, map_location=device))
                    ref = prepare_session(cfg, sessions[day0], device)
                    for n in names:
                        pr = ref if n == day0 else prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                        stream(m, pr, ref, primate=primate, seed=seed, scenario="crossday", day=names.index(n))
                        del pr
                    del ref

    # tablo: senaryo × oturum × homeostaz, her η için ortalama ± std (seed'ler)
    groups: dict[tuple, dict[float, list[float]]] = {}
    copy_of: dict[str, float] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["session"], r["variant"]), {}).setdefault(r["eta"], []).append(r["r2"])
        copy_of[r["session"]] = r["copy_r2"]
    head = " | ".join(f"η={e:g}" for e in etas)
    lines = [f"| Senaryo | Oturum | varyant | {head} | kopya y(t−d) |", "|" + "---|" * (4 + len(etas))]
    for (sc, n, ho), by_eta in groups.items():
        cells = " | ".join(f"{np.mean(by_eta[e]):.3f} ± {np.std(by_eta[e]):.3f}" for e in etas)
        lines.append(f"| {sc} | {n} | {ho} | {cells} | {copy_of[n]:.3f} |")
    # η seçimi önerisi: senaryo ve homeostaz başına, oturum ortalaması en yüksek η (yalnızca val'de anlamlı)
    lines += ["", "| Senaryo | varyant | en iyi η (oturum ort.) | R² ort. | η = 0 R² ort. |", "|---|---|---|---|---|"]
    for sc in a.scenarios:
        for ho in [v["name"] for v in variants]:
            means = {e: np.mean([np.mean(g[e]) for (s_, _, h_), g in groups.items() if s_ == sc and h_ == ho])
                     for e in etas if any(s_ == sc and h_ == ho for (s_, _, h_) in groups)}
            if means:
                best = max(means, key=means.get)
                lines.append(f"| {sc} | {ho} | {best:g} | {means[best]:.3f} | {means[0.0]:.3f} |")
    lines += ["", f"{a.split} R² (NeuroBench), seed {seeds}, etiket gecikmesi {cfg['online']['label_delay_ms']:g} ms. "
              "crossday: Day-0 modeli, tüm oturumlar Day-0 istatistikleriyle. kopya: v̂_t = y_{t−d}."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
