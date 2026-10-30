"""Faz 3: NeuroBench primat motor tahmini, konfigürasyon A ve B.

Akış (her oturum ve seed için):
  1. Oturum yüklenir, kanal başına ölçek yalnızca train'den hesaplanır (src/data/primate.py).
  2. Sabit rezervuar kurulur ve θ(τ) kalibre edilir. Regülatörler kapalı (K5), rezervuar sabit
     (K1): r(t) öğrenmeden bağımsızdır → tüm dosya için bir kez üretilir, A ve B aynı r'yi okur.
  3. A ve B ajanları train bölmesinde çevrimiçi öğrenir (`nhp.epochs` tur); val/test'te
     ağırlıklar ve normalizasyon donuktur.
  4. Metrikler: test/val R² (NeuroBench R2 + kendi R²'miz), ridge üst sınırı (λ val'de),
     ortalama-tahmin tabanı, ateşleme oranı (K7), beyincik tahmin MSE'si ve "tekrarla" tabanı.

Kullanım:
    uv run python experiments/nhp_motor.py --config configs/nhp_motor.yaml [--files ...] [--seeds 0 1]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `experiments.common` için depo kökü
import copy
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch

from experiments.common import (build_reservoir, collect_features, firing_rate, r2_columns,
                                r2_neurobench, ridge_reference)
from src.agent import Agent
from src.data.primate import Session, load_session, prepare
from src.utils.config import apply_overrides, load_config, resolve_device
from src.utils.logging import RunLogger


def make_agent(cfg: dict[str, Any], c: str, s: Session, res: Any, seed: int) -> Agent:
    """c: "A", "B" ya da "B+Z" (Faz 4: B + duygu uzayı, adım 9)."""
    cc = copy.deepcopy(cfg)
    cc["agent"]["config"] = c.split("+")[0]
    cc["affect"]["enabled"] = c.endswith("+Z")
    return Agent(cc, D_in=s.D_in, D_out=s.y.shape[1], seed=seed, reservoir=res)


def _trace(out: dict[str, Any], log: dict[str, list[torch.Tensor]]) -> None:
    """Duygu uzayı izleri: z, baş tahminleri ve hedefler (özgün ölçek), standart baş hataları."""
    if "z" not in out:
        return
    log["z"].append(out["z"][0])
    log["heads"].append(torch.stack([out["heads"][h][0] for h in ("q", "u", "s")]))
    log["targets"].append(torch.stack([out["targets"][h][0] for h in ("q", "u", "s")]))
    log["head_err"].append(out["head_err"][0])


def train_and_predict(cfg: dict[str, Any], r: torch.Tensor, s: Session, configs: list[str],
                      res: Any, seed: int) -> dict[str, dict[str, Any]]:
    """Ajanları aynı r akışıyla eğitir. → config başına val/test tahminleri, beyincik hataları, duygu izleri."""
    nh = cfg["nhp"]
    agents = {c: make_agent(cfg, c, s, res, seed) for c in configs}
    x, y = s.x, s.y
    tr = s.split("train")
    curve: dict[str, list[float]] = {c: [] for c in configs}
    keys = ("z", "heads", "targets", "head_err")
    train_log = {c: {k: [] for k in keys} for c in configs}
    head_curve: dict[str, list[list[float]]] = {c: [] for c in configs}       # tur başına baş MSE (standart ölçek)
    for epoch in range(nh["epochs"]):
        last = epoch == nh["epochs"] - 1
        sq = {c: 0.0 for c in configs}
        hsq = {c: torch.zeros(3) for c in configs}
        for t in range(tr.start + nh["washout"], tr.stop):
            rt, yt = r[t: t + 1], y[t: t + 1]
            x_next = x[t + 1: t + 2] if t + 1 < tr.stop else None            # x(t+1) yalnızca train içinden
            for c, ag in agents.items():
                ag.act(rt)
                o = ag.learn(yt, x_next)
                sq[c] += o["delta"].pow(2).mean().item()
                if "head_err" in o:
                    hsq[c] += o["head_err"][0].pow(2)
                if last:
                    _trace(o, train_log[c])
        n = tr.stop - tr.start - nh["washout"]
        for c in configs:
            curve[c].append(sq[c] / n)
            if agents[c].affect is not None:
                head_curve[c].append((hsq[c] / n).tolist())
            if not math.isfinite(curve[c][-1]):
                raise FloatingPointError(f"{s.name} seed {seed} config {c}: eğitim ıraksadı")

    out: dict[str, dict[str, Any]] = {}
    for c, ag in agents.items():
        ag.set_eval(True)
        preds, e_sq = [], []
        eval_log = {k: [] for k in keys}
        for t in range(tr.stop, len(x)):                                      # val + test, donuk
            preds.append(ag.act(r[t: t + 1]))
            if ag.cerebellum is not None and t + 1 < len(x):
                o = ag.errors(y[t: t + 1], x[t + 1: t + 2])
                e_sq.append(o["e"].pow(2).mean().item())
                _trace(o, eval_log)
        P = torch.cat(preds)                                                  # [n_val + n_test, 2]
        out[c] = {"val": P[: s.test_start - s.val_start], "test": P[s.test_start - tr.stop:],
                  "curve": curve[c], "e_sq": e_sq}
        if ag.affect is not None:                                             # eval izi val başından, son adım hariç
            out[c]["affect"] = {"train": {k: torch.stack(v) for k, v in train_log[c].items()},
                                "eval": {k: torch.stack(v) for k, v in eval_log.items()},
                                "train_start": tr.start + nh["washout"], "eval_start": tr.stop,
                                "head_curve": torch.tensor(head_curve[c])}                # [epochs, 3]
    return out


def evaluate_session(cfg: dict[str, Any], name: str, seed: int, device: torch.device,
                     configs: list[str], with_ridge: bool = True) -> dict[str, Any]:
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    nh = cfg_s["nhp"]
    if nh.get("max_steps"):                                                   # duman koşusu: kısaltılmış oturum
        s = shorten(s, nh["max_steps"])
    t0 = time.time()
    res, calib_rates = build_reservoir(cfg_s, s.D_in, seed, device, x_calib=s.x[s.split("train")])
    r = collect_features(cfg_s, res, s.x, device)
    t_res = time.time() - t0
    val, test = s.split("val"), s.split("test")
    rho = cfg_s["reservoir"]["rho"]
    row: dict[str, Any] = {
        "session": name, "seed": seed, "D_in": s.D_in, "n_steps": len(s.x),
        "firing_rate": firing_rate(r, res.N, slice(nh["washout"], len(r))),
        "calib_rate_tau_default": float(np.interp(math.log(cfg_s["regulators"]["defaults"]["tau"]),
                                                  np.log(res.theta_tau.cpu().numpy()), calib_rates.cpu().numpy())),
        "sec_reservoir": t_res,
    }
    row["k7_ok"] = bool(rho / 2 <= row["firing_rate"] <= 2 * rho)
    y_test, y_val = s.y[test], s.y[val]
    row["r2_mean_baseline"] = r2_neurobench(s.y[s.split("train")].mean(dim=0).expand_as(y_test), y_test)

    t0 = time.time()
    outs = train_and_predict(cfg_s, r, s, configs, res, seed)
    row["sec_readouts"] = time.time() - t0
    for c, o in outs.items():
        row[f"r2_{c}"] = r2_neurobench(o["test"], y_test)
        row[f"r2_{c}_val"] = r2_neurobench(o["val"], y_val)
        row[f"r2_{c}_xy_own"] = r2_columns(o["test"], y_test)
        row[f"train_mse_{c}_last"] = o["curve"][-1]
        if o["e_sq"]:
            row[f"cereb_mse_{c}"] = float(np.mean(o["e_sq"]))
    if "B" in outs:                                                           # beyincik tabanı: x̂(t+1) = x(t)
        xs = s.x[s.split("train").stop:]
        row["cereb_mse_repeat"] = (xs[1:] - xs[:-1]).pow(2).mean().item()
    if with_ridge:
        pred, lam = ridge_reference(r, s.y, s.split("train"), val, test, nh["ridge_grid"], nh["washout"])
        row["r2_ridge"], row["ridge_lambda"] = r2_neurobench(pred, y_test), lam
    if "B+Z" in outs and "B" in outs:                                        # Faz 4 sızıntı kontrolü
        row["leak_free"] = bool(torch.equal(outs["B+Z"]["test"], outs["B"]["test"]))
    row["preds"] = {c: o["test"] for c, o in outs.items()}
    row["affect"] = {c: o["affect"] for c, o in outs.items() if "affect" in o}
    row["curves"] = {c: o["curve"] for c, o in outs.items()}
    return row


def shorten(s: Session, n: int) -> Session:
    """Duman koşusu için: her bölmeden orantılı kısa parça (zaman sırası korunur)."""
    frac = n / len(s.x)
    parts = [s.split(p) for p in ("train", "val", "test")]
    keep = [slice(p.start, p.start + max(20, int((p.stop - p.start) * frac))) for p in parts]
    x = torch.cat([s.x[k] for k in keep])
    y = torch.cat([s.y[k] for k in keep])
    n_tr, n_va = keep[0].stop - keep[0].start, keep[1].stop - keep[1].start
    return Session(s.name, x, y, n_tr, n_tr + n_va)


def summarize(rows: list[dict[str, Any]], configs: list[str]) -> dict[str, Any]:
    keys = [f"r2_{c}" for c in configs] + ["r2_ridge", "r2_mean_baseline", "firing_rate"]
    summary: dict[str, Any] = {}
    for name in dict.fromkeys(r["session"] for r in rows):
        rs = [r for r in rows if r["session"] == name]
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))}
                         for k in keys if k in rs[0]}
        if {"A", "B"} <= set(configs):
            diff = [r["r2_B"] - r["r2_A"] for r in rs]                       # eşli fark, seed başına
            summary[name]["B_minus_A"] = {"mean": float(np.mean(diff)), "std": float(np.std(diff))}
    return summary


def _task(args: tuple) -> dict[str, Any]:
    """Tek (oturum, seed) işi; ayrı süreçte, 1 iş parçacığıyla (adım adım akışta en hızlısı)."""
    cfg, name, seed, configs, with_ridge, out_dir = args
    torch.set_num_threads(1)
    device = resolve_device(cfg["experiment"]["device"])
    row = evaluate_session(cfg, name, seed, device, configs, with_ridge=with_ridge)
    torch.save({"preds": row.pop("preds"), "curves": row.pop("curves"), "affect": row.pop("affect")},
               Path(out_dir) / f"preds_{name}_seed{seed}.pt")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--no-ridge", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="paralel süreç sayısı (oturum × seed işleri)")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    files = args.files or cfg["data"]["files"]
    seeds = args.seeds if args.seeds is not None else cfg["experiment"]["seeds"]
    configs = cfg["nhp"]["configs"]
    logger = RunLogger(cfg, seeds, args.config)
    tasks = [(cfg, name, seed, configs, not args.no_ridge, str(logger.dir)) for name in files for seed in seeds]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_task, tasks):
            rows.append(row)
            logger.log(len(rows) - 1, **row)
            print(f"{row['session']} seed {row['seed']}: " + "  ".join(
                f"{k}={v:.4f}" for k, v in row.items() if k.startswith("r2_") and isinstance(v, float))
                + f"  rate={row['firing_rate']:.4f}  ({row['sec_reservoir']:.0f}+{row['sec_readouts']:.0f} s)",
                flush=True)
            if not row["k7_ok"]:
                print("  UYARI (K7): atesleme orani [rho/2, 2rho] disinda", flush=True)
    summary = summarize(rows, configs)
    logger.save_json("summary.json", summary)
    print(f"\nlog: {logger.dir}")


if __name__ == "__main__":
    main()
