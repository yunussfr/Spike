"""Faz 3 + 4 tam koşu: 6 oturum × 5 seed × {A, B, B+Z}, iki aşamada.

Aşama 1 (`stage1`, CPU, (oturum, seed) başına bir süreç):
  rezervuar kurulur ve kalibre edilir, r(t) üretilir ve diske yazılır (`<run>/r/<oturum>_seed<s>.npy`,
  float32). Ateşleme oranı (K7), ortalama tabanı ve ridge üst sınırı burada hesaplanır.
Aşama 2 (`stage2`, oturum başına bir süreç):
  5 seed'in okuma blokları `src/batched.py` ile tek tensörde eğitilir (A, B, B+Z ayrı gruplar).
  r diskten parça parça okunur (bellek eşlemesi). Sonuçlar `nhp_motor.py` ile aynı biçimdedir:
  `metrics.jsonl` (satır = oturum × seed), `preds_<oturum>_seed<s>.pt`. Böylece
  `scripts/make_tables.py` ve `experiments/affect_viz.py` doğrudan kullanılır.

Kullanım:
    uv run python experiments/nhp_full.py --config configs/nhp_motor.yaml --workers1 10 --workers2 6
    (--run <klasör> ile yarıda kalan bir koşu sürdürülebilir; biten işler atlanır)
"""

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (build_reservoir, collect_features, firing_rate, r2_columns,  # noqa: E402
                                r2_neurobench, ridge_reference)
from experiments.nhp_motor import make_agent, summarize  # noqa: E402
from src.batched import BatchedAgent  # noqa: E402
from src.data.primate import Session, load_session, prepare  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402

TRACE_KEYS = ("z", "heads", "targets", "head_err")


# --- Aşama 1: rezervuar ------------------------------------------------------------------

def stage1_job(args: tuple) -> dict[str, Any]:
    cfg, name, seed, run_dir = args
    out_json = Path(run_dir) / "stage1" / f"{name}_seed{seed}.json"
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    nh = cfg_s["nhp"]
    t0 = time.time()
    res, calib_rates = build_reservoir(cfg_s, s.D_in, seed, torch.device("cpu"), x_calib=s.x[s.split("train")])
    r = collect_features(cfg_s, res, s.x, torch.device("cpu"))              # [n, 2N]
    np.save(Path(run_dir) / "r" / f"{name}_seed{seed}.npy", r.numpy())
    rho = cfg_s["reservoir"]["rho"]
    val, test = s.split("val"), s.split("test")
    y_test = s.y[test]
    pred, lam = ridge_reference(r, s.y, s.split("train"), val, test, nh["ridge_grid"], nh["washout"])
    row = {
        "session": name, "seed": seed, "D_in": s.D_in, "n_steps": len(s.x),
        "firing_rate": firing_rate(r, res.N, slice(nh["washout"], len(r))),
        "calib_rate_tau_default": float(np.interp(math.log(cfg_s["regulators"]["defaults"]["tau"]),
                                                  np.log(res.theta_tau.numpy()), calib_rates.numpy())),
        "sec_reservoir": time.time() - t0,
        "r2_mean_baseline": r2_neurobench(s.y[s.split("train")].mean(dim=0).expand_as(y_test), y_test),
        "r2_ridge": r2_neurobench(pred, y_test), "ridge_lambda": lam,
    }
    row["k7_ok"] = bool(rho / 2 <= row["firing_rate"] <= 2 * rho)
    out_json.write_text(json.dumps(row), encoding="utf-8")
    return row


# --- Aşama 2: batch'li okuma blokları ----------------------------------------------------

class _CachedReservoir:
    def __init__(self, N: int) -> None:
        self.N = N


def _chunks(lo: int, hi: int, size: int) -> list[tuple[int, int]]:
    return [(i, min(i + size, hi)) for i in range(lo, hi, size)]


def train_batched(cfg: dict[str, Any], rs: list[np.ndarray], s: Session, configs: list[str],
                  seeds: list[int], device: torch.device) -> dict[str, dict[str, Any]]:
    """rs: seed başına [n, 2N] bellek eşlemeli r. → config başına seed'lere göre sonuçlar."""
    nh = cfg["nhp"]
    N = rs[0].shape[1] // 2
    M = len(seeds)
    groups = {c: BatchedAgent.from_agents([make_agent(cfg, c, s, _CachedReservoir(N), sd) for sd in seeds], device)
              for c in configs}
    x, y = s.x.to(device), s.y.to(device)
    tr = s.split("train")
    chunk = nh.get("chunk", 4096)
    curve = {c: [] for c in configs}
    head_curve = {c: [] for c in configs}
    train_log = {c: {k: [] for k in TRACE_KEYS} for c in configs}

    def r_block(a: int, b: int) -> torch.Tensor:                             # [M, b−a, 2N]
        return torch.from_numpy(np.stack([np.asarray(r[a:b]) for r in rs])).to(device)

    for epoch in range(nh["epochs"]):
        last = epoch == nh["epochs"] - 1
        sq = {c: torch.zeros(M, device=device) for c in configs}
        hsq = {c: torch.zeros(M, 3, device=device) for c in configs}
        for a, b in _chunks(tr.start + nh["washout"], tr.stop, chunk):
            R = r_block(a, b)
            for i, t in enumerate(range(a, b)):
                yt = y[t].expand(M, -1)
                x_next = x[t + 1].expand(M, -1) if t + 1 < tr.stop else None   # x(t+1) yalnızca train içinden
                for c, g in groups.items():
                    g.act(R[:, i])
                    o = g.learn(yt, x_next)
                    sq[c] += o["delta"].pow(2).mean(-1)
                    if "head_err" in o:
                        hsq[c] += o["head_err"].pow(2)
                        if last:
                            for k in TRACE_KEYS:
                                train_log[c][k].append(o[k].cpu())
        n = tr.stop - tr.start - nh["washout"]
        for c, g in groups.items():
            curve[c].append((sq[c] / n).cpu())
            if g.has_affect:
                head_curve[c].append((hsq[c] / n).cpu())
            if not torch.isfinite(curve[c][-1]).all():
                raise FloatingPointError(f"{s.name} config {c}: eğitim ıraksadı")
        print(f"  {s.name}: tur {epoch + 1}/{nh['epochs']} — " + "  ".join(
            f"{c}: {curve[c][-1].mean():.4f}" for c in configs), flush=True)

    out: dict[str, dict[str, Any]] = {}
    for c, g in groups.items():
        g.set_eval(True)
        preds, e_sq = [], []
        eval_log = {k: [] for k in TRACE_KEYS}
        for a, b in _chunks(tr.stop, len(s.x), chunk):
            R = r_block(a, b)
            for i, t in enumerate(range(a, b)):
                preds.append(g.act(R[:, i]).cpu())
                if g.has_cereb and t + 1 < len(s.x):
                    o = g.errors(y[t].expand(M, -1), x[t + 1].expand(M, -1))
                    e_sq.append(o["e"].pow(2).mean(-1).cpu())
                    if g.has_affect:
                        for k in TRACE_KEYS:
                            eval_log[k].append(o[k].cpu())
        P = torch.stack(preds, dim=1)                                         # [M, n_val + n_test, 2]
        n_val = s.test_start - s.val_start
        out[c] = {"val": P[:, :n_val], "test": P[:, n_val:], "curve": torch.stack(curve[c], dim=1),
                  "e_sq": torch.stack(e_sq, dim=1).mean(dim=1) if e_sq else None}
        if g.has_affect:
            out[c]["affect"] = {
                "train": {k: torch.stack(v, dim=1) for k, v in train_log[c].items()},   # [M, n, ...]
                "eval": {k: torch.stack(v, dim=1) for k, v in eval_log.items()},
                "head_curve": torch.stack(head_curve[c], dim=1),                       # [M, epochs, 3]
                "train_start": tr.start + nh["washout"], "eval_start": tr.stop,
            }
    return out


def stage2_job(args: tuple) -> list[dict[str, Any]]:
    cfg, name, seeds, run_dir, threads = args
    done = Path(run_dir) / "stage2" / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    torch.set_num_threads(threads)
    device = resolve_device(cfg["nhp"]["stage2_device"])
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    configs = cfg_s["nhp"]["configs"]
    rs = [np.load(Path(run_dir) / "r" / f"{name}_seed{sd}.npy", mmap_mode="r") for sd in seeds]
    t0 = time.time()
    outs = train_batched(cfg_s, rs, s, configs, seeds, device)
    sec = time.time() - t0
    y_test, y_val = s.y[s.split("test")], s.y[s.split("val")]
    xs = s.x[s.split("train").stop:]
    rows = []
    for m, sd in enumerate(seeds):
        row = json.loads((Path(run_dir) / "stage1" / f"{name}_seed{sd}.json").read_text(encoding="utf-8"))
        row["sec_readouts"] = sec / len(seeds)
        for c, o in outs.items():
            row[f"r2_{c}"] = r2_neurobench(o["test"][m], y_test)
            row[f"r2_{c}_val"] = r2_neurobench(o["val"][m], y_val)
            row[f"r2_{c}_xy_own"] = r2_columns(o["test"][m], y_test)
            row[f"train_mse_{c}_last"] = float(o["curve"][m, -1])
            if o["e_sq"] is not None:
                row[f"cereb_mse_{c}"] = float(o["e_sq"][m])
        if any(c.startswith("B") for c in outs):
            row["cereb_mse_repeat"] = (xs[1:] - xs[:-1]).pow(2).mean().item()
        if "B+Z" in outs and "B" in outs:                                     # Faz 4 sızıntı kontrolü
            row["leak_free"] = bool(torch.equal(outs["B+Z"]["test"][m], outs["B"]["test"][m]))
        if "A" in outs and "B" in outs:                                       # K3: beyincik kararı değiştirmez
            row["k3_A_equals_B"] = bool(torch.equal(outs["A"]["test"][m], outs["B"]["test"][m]))
        affect = {}
        for c, o in outs.items():
            if "affect" in o:
                a = o["affect"]
                affect[c] = {"train": {k: v[m] for k, v in a["train"].items()},
                             "eval": {k: v[m] for k, v in a["eval"].items()},
                             "head_curve": a["head_curve"][m],
                             "train_start": a["train_start"], "eval_start": a["eval_start"]}
        torch.save({"preds": {c: o["test"][m] for c, o in outs.items()},
                    "curves": {c: o["curve"][m].tolist() for c, o in outs.items()},
                    "affect": affect}, Path(run_dir) / f"preds_{name}_seed{sd}.pt")
        rows.append(row)
    done.write_text(json.dumps(rows), encoding="utf-8")
    print(f"[aşama 2] {name}: {sec / 60:.1f} dk", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--run", default=None, help="var olan koşu klasörü (sürdürmek için)")
    p.add_argument("--workers1", type=int, default=8)
    p.add_argument("--workers2", type=int, default=6)
    p.add_argument("--threads2", type=int, default=2)
    a = p.parse_args()

    cfg = apply_overrides(load_config(a.config), a.set)
    files = a.files or cfg["data"]["files"]
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["seeds"]
    if a.run:
        run_dir = Path(a.run)
    else:
        cfg = copy.deepcopy(cfg)
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_full"
        run_dir = RunLogger(cfg, seeds, a.config).dir
    for sub in ("r", "stage1", "stage2"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"koşu: {run_dir}", flush=True)

    jobs1 = sorted(((cfg, n, sd, str(run_dir)) for n in files for sd in seeds),
                   key=lambda j: -json.loads((Path(cfg["data"]["cache_dir"]) / "summary.json")
                                             .read_text(encoding="utf-8"))[j[1]]["n_steps"])   # uzun işler önce
    with ProcessPoolExecutor(max_workers=a.workers1) as pool:
        for row in pool.map(stage1_job, jobs1):
            print(f"[aşama 1] {row['session']} seed {row['seed']}: oran={row['firing_rate']:.4f} "
                  f"ridge={row['r2_ridge']:.4f} ({row['sec_reservoir']:.0f} s)", flush=True)

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=a.workers2) as pool:
        for rs in pool.map(stage2_job, [(cfg, n, seeds, str(run_dir), a.threads2) for n in files]):
            rows.extend(rs)
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text("".join(json.dumps({"step": i, **r}) + "\n" for i, r in enumerate(rows)), encoding="utf-8")
    summary = summarize(rows, cfg["nhp"]["configs"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"{r['session']} seed {r['seed']}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in r.items() if k.startswith("r2_") and isinstance(v, float) and "_val" not in k)
            + f"  leak_free={r.get('leak_free')}  K3={r.get('k3_A_equals_B')}  K7={'ok' if r['k7_ok'] else 'IHLAL'}")
    print(f"\nlog: {run_dir}")


if __name__ == "__main__":
    main()
