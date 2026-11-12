"""Faz V0 tavanı: KR-033'teki Wiener ve GRU ölçümünün Δt = `time.dt_ms` (varsayılan 4 ms) sürümü.

Aynı NeuroBench 80/10/10 bölmesi (KR-035), tahmin ve skor her 4 ms adımında, test R² (NeuroBench `R2`):
  wiener — son K kutunun (her biri `block_ms` genişliğinde) nedensel spike toplamları üzerinde ridge. Deterministik.
  gru    — ham sayımlar üzerinde BPTT ile eğitilen GRU (experiments/v2_baselines.py). 5 seed.
Bu sayılar v2'nin (PhasesV2.md V2–V3) hedef referansıdır. Tüm seçimler val'de; test yalnızca raporlanır.

Kullanım:
    uv run python experiments/v2_ceiling.py --config configs/v2_ceiling.yaml [--files ...] [--seeds 0 1] [--skip-gru]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.v2_baselines import gru  # noqa: E402
from src.v2.metrics import r2_neurobench  # noqa: E402
from src.data.primate import Session  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.data import check_time, load_v2_session, ms_to_steps  # noqa: E402


# --- Wiener: nedensel kutu toplamları, parça parça ridge ------------------------------------

def block_features(cs: torch.Tensor, rows: slice, w: int, K: int) -> torch.Tensor:
    """cs: [n+1, D] kümülatif toplam (cs[0] = 0). Satır t, b = 0..K−1 için x[t−(b+1)w+1 .. t−bw] toplamları.

    Dizi başında eksik geçmiş sıfır sayılır (nedensel). → [len(rows), K·D] float64
    """
    t = torch.arange(rows.start, rows.stop)                                  # [r]
    feats = []
    for b in range(K):
        hi = (t - b * w + 1).clamp(min=0)
        lo = (t - (b + 1) * w + 1).clamp(min=0)
        feats.append((cs[hi] - cs[lo]).double())                             # [r, D]
    return torch.cat(feats, dim=1)


def wiener(cfg: dict[str, Any], s: Session, dt: float) -> list[dict[str, Any]]:
    """Her K için λ val'de seçilir; train'de eğitilen model val ve test'te skorlanır. → K başına satırlar"""
    wc = cfg["ceiling"]["wiener"]
    w = ms_to_steps(wc["block_ms"], dt)
    ch = wc["chunk"]
    cs = torch.zeros(len(s.x) + 1, s.D_in, dtype=torch.float64)
    cs[1:] = torch.cumsum(s.x.double(), dim=0)                               # [n+1, D]; sayımlarda tam, kodlanmış girdide de çalışır
    tr, va, te = s.split("train"), s.split("val"), s.split("test")
    rows = []
    for K in wc["n_blocks"]:
        washout = K * w
        trw = slice(tr.start + washout, tr.stop)
        F = K * s.D_in

        def parts(sl: slice):
            for i in range(sl.start, sl.stop, ch):
                yield slice(i, min(i + ch, sl.stop))

        # standartlaştırma istatistikleri yalnızca train'den
        S1 = torch.zeros(F, dtype=torch.float64)
        S2 = torch.zeros(F, dtype=torch.float64)
        for p in parts(trw):
            X = block_features(cs, p, w, K)
            S1 += X.sum(0)
            S2 += (X * X).sum(0)
        n = trw.stop - trw.start
        mu = S1 / n
        sd = (S2 / n - mu * mu).clamp(min=0).sqrt().clamp(min=1e-8)

        G = torch.zeros(F + 1, F + 1, dtype=torch.float64)                   # [F+1, F+1]
        B = torch.zeros(F + 1, 2, dtype=torch.float64)                       # [F+1, 2]
        for p in parts(trw):
            Z = torch.cat([(block_features(cs, p, w, K) - mu) / sd, torch.ones(p.stop - p.start, 1, dtype=torch.float64)], 1)
            G += Z.T @ Z
            B += Z.T @ s.y[p].double()

        def predict(Wt: torch.Tensor, sl: slice) -> torch.Tensor:
            out = []
            for p in parts(sl):
                Z = torch.cat([(block_features(cs, p, w, K) - mu) / sd, torch.ones(p.stop - p.start, 1, dtype=torch.float64)], 1)
                out.append(Z @ Wt)
            return torch.cat(out).float()                                    # [len(sl), 2]

        best = None
        for lam in wc["ridge_grid"]:
            reg = lam * torch.eye(F + 1, dtype=torch.float64)
            reg[-1, -1] = 0.0                                                # bias düzenlenmez
            Wt = torch.linalg.solve(G + reg, B)
            r2v = r2_neurobench(predict(Wt, va), s.y[va])
            if best is None or r2v > best[0]:
                best = (r2v, lam, Wt)
        r2v, lam, Wt = best
        rows.append({"blocks": K, "history_ms": K * wc["block_ms"], "lambda": lam,
                     "lambda_at_edge": lam in (wc["ridge_grid"][0], wc["ridge_grid"][-1]),
                     "r2_val": r2v, "r2_test": r2_neurobench(predict(Wt, te), s.y[te])})
    return rows


# --- tablo ---------------------------------------------------------------------------------

def fmt(v: list[float]) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def write_table(logger: RunLogger, cfg: dict[str, Any], results: dict[str, Any]) -> None:
    rb = cfg["ceiling"]["wiener"]["report_blocks"]
    rep_ms = rb * cfg["ceiling"]["wiener"]["block_ms"]
    lines = [f"| Oturum | Wiener {rep_ms:.0f} ms | Wiener (val'de seçilen) | GRU | GRU gizli (seed başına) |",
             "|---|---|---|---|---|"]
    by_monkey: dict[str, dict[str, list[float]]] = {}
    for name, r in results.items():
        wr = next(w for w in r["wiener"] if w["blocks"] == rb)["r2_test"]
        ws = max(r["wiener"], key=lambda w: w["r2_val"])
        g = [x["r2_test"] for x in r["gru"]]
        hid = ",".join(str(x["hidden"]) for x in r["gru"]) or "—"
        lines.append(f"| {name} | {wr:.3f} | {ws['r2_test']:.3f} ({ws['history_ms']:.0f} ms) | "
                     f"{fmt(g) if g else '—'} | {hid} |")
        mk = by_monkey.setdefault(name.split("_")[0], {"w": [], "ws": [], "g": []})
        mk["w"].append(wr); mk["ws"].append(ws["r2_test"])
        if g:
            mk["g"].append(float(np.mean(g)))
    lines += ["", "| Maymun | Wiener sabit | Wiener seçilen | GRU |", "|---|---|---|---|"]
    for mk, d in by_monkey.items():
        gm = f"{np.mean(d['g']):.3f}" if d["g"] else "—"
        lines.append(f"| {mk} | {np.mean(d['w']):.3f} | {np.mean(d['ws']):.3f} | {gm} |")
    lines += ["", f"Test R² (NeuroBench `R2`), Δt = {cfg['time']['dt_ms']} ms, 80/10/10 bölme. "
              "GRU 5 seed ortalama ± std; Wiener deterministik."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--skip-gru", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="config geçersiz kılmaları: a.b=değer")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")                                 # Windows konsolu (cp1254) Δ, ² basamıyor
    cfg = apply_overrides(load_config(a.config), a.set)
    dt = check_time(cfg)
    files = a.files or cfg["data"]["files"]
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seeds, a.config)

    # experiments/v2_baselines.py GRU.su adım cinsinden `chunk` okur; süre → adım çevrilir.
    gcfg = copy.deepcopy(cfg)
    gcfg["ceiling"]["gru"]["chunk"] = ms_to_steps(cfg["ceiling"]["gru"]["chunk_ms"], dt)

    results: dict[str, Any] = {}
    step = 0
    for name in files:
        s = load_v2_session(cfg, name)
        t0 = time.time()
        wr = wiener(cfg, s, dt)
        print(json.dumps({"session": name, "wiener": wr, "sec": round(time.time() - t0)}), flush=True)
        for w in wr:
            logger.log(step, session=name, kind="wiener", **w)
            step += 1
        gr = []
        for seed in ([] if a.skip_gru else seeds):
            row = gru(gcfg, s, seed, device)
            gr.append({"seed": seed, **row})
            print(json.dumps({"session": name, "seed": seed, "gru": row}), flush=True)
            logger.log(step, session=name, seed=seed, kind="gru", **row)
            step += 1
        results[name] = {"wiener": wr, "gru": gr}
        logger.save_json("results.json", results)
    write_table(logger, cfg, results)
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
