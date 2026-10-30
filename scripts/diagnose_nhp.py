"""Faz 3 teşhis tablosu: hangi temsil hız bilgisini taşıyor? (yalnızca train/val; test kullanılmaz)

Her satır için ridge (λ val'de seçilir) ile val R² ve ateşleme oranı raporlanır:
  raw          — o anki 20 ms sayımları x(t)            (kodlayıcı/rezervuar yok)
  hist_H       — son H adımın sayımları [x(t−H+1..t)]   (doğrusal geçmiş referansı)
  res:<ayar>   — rezervuar r = [f, v], verilen config geçersiz kılmalarıyla

Kullanım:
    uv run python scripts/diagnose_nhp.py --config configs/nhp_motor.yaml --file indy_20170131_02 \
        --variants "base" "reservoir.calibration.source=train" "reservoir.c_in=10"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import build_reservoir, collect_features, firing_rate, r2_neurobench  # noqa: E402
from src.data.primate import load_session, prepare  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.metrics import ridge_fit, ridge_predict  # noqa: E402


def val_r2(X: torch.Tensor, y: torch.Tensor, tr: slice, va: slice, grid: list[float]) -> tuple[float, float]:
    mu, sd = X[tr].double().mean(0), X[tr].double().std(0).clamp(min=1e-8)
    Z = lambda s: (X[s].double() - mu) / sd  # noqa: E731
    best = (-1e9, None)
    for lam in grid:
        p = ridge_predict(Z(va), ridge_fit(Z(tr), y[tr].double(), lam))
        best = max(best, (r2_neurobench(p, y[va]), lam), key=lambda b: b[0])
    return best


def history(x: torch.Tensor, H: int) -> torch.Tensor:
    pads = [torch.cat([torch.zeros(h, x.shape[1]), x[: len(x) - h]]) for h in range(H)]
    return torch.cat(pads, dim=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--variants", nargs="*", default=["base"])
    p.add_argument("--hist", nargs="*", type=int, default=[5, 10])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg0 = load_config(a.config)
    seed = a.seed if a.seed is not None else cfg0["experiment"]["tune_seed"]
    device = resolve_device(cfg0["experiment"]["device"])
    grid = cfg0["nhp"]["ridge_grid"]
    w = cfg0["nhp"]["washout"]

    rows = []
    cfg, s = prepare(cfg0, load_session(cfg0, a.file))
    tr = slice(w, s.val_start)
    va = s.split("val")
    r2, lam = val_r2(s.x, s.y, tr, va, grid)
    rows.append({"variant": "raw", "val_r2": r2, "lambda": lam})
    for H in a.hist:
        r2, lam = val_r2(history(s.x, H), s.y, tr, va, grid)
        rows.append({"variant": f"hist_{H}", "val_r2": r2, "lambda": lam})
    for row in rows:
        print(json.dumps(row), flush=True)

    for v in a.variants:
        sets = [] if v == "base" else v.split(";")
        c = apply_overrides(cfg0, sets)
        c, s = prepare(c, load_session(c, a.file))
        t0 = time.time()
        res, _ = build_reservoir(c, s.D_in, seed, device, x_calib=s.x[s.split("train")])
        r = collect_features(c, res, s.x, device)
        r2, lam = val_r2(r, s.y, tr, va, grid)
        r2f, _ = val_r2(r[:, : res.N], s.y, tr, va, grid)
        row = {"variant": f"res:{v}", "val_r2": r2, "val_r2_f_only": r2f, "lambda": lam,
               "rate": firing_rate(r, res.N, slice(w, len(r))), "sec": time.time() - t0}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
