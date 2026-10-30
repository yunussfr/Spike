"""Faz 4 ayarı: duygu uzayı öğrenme hızlarını yalnızca val bölmesinde seçer (önbellekteki r ile).

Ölçütler (val): z doygunluğu (|z| > 0,99 oranı), iyi/kötü AUC, baş hatası düşüşü (train ilk/son %10).
Kullanım:
    uv run python scripts/tune_affect.py --config configs/nhp_motor.yaml --files indy_20170131_02 \
        --grid affect.lr_z=0.01,0.1 affect.lr_head=0.1 --workers 6
"""

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.affect_viz import analyse  # noqa: E402
from experiments.nhp_motor import train_and_predict  # noqa: E402
from experiments.tune_nhp import _CachedReservoir, cached_r  # noqa: E402
from src.data.primate import load_session, prepare  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402


def _job(args: tuple) -> dict:
    cfg, name, seed, sets = args
    torch.set_num_threads(1)
    c = apply_overrides(cfg, sets)
    c, s = prepare(c, load_session(c, name))
    r = cached_r(c, name, seed)[: len(s.x)]
    a = train_and_predict(c, r, s, ["B+Z"], res=_CachedReservoir(r.shape[1] // 2), seed=seed)["B+Z"]["affect"]
    n_val = s.test_start - s.val_start                                       # yalnızca val (test'e bakılmaz)
    a["eval"] = {k: v[:n_val] for k, v in a["eval"].items()}
    out = analyse(a)
    out["z_saturated"] = float((a["eval"]["z"].abs() > 0.99).float().mean())
    return {"session": name, "sets": sets, **out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", required=True)
    p.add_argument("--grid", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    cfg = load_config(a.config)
    seed = cfg["experiment"]["tune_seed"]
    keys = [g.split("=")[0] for g in a.grid]
    values = [g.split("=", 1)[1].split("|" if "|" in g else ",") for g in a.grid]   # liste değerleri için ayraç "|"
    combos = [[f"{k}={v}" for k, v in zip(keys, vs)] for vs in itertools.product(*values)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(_job, [(cfg, n, seed, c) for n in a.files for c in combos]):
            keep = {k: row[k] for k in ("session", "sets", "auc_good_bad_from_z", "z_saturated", "z_std")}
            keep.update({k: round(v, 3) for k, v in row.items() if k.startswith("head_mse") and "epoch" in k})
            keep.update({k: v for k, v in row.items() if k.startswith("head_declines")})
            print(json.dumps(keep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
