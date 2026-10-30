"""Faz 3 ayarı: okuma bloğu ayarlarını yalnızca val bölmesinde seçer (CLAUDE.md kural 11).

Rezervuar ayarları sabitken r(t) ayardan bağımsızdır: her oturum için ayar seed'inde bir kez
üretilir ve önbelleğe alınır (`<cache_dir>/r_<oturum>_seed<seed>.pt`). Sonra her ayar
kombinasyonu bu r üzerinde yalnızca A ajanıyla eğitilir (B'nin ŷ'si A ile birebir aynıdır, K3).

Biyolojik gecikme d: x(t) → y(t+d). r yalnızca x'in geçmişine bağlı olduğu için gecikmeli
oturumun r'si, tam r'nin ilk len(x) − d satırıdır.

Kullanım:
    uv run python experiments/tune_nhp.py --config configs/nhp_motor.yaml --files indy_20170131_02 \
        --grid nhp.epochs=1,3 regulators.defaults.eta=0.01,0.03 data.biological_delay=0,2 --workers 8
"""

import argparse
import copy
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import build_reservoir, collect_features, r2_neurobench  # noqa: E402
from experiments.nhp_motor import train_and_predict  # noqa: E402
from src.data.primate import load_session, prepare  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402


def cached_r(cfg: dict[str, Any], name: str, seed: int) -> torch.Tensor:
    path = Path(cfg["data"]["cache_dir"]) / f"r_{name}_seed{seed}.pt"
    if path.exists():
        return torch.load(path)
    c = copy.deepcopy(cfg)
    c["data"]["biological_delay"] = 0
    c, s = prepare(c, load_session(c, name))
    device = resolve_device(c["experiment"]["device"])
    res, _ = build_reservoir(c, s.D_in, seed, device, x_calib=s.x[s.split("train")])
    r = collect_features(c, res, s.x, device)
    torch.save(r, path)
    return r


def _job(args: tuple) -> dict[str, Any]:
    cfg, name, seed, sets = args
    torch.set_num_threads(1)
    c = apply_overrides(cfg, sets)
    c, s = prepare(c, load_session(c, name))
    r = torch.load(Path(c["data"]["cache_dir"]) / f"r_{name}_seed{seed}.pt")[: len(s.x)]
    out = train_and_predict(c, r, s, ["A"], res=_CachedReservoir(r.shape[1] // 2), seed=seed)["A"]
    return {"session": name, "sets": sets, "val_r2": r2_neurobench(out["val"], s.y[s.split("val")])}


class _CachedReservoir:
    """Önbellekteki r kullanıldığında ajana verilen yer tutucu: ajan rezervuardan yalnızca N okur."""

    def __init__(self, N: int) -> None:
        self.N = N


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", required=True)
    p.add_argument("--grid", nargs="*", default=[], metavar="ANAHTAR=V1,V2")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    cfg = load_config(a.config)
    seed = cfg["experiment"]["tune_seed"]
    torch.set_num_threads(1)

    with ProcessPoolExecutor(max_workers=a.workers) as pool:            # önce eksik r'ler (oturum başına bir iş)
        list(pool.map(_cache_job, [(cfg, n, seed) for n in a.files]))

    keys = [g.split("=")[0] for g in a.grid]
    values = [g.split("=")[1].split(",") for g in a.grid]
    combos = [[f"{k}={v}" for k, v in zip(keys, vs)] for vs in itertools.product(*values)]
    logger = RunLogger({**cfg, "experiment": {**cfg["experiment"], "name": "nhp_tune"}}, seed, a.config)
    jobs = [(cfg, n, seed, sets) for n in a.files for sets in combos]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(_job, jobs):
            rows.append(row)
            logger.log(len(rows) - 1, **row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    # Oturumlar üzerinden ortalama val R²'ye göre sırala
    table: dict[str, list[float]] = {}
    for row in rows:
        table.setdefault(";".join(row["sets"]), []).append(row["val_r2"])
    ranked = sorted(((sum(v) / len(v), k) for k, v in table.items()), reverse=True)
    logger.save_json("ranking.json", [{"sets": k, "mean_val_r2": m} for m, k in ranked])
    print("\nSıralama (ortalama val R²):")
    for m, k in ranked:
        print(f"  {m:.4f}  {k}")
    print(f"\nlog: {logger.dir}")


def _cache_job(args: tuple) -> None:
    torch.set_num_threads(1)
    cached_r(*args)


if __name__ == "__main__":
    main()
