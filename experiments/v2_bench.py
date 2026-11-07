"""Hesap bütçesi (KR-036): gerçek `src/v2` omurgasıyla bir eğitim batch'inin süresi → epoch ve protokol süresi.

Eğitim adımı `src/v2/train.TrainStep` ile ölçülür (config'teki `train.cuda_graph` ayarıyla, eğitimdekiyle aynı yol).
Pencere sayıları gerçek önbelleklerden (train dilimi) hesaplanır; değerlendirme (val akışı) süresi dahil değildir.

Kullanım:
    uv run python experiments/v2_bench.py --config configs/v2_base.yaml
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.data import load_v2_session, ms_to_steps, train_window_starts  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.train import TrainStep  # noqa: E402


def time_batch(cfg: dict, mode: str, C: int, L: int, device: torch.device) -> dict[str, float]:
    bc, B = cfg["bench"], cfg["train"]["batch"]
    torch.manual_seed(0)
    model = V2Model(cfg, C, 2, mode).to(device)
    model.train()
    step = TrainStep(model, cfg, 10, B, L, C, 2, device)
    x = (torch.rand(B, L, C, device=device) < 0.05).float()
    y = torch.randn(B, L, 2, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    n_skip = bc["warmup"] + (cfg["train"]["graph_warmup"] + 1 if step.graph else 0)   # ısınma + yakalama ölçülmez
    for i in range(n_skip + bc["iters"]):
        t0 = time.perf_counter()
        step(x, y)                                                           # float(loss) senkronize eder
        if i >= n_skip:
            times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return {"sec_per_batch": sum(times) / len(times), "peak_gib": mem}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = "v2_bench"
    bc, B = cfg["bench"], cfg["train"]["batch"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, cfg["experiment"]["seed"], a.config)

    windows: dict[str, int] = {}
    C_of: dict[str, int] = {}
    for primate, names in cfg["primates"].items():
        for name in names:
            s = load_v2_session(cfg, name)
            windows[name] = len(train_window_starts(s, cfg))
            C_of[primate] = s.D_in

    rows = []
    for dt in bc["dt_ms"]:
        c = copy.deepcopy(cfg)
        c["time"]["dt_ms"] = dt                                              # yalnızca adım sayısı ve sızıntılar değişir
        L = ms_to_steps(cfg["windows"]["length_ms"], dt)
        for mode in bc["modes"]:
            for primate, names in cfg["primates"].items():
                r = time_batch(c, mode, C_of[primate], L, device)
                pre_b = sum(windows[n] for n in names) // B                   # drop_last
                ft_b = [windows[n] // B for n in names]
                pre_h = pre_b * r["sec_per_batch"] * bc["epochs_pretrain"] / 3600
                ft_h = [b * r["sec_per_batch"] * bc["epochs_finetune"] / 3600 for b in ft_b]
                row = {"dt_ms": dt, "mode": mode, "primate": primate, "C": C_of[primate], "L": L, **r,
                       "batches_pretrain": pre_b, "pretrain_h": pre_h, "finetune_h": ft_h, "one_seed_h": pre_h + sum(ft_h)}
                rows.append(row)
                logger.log(len(rows), **row)
                print(json.dumps(row), flush=True)

    lines = ["| Δt | mod | primat | s/batch | GiB | ön-eğitim (saat) | ince ayar, 3 oturum (saat) | 1 seed toplam (saat) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dt_ms']:.0f} ms | {r['mode']} | {r['primate']} | {r['sec_per_batch']:.3f} | {r['peak_gib']:.1f} | "
                     f"{r['pretrain_h']:.2f} | {sum(r['finetune_h']):.2f} | {r['one_seed_h']:.2f} |")
    lines += ["", f"batch {B}, N = {cfg['backbone']['N']}, ön-eğitim {bc['epochs_pretrain']} / ince ayar "
              f"{bc['epochs_finetune']} epoch, cihaz {device}, cuda_graph = {cfg['train']['cuda_graph']}. "
              f"Val değerlendirmesi dahil değil. Pencere sayıları: {windows}"]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
