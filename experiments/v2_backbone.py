"""Faz V2: üç omurga modunu tek oturumda, tek başlıklı okumayla karşılaştırır (ayar seed'leri, val).

Her (oturum, mod, seed) için model train pencerelerinde eğitilir, en iyi epoch val R²'si ile seçilir.
Mod seçimi yalnızca val'e göre yapılır. Test R² yalnızca `--report-test` verilince, seçim bittikten sonra hesaplanır.
Karşılaştırma: 4 ms GRU tavanının val R²'si (KR-037 koşusu, `v2_backbone.ceiling_run`).

Kullanım:
    uv run python experiments/v2_backbone.py --config configs/v2_backbone.yaml [--files ...] [--modes ...] [--seeds ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v2.metrics import r2_neurobench  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.v2.data import load_v2_session  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.train import evaluate, prepare_session, train_model  # noqa: E402


def ceiling_val(run: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if m.get("kind") == "gru":
            out.setdefault(m["session"], []).append(m["r2_val"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="duman koşusu için")
    p.add_argument("--report-test", action="store_true")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    vb = cfg["v2_backbone"]
    files = a.files or vb["files"]
    modes = a.modes or vb["modes"]
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seeds, a.config)
    ceil = ceiling_val(Path(vb["ceiling_run"]))

    results: dict[str, dict[str, list[dict]]] = {}
    step = 0
    for name in files:
        prep = prepare_session(cfg, load_v2_session(cfg, name), device)
        for mode in modes:
            for seed in seeds:
                set_seed(seed)
                model = V2Model(cfg, prep.session.D_in, prep.y.shape[1], mode).to(device)

                def log(row: dict, name=name, mode=mode, seed=seed) -> None:
                    nonlocal step
                    logger.log(step, session=name, mode=mode, seed=seed, kind="epoch", **row)
                    step += 1
                    print(json.dumps({"session": name, "mode": mode, "seed": seed, **{k: round(v, 4) for k, v in row.items()}}),
                          flush=True)

                res = train_model(model, prep, cfg, seed, r2_neurobench, a.epochs, log)
                row = {"best_val": res.best_val, "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                       "sec_per_epoch": res.sec_per_epoch,
                       **{f"best_{k}": v for k, v in res.history[res.best_epoch].items() if k.startswith("val_rate")}}
                if a.report_test:
                    row["test_r2"] = evaluate(model, prep, "test", cfg, r2_neurobench)["r2"]
                torch.save(res.state, logger.path(f"model_{name}_{mode}_seed{seed}.pt"))
                results.setdefault(name, {}).setdefault(mode, []).append({"seed": seed, **row})
                logger.log(step, session=name, mode=mode, seed=seed, kind="final", **row)
                step += 1
                logger.save_json("results.json", results)

    h = cfg["backbone"]["homeostasis"]
    lines = ["| Oturum | Mod | val R² | en iyi epoch | oran hızlı / yavaş (val) | s/epoch | GRU tavanı val |",
             "|---|---|---|---|---|---|---|"]
    for name, by_mode in results.items():
        cv = ceil.get(name, [])
        cstr = f"{np.mean(cv):.3f}" if cv else "—"
        for mode, rows in by_mode.items():
            v = [r["best_val"] for r in rows]
            rf = np.mean([r["best_val_rate_fast"] for r in rows])
            rs = np.mean([r["best_val_rate_slow"] for r in rows])
            ok = "" if mode == "tau_gru" or (h["f_min"] <= rf <= h["f_max"] and h["f_min"] <= rs <= h["f_max"]) else " ✗"
            lines.append(f"| {name} | {mode} | {np.mean(v):.3f} ± {np.std(v):.3f} | "
                         f"{','.join(str(r['best_epoch']) for r in rows)} | {rf:.3f} / {rs:.3f}{ok} | "
                         f"{np.mean([r['sec_per_epoch'] for r in rows]):.1f} | {cstr} |")
    lines += ["", f"Seed {seeds}, val R² (NeuroBench), tek başlık τ_read = {cfg['readout']['taus_ms']} ms. "
              f"✗: alt-popülasyon oranı [{h['f_min']}, {h['f_max']}] dışında (KV7)."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
