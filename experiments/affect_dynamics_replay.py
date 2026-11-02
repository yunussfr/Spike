"""Faz 5: E/M/R duygu dinamiğinin kayıtlı z izleri üzerinde çevrimdışı tekrarı (ARCHITECTURE.md §3.5).

Faz 4 tam koşusundaki z izleri (`preds_<oturum>_seed<n>.pt`, `affect["B+Z"]["train"]["z"]`, son train
turu) dinamikten geçirilir. Rezervuar çalışmaz; döngü kapalı değildir (regülatörler z'yi etkilemez).
Amaç dinamiğin kendisini ölçmek: titreme ne kadar azalıyor, E doygun kalıyor mu, R ne sıklıkta devrede?
Merkezlemeli (config) ve merkezlemesiz (z_in = z) iki sürüm karşılaştırılır.

Kullanım:
    uv run python experiments/affect_dynamics_replay.py --config configs/nhp_motor.yaml \\
        --run runs/20261030-101907_nhp_motor_full_seeds0-1-2-3-4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import numpy as np
import torch

from src.affect import AffectDynamics
from src.utils.config import apply_overrides, load_config
from src.utils.logging import RunLogger


@torch.no_grad()
def replay(cfg: dict[str, Any], z: torch.Tensor, center: bool) -> dict[str, torch.Tensor]:
    """z: [S, B, k] → E, M, R izleri: her biri [S, B, k]. B = seed'ler (aynı oturum, aynı uzunluk)."""
    dyn = AffectDynamics.from_config(cfg)
    out = {key: [] for key in ("E", "M", "R")}
    for t in range(len(z)):
        o = dyn(z[t])
        if t == 0 and not center:
            dyn.mu, dyn.center_rate = torch.zeros_like(z[0]), 0.0              # z_in = z (merkezlemesiz)
        for key in out:
            out[key].append(o[key].clone())
    return {key: torch.stack(v) for key, v in out.items()}


def metrics(z: torch.Tensor, tr: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Seed başına ([B]) ölçütler; boyutlar (k) üzerinden ortalama."""
    E, R = tr["E"], tr["R"]
    dz = (z[1:] - z[:-1]).abs().mean(dim=(0, 2))                             # [B]
    dE = (E[1:] - E[:-1]).abs().mean(dim=(0, 2))
    return {
        "dz": dz.numpy(), "dE": dE.numpy(), "jitter_ratio": (dE / dz).numpy(),
        "sat": (E.abs() > 0.95).float().mean(dim=(0, 2)).numpy(),             # E'nin doygun olduğu süre
        "sat_max_dim": (E.abs() > 0.95).float().mean(dim=0).max(dim=-1).values.numpy(),
        "E_std": E.std(dim=0).mean(dim=-1).numpy(),
        "R_active": (R.abs() > 1e-3).float().mean(dim=(0, 2)).numpy(),
    }


def plot(z: torch.Tensor, tr: dict[str, torch.Tensor], path: Path, title: str, window: int = 3000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = z.shape[-1]
    t0 = len(z) // 2
    sl = slice(t0, t0 + window)
    fig, axes = plt.subplots(k, 1, figsize=(11, 2.4 * k), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(z[sl, 0, i], lw=0.4, color="0.6", label="z (ham)")
        ax.plot(tr["E"][sl, 0, i], lw=0.9, label="E")
        ax.plot(tr["M"][sl, 0, i], lw=1.2, label="M")
        ax.plot(tr["R"][sl, 0, i], lw=0.8, label="R")
        ax.set_ylabel(f"boyut {i}")
        ax.set_ylim(-1.05, 1.05)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel(f"adım ({t0}'dan itibaren, 20 ms)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True, help="Faz 4 tam koşu klasörü (preds_*.pt)")
    p.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = "affect_dynamics_replay"
    seeds = cfg["experiment"]["seeds"]
    torch.set_num_threads(1)
    logger = RunLogger(cfg, seeds, a.config)
    run = Path(a.run)
    rows: dict[str, Any] = {}
    lines = ["| Oturum | \\|Δz\\| | \\|ΔE\\|/\\|Δz\\| | Doygun (merkezsiz) | Doygun (merkezli) | En kötü boyut (merkezli) | R aktif | E std |",
             "|---|---|---|---|---|---|---|---|"]
    for name in cfg["data"]["files"]:
        z = torch.stack([torch.load(run / f"preds_{name}_seed{sd}.pt", weights_only=False)
                         ["affect"]["B+Z"]["train"]["z"] for sd in seeds], dim=1)  # [S, B, k]
        raw = metrics(z, replay(cfg, z, center=False))
        tr = replay(cfg, z, center=True)
        cen = metrics(z, tr)
        plot(z, tr, logger.path(f"emr_{name}_seed{seeds[0]}.png"), f"{name} seed {seeds[0]} (merkezli)")
        rows[name] = {"raw": {k: v.tolist() for k, v in raw.items()},
                      "centered": {k: v.tolist() for k, v in cen.items()}}

        def ms(x: np.ndarray, pct: bool = False) -> str:
            f = 100 if pct else 1
            return f"{x.mean() * f:.{0 if pct else 3}f}{'%' if pct else ''} ± {x.std() * f:.{0 if pct else 3}f}{'%' if pct else ''}"
        lines.append(f"| {name} | {ms(cen['dz'])} | {ms(cen['jitter_ratio'])} | {ms(raw['sat'], True)} | "
                     f"{ms(cen['sat'], True)} | {ms(cen['sat_max_dim'], True)} | {ms(cen['R_active'], True)} | {ms(cen['E_std'])} |")
        print(lines[-1], flush=True)
    logger.save_json("replay_summary.json", rows)
    (logger.dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlog: {logger.dir}")


if __name__ == "__main__":
    main()
