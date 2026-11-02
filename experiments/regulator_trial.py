"""Faz 5 deneme koşusu: regülatörler gerçek veride çalışıyor mu? (yalnızca ayar seed'i, yalnızca val)

Çıkış kriteri 1'e bakar: g, τ, η, β'nın zaman izleri sabit kalmıyor ve sınırlara yapışmıyor mu?
C'de r öğrenmeye bağlıdır (g, τ z_{t−1}'den gelir), bu yüzden r önceden üretilemez: her ajan kendi
rezervuar kopyasıyla adım adım akar (K1: kopyalar aynı sabit W_in, W_rec'i taşır).

Konfigürasyonlar (`regtrial.configs`): "B+Z" (karşılaştırma), "C:hand" (elle eşleme), "C:perturb" (§3.5).
"+EMR" son eki E/M/R duygu dinamiğini açar (ör. "C:perturb+EMR"; regülatörler z yerine E'yi okur).
Train bölmesinde öğrenir (`regtrial.epochs` tur, her tur başında durum sıfırlanır), val'de donuk.
Test bölmesine bakılmaz.

Kullanım:
    uv run python experiments/regulator_trial.py --config configs/nhp_motor.yaml [--files indy_20170131_02]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import math
import time
from typing import Any

import numpy as np
import torch

from experiments.common import build_reservoir, r2_neurobench
from experiments.nhp_motor import shorten
from src.affect import PARAMS
from src.agent import Agent
from src.data.primate import Session, load_session, prepare
from src.utils.config import apply_overrides, load_config, resolve_device
from src.utils.logging import RunLogger


def make_agent(cfg: dict[str, Any], c: str, s: Session, res: Any, seed: int) -> Agent:
    cc = copy.deepcopy(cfg)
    cc["affect"]["dynamics"]["enabled"] = c.endswith("+EMR")
    c = c.removesuffix("+EMR")
    if c.startswith("C:"):
        cc["agent"]["config"] = "C"
        cc["affect"]["enabled"] = True
        cc["regulators"]["enabled"] = True
        cc["regulators"]["mode"] = c.split(":")[1]
    else:
        cc["agent"]["config"] = c.split("+")[0]
        cc["affect"]["enabled"] = c.endswith("+Z")
    return Agent(cc, D_in=s.D_in, D_out=s.y.shape[1], seed=seed, reservoir=copy.deepcopy(res))


def _values(ag: Agent) -> list[float]:
    return [float(torch.as_tensor(getattr(ag, p)).mean()) for p in PARAMS]


def trace_stats(tr: np.ndarray, bounds: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """tr: [n, 4] → parametre başına ortalama, std, min, maks ve sınıra yapışma oranı (aralığın %1'i)."""
    out = {}
    for i, p in enumerate(PARAMS):
        lo, hi = bounds[p]
        v = np.log(tr[:, i]) if p == "tau" else tr[:, i]
        a, b = (math.log(lo), math.log(hi)) if p == "tau" else (lo, hi)
        pos = (v - a) / (b - a)                                               # [n] ∈ [0, 1]
        out[p] = {"mean": float(tr[:, i].mean()), "std": float(tr[:, i].std()),
                  "min": float(tr[:, i].min()), "max": float(tr[:, i].max()),
                  "pos_std": float(pos.std()), "frac_at_bound": float(((pos < 0.01) | (pos > 0.99)).mean())}
    return out


@torch.no_grad()
def run_session(cfg: dict[str, Any], name: str, seed: int, device: torch.device) -> dict[str, Any]:
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    rt = cfg_s["regtrial"]
    if rt["max_steps"]:
        s = shorten(s, rt["max_steps"])
    res, _ = build_reservoir(cfg_s, s.D_in, seed, device, x_calib=s.x[s.split("train")])
    N = res.N
    agents = {c: make_agent(cfg_s, c, s, res, seed) for c in rt["configs"]}
    tr, val = s.split("train"), s.split("val")
    washout = cfg_s["nhp"]["washout"]
    x, y = s.x.to(device), s.y.to(device)
    out: dict[str, Any] = {"session": name, "seed": seed, "n_train": tr.stop - tr.start}
    for c, ag in agents.items():
        t0 = time.time()
        trace, rates, trials, E_tr = [], [], [], []
        for epoch in range(rt["epochs"]):
            ag.reset_state()                                                  # her tur yeni dizi (K6)
            for t in range(tr.start, tr.stop):
                r = ag.perceive(x[t: t + 1])
                ag.act(r)
                if t - tr.start < washout:
                    continue                                                  # ısınma: öğrenme yok
                o = ag.learn(y[t: t + 1], x[t + 1: t + 2] if t + 1 < tr.stop else None)
                if epoch == rt["epochs"] - 1:
                    trace.append(_values(ag))
                    rates.append(float(r[0, :N].mean()))
                    if "E" in o:
                        E_tr.append(o["E"][0])
                if "trial" in o:
                    trials.append(o["trial"])
        ag.set_eval(True)                                                     # val: donuk, durum train'den sürer
        preds, val_trace = [], []
        for t in range(val.start, val.stop):
            preds.append(ag.act(ag.perceive(x[t: t + 1])))
            if ag.cerebellum is not None and t + 1 < len(x):
                ag.errors(y[t: t + 1], x[t + 1: t + 2])                       # z güncellensin (regülatör girdisi)
            val_trace.append(_values(ag))
        P = torch.cat(preds)
        tr_np = np.array(trace)
        out[c] = {
            "r2_val": r2_neurobench(P, y[val]),
            "firing_rate": float(np.mean(rates)),
            "stats_train": trace_stats(tr_np, cfg_s["regulators"]["bounds"]),
            "n_trials": len(trials), "n_accepted": int(sum(t["accepted"] for t in trials)),
            "sec": time.time() - t0,
            "trace_train": tr_np, "trace_val": np.array(val_trace),
        }
        if E_tr:
            E = torch.stack(E_tr)                                             # [n, k]
            out[c]["E_sat"] = float((E.abs() > 0.95).float().mean())
            out[c]["E_jitter"] = float((E[1:] - E[:-1]).abs().mean())
            print(f"    E: doygun={out[c]['E_sat']:.1%}  |ΔE|={out[c]['E_jitter']:.3f}", flush=True)
        rho = cfg_s["reservoir"]["rho"]
        out[c]["k7_ok"] = bool(rho / 2 <= out[c]["firing_rate"] <= 2 * rho)
        print(f"{name} {c}: R²(val)={out[c]['r2_val']:.4f}  oran={out[c]['firing_rate']:.4f}  "
              f"deneme={len(trials)} kabul={out[c]['n_accepted']}  ({out[c]['sec']:.0f} s)", flush=True)
        for p, st in out[c]["stats_train"].items():
            print(f"    {p:>4}: ort={st['mean']:.4g} std={st['std']:.3g} [{st['min']:.4g}, {st['max']:.4g}]"
                  f"  konum std={st['pos_std']:.3f}  sınırda={st['frac_at_bound']:.1%}", flush=True)
    return out


def plot(out: dict[str, Any], path: Path, configs: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(10, 8), sharex=True)
    for c in configs:
        tr = np.concatenate([out[c]["trace_train"], out[c]["trace_val"]])
        for i, p in enumerate(PARAMS):
            axes[i].plot(tr[:, i], lw=0.6, label=c)
            axes[i].set_ylabel(p)
    n_tr = len(out[configs[0]]["trace_train"])
    for ax in axes:
        ax.axvline(n_tr, color="k", ls="--", lw=0.8)                          # train | val
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("adım (son train turu | val)")
    fig.suptitle(f"{out['session']} seed {out['seed']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    p.add_argument("--files", nargs="*", default=None)
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_regtrial"
    seed = cfg["experiment"]["tune_seed"]                                     # rapor seed'leri değil
    torch.set_num_threads(1)
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seed, a.config)
    configs = cfg["regtrial"]["configs"]
    for i, name in enumerate(a.files or cfg["data"]["files"][:1]):
        out = run_session(cfg, name, seed, device)
        plot(out, logger.path(f"regulators_{name}.png"), configs)
        torch.save({c: {k: out[c][k] for k in ("trace_train", "trace_val")} for c in configs},
                   logger.path(f"traces_{name}.pt"))
        logger.log(i, session=name, **{f"{c}/{k}": v for c in configs for k, v in out[c].items()
                                        if not k.startswith("trace")})
    print(f"\nlog: {logger.dir}")


if __name__ == "__main__":
    main()
