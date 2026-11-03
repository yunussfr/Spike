"""Faz 6 deneme koşusu: çevrimiçi sınırlı okuma adaptasyonu ve nöromodüle plastisite (KR-031).

Protokol KR-029/030 ile aynı (ayar seed'leri, oturum `faz6.max_steps` adımla kısaltılır,
`faz6.epochs` tur train, sonra val). Test bölmesine bakılmaz.

Her (oturum, seed) işi:
  1. Rezervuar bir kez kurulur ve kalibre edilir. `faz6.train` içindeki her konfigürasyon kendi
     kopyasıyla train'de öğrenir.
  2. Val başında eğitilmiş ajan varyant başına kopyalanır (`faz6.variants`):
     - frozen: Faz 5 protokolü, W donuk.
     - fixed: W_base + ΔW, sabit η = η_base.
     - modulated: W_base + ΔW, η_t = η_base + η_mod · gate(τ_t).
     Val'de duygu uzayı ve regülatörler donuk ve z yalnızca r'yi okuyor. Bu yüzden ΔW τ izini değiştirmez
     (`test_adaptation_does_not_change_tau_trace`), kopyalar aynı r akışını görür.
  3. Prequential akış: ŷ_t skorlanır, sonra e_t = y_t − ŷ_t ile ΔW güncellenir (`decision.fast.delay`).

"+anchor" son eki (ör. "C:perturb+EMR+anchor") rezervuarda çapa alt-popülasyonunu açar (`faz6.anchor_frac`).

Kullanım:
    uv run python experiments/faz6_trial.py --config configs/nhp_motor.yaml --workers 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch

from experiments.common import build_reservoir, r2_neurobench
from experiments.nhp_motor import shorten
from experiments.regulator_trial import make_agent
from src.agent import Agent
from src.data.primate import load_session, prepare
from src.reservoir import LIFReservoir
from src.utils.config import apply_overrides, load_config
from src.utils.logging import RunLogger

DEFAULT_FILES = ["indy_20170131_02", "loco_20170215_02"]                    # KR-029/030 oturumları


def build(cfg: dict[str, Any], c: str, s: Any, res: LIFReservoir, seed: int) -> Agent:
    anchor = c.endswith("+anchor")
    ag = make_agent(cfg, c.removesuffix("+anchor"), s, res, seed)
    if anchor:                                                               # çapa maskesi kopyalara (K1: W aynı)
        frac = cfg["faz6"]["anchor_frac"]
        mask = LIFReservoir._init_anchor(res.N, frac, seed + cfg["reservoir"]["anchor_seed_offset"])
        for r in (ag.reservoir, ag.shadow.reservoir if ag.shadow is not None else None):
            if r is not None:
                r.anchor_frac = frac
                r.anchor.copy_(mask)
    return ag


@torch.no_grad()
def evaluate(ag: Agent, mode: str, cfg: dict[str, Any], x: torch.Tensor, y: torch.Tensor,
             val: slice) -> dict[str, Any]:
    """Val'de prequential değerlendirme. mode: frozen | fixed | modulated."""
    ag.fast.update(cfg["decision"]["fast"])
    ag.fast["enabled"] = mode != "frozen"
    if mode != "frozen":
        ag.fast["eta_mode"] = mode
    ag.set_eval(True)                                                        # ΔW_0 = 0 (açıksa)
    N = ag.reservoir.N
    preds, taus, etas, dws, rates = [], [], [], [], []
    w_base = float(ag.decision.W_a.norm())
    for t in range(val.start, val.stop):
        r = ag.perceive(x[t: t + 1])
        preds.append(ag.act(r))                                              # skor: ΔW_t ile, y_t görülmeden
        taus.append(float(torch.as_tensor(ag.tau).mean()))
        rates.append(float(r[0, :N].mean()))
        if ag.cerebellum is not None and t + 1 < len(x):
            ag.errors(y[t: t + 1], x[t + 1: t + 2])                          # z, E güncellenir (donuk ağırlık)
        if ag.decision.dW is not None:
            etas.append(ag.adapt(y[t: t + 1]))                               # sonra uyum
            dws.append(float(ag.decision.dW.norm()) / w_base)
    P = torch.cat(preds)
    half = len(P) // 2
    tau = np.array(taus)
    out = {"r2_val": r2_neurobench(P, y[val]),
           "r2_val_h1": r2_neurobench(P[:half], y[val][:half]),
           "r2_val_h2": r2_neurobench(P[half:], y[val][half:]),
           "tau_median": float(np.median(tau)), "tau_min": float(tau.min()), "tau_max": float(tau.max()),
           "firing_rate": float(np.mean(rates))}
    if etas:
        rho_max = ag.fast["rho_max"]
        dw = np.array(dws)
        out.update(eta_mean=float(np.mean(etas)), eta_max=float(np.max(etas)),
                   dw_rel_final=float(dw[-1]), dw_rel_max=float(dw.max()),
                   frac_at_bound=float((dw >= rho_max * (1 - 1e-4)).mean()))
    return out


@torch.no_grad()
def job(args: tuple[dict[str, Any], str, int, str]) -> dict[str, Any]:
    cfg, name, seed, run_dir = args
    done = Path(run_dir) / "jobs" / f"{name}_seed{seed}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    t0 = time.time()
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    f6 = cfg_s["faz6"]
    if f6["max_steps"]:
        s = shorten(s, f6["max_steps"])
    res, _ = build_reservoir(cfg_s, s.D_in, seed, torch.device("cpu"), x_calib=s.x[s.split("train")])
    tr, val = s.split("train"), s.split("val")
    washout = cfg_s["nhp"]["washout"]
    x, y = s.x, s.y
    row: dict[str, Any] = {"session": name, "seed": seed}
    agents = {c: build(cfg_s, c, s, res, seed) for c in f6["train"]}         # KR-030 ile aynı kurulum sırası
    for c, ag in agents.items():
        for epoch in range(f6["epochs"]):
            ag.reset_state()                                                 # her tur yeni dizi (K6)
            for t in range(tr.start, tr.stop):
                ag.act(ag.perceive(x[t: t + 1]))
                if t - tr.start < washout:
                    continue
                ag.learn(y[t: t + 1], x[t + 1: t + 2] if t + 1 < tr.stop else None)
        for v, (c_v, mode) in f6["variants"].items():
            if c_v != c:
                continue
            o = evaluate(copy.deepcopy(ag), mode, cfg_s, x, y, val)          # durum train'den sürer
            row.update({f"{v}/{k}": val_ for k, val_ in o.items()})
            print(f"{name} s{seed} {v}: R²(val)={o['r2_val']:.4f}  τ med={o['tau_median']:.0f}"
                  + (f"  η̄={o['eta_mean']:.2e}  ‖ΔW‖/‖W‖={o['dw_rel_final']:.3f}" if "eta_mean" in o else ""),
                  flush=True)
    rho = cfg_s["reservoir"]["rho"]
    row["k7_ok"] = {v: bool(rho / 2 <= row[f"{v}/firing_rate"] <= 2 * rho) for v in f6["variants"]}
    row["sec"] = time.time() - t0
    done.write_text(json.dumps(row), encoding="utf-8")
    return row


def table(rows: list[dict[str, Any]], variants: list[str]) -> str:
    def ms(v: list[float]) -> str:
        return f"{np.mean(v):+.3f} ± {np.std(v):.3f}" if v else "—"
    sessions = list(dict.fromkeys(r["session"] for r in rows))
    lines = ["## R² (val), seed başına", "", "| Oturum | Seed | " + " | ".join(variants) + " |",
             "|---|---|" + "---|" * len(variants)]
    for r in rows:
        lines.append(f"| {r['session']} | {r['seed']} | " + " | ".join(f"{r[f'{v}/r2_val']:.3f}" for v in variants) + " |")
    lines += ["", "## R² (val), ortalama ± std", "", "| Oturum | " + " | ".join(variants) + " |",
              "|---|" + "---|" * len(variants)]
    for n in sessions:
        rs = [r for r in rows if r["session"] == n]
        lines.append(f"| {n} | " + " | ".join(
            f"{np.mean([r[f'{v}/r2_val'] for r in rs]):.3f} ± {np.std([r[f'{v}/r2_val'] for r in rs]):.3f}"
            for v in variants) + " |")
    ref = variants[0]
    lines += ["", f"## Eşli fark: X − {ref} (val), ortalama ± std (kötü seed / toplam)", "",
              "| Oturum | " + " | ".join(variants[1:]) + " |", "|---|" + "---|" * (len(variants) - 1)]
    for n in sessions:
        rs = [r for r in rows if r["session"] == n]
        cells = []
        for v in variants[1:]:
            d = [r[f"{v}/r2_val"] - r[f"{ref}/r2_val"] for r in rs]
            cells.append(f"{ms(d)} ({sum(x < 0 for x in d)}/{len(d)})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Adaptasyon istatistikleri (seed ortalaması)", "",
              "| Oturum | Varyant | η̄ | ‖ΔW‖/‖W_base‖ son | sınırda | τ medyan [min, maks] | R² ilk yarı → ikinci yarı |",
              "|---|---|---|---|---|---|---|"]
    for n in sessions:
        rs = [r for r in rows if r["session"] == n]
        for v in variants:
            g = lambda k: np.mean([r[f"{v}/{k}"] for r in rs]) if f"{v}/{k}" in rs[0] else float("nan")
            lines.append(f"| {n} | {v} | {g('eta_mean'):.2e} | {g('dw_rel_final'):.3f} | {g('frac_at_bound'):.1%} | "
                         f"{g('tau_median'):.0f} [{g('tau_min'):.0f}, {g('tau_max'):.0f}] | "
                         f"{g('r2_val_h1'):.3f} → {g('r2_val_h2'):.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    p.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--seeds", nargs="*", type=int, default=[100, 101, 102])  # ayar seed'leri (rapor seed'leri değil)
    p.add_argument("--run", default=None, help="var olan koşu klasörü (sürdürmek için; config oradan okunur)")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.run:
        run_dir = Path(a.run)
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = apply_overrides(load_config(a.config), a.set)
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_faz6"
        run_dir = RunLogger(cfg, a.seeds, a.config).dir
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)
    print(f"koşu: {run_dir}", flush=True)
    jobs = [(cfg, n, sd, str(run_dir)) for n in a.files for sd in a.seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            print(f"bitti: {row['session']} seed {row['seed']} ({row['sec'] / 60:.1f} dk)", flush=True)
    variants = list(cfg["faz6"]["variants"])
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i, **r}, ensure_ascii=False) + "\n" for i, r in enumerate(rows)), encoding="utf-8")
    (run_dir / "table.md").write_text(table(rows, variants), encoding="utf-8")
    print("\n" + table(rows, variants))
    print(f"log: {run_dir}")


if __name__ == "__main__":
    main()
