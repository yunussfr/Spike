"""Faz 5 çıkış koşusu: 6 oturum × 5 rapor seed'i × {A, B, B+Z, C} (PHASES.md Faz 5, kriter 2).

İş birimi (oturum, seed); her iş ayrı süreçte, 1 iş parçacığıyla. Her iş içinde iki rezervuar akışı:
  - Sabit akış: varsayılan g, τ ile tek rezervuar + kodlayıcı. r her adımda bir kez üretilir ve
    A, B, B+Z aynı r'yi okur (KR-016 mantığı; K1: rezervuar öğrenmeden bağımsız).
  - C akışı: her C ajanı kendi rezervuar kopyasıyla adım adım akar (g, τ öğrenmeye bağlı, r önceden
    üretilemez). Eşli perturbasyonda C'nin gölge ajanı da kendi kopyasını taşır (§3.5).
r diske yazılmaz; her tur yeniden üretilir (bellek iş başına sabit, ~14 paralel iş mümkün).

Protokol: train'de `faz5.epochs` tur çevrimiçi öğrenme (her tur başında durum sıfırlanır, K6), sonra
val + test tek dizi olarak, ağırlıklar donuk. `agent.freeze_norm_eval` config'ten (KR-030: false →
girdi normları nedensel EMA ile güncellenir). Ayar yapılmaz; test yalnızca raporlanır.

Kaydedilenler (iş başına, `jobs/` altında; dosya varsa iş atlanır → `--run` ile sürdürülebilir):
  - R² val/test (NeuroBench), ateşleme oranı (sabit akış + C için τ kutularına göre, K7)
  - C: seyreltilmiş g/τ/η/β ve E izleri (her `faz5.trace_every` adımda bir), perturbasyon kabul/ret
  - leak_free: B+Z ve B'nin test tahminleri birebir aynı mı (Faz 4 sızıntı kontrolü, K3)
Özet: oturum başına ortalama ± std ve eşli C − A, C − B, C − B+Z farkları (`summary.json`, `table.md`).

Kullanım:
    uv run python experiments/faz5_full.py --config configs/nhp_motor.yaml --workers 14
    uv run python experiments/faz5_full.py --config configs/nhp_motor.yaml --run runs/<klasör>   # sürdür
    duman: --files indy_20170131_02 --seeds 0 1 --set faz5.max_steps=2000
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
from experiments.common import build_reservoir, r2_neurobench  # noqa: E402
from experiments.nhp_motor import shorten  # noqa: E402
from experiments.regulator_trial import make_agent  # noqa: E402
from src.affect import PARAMS  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.data.primate import Session, load_session, prepare  # noqa: E402
from src.encoder import SpikeEncoder  # noqa: E402
from src.reservoir import LIFReservoir  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402


class TauBins:
    """C için K7: ateşleme oranı τ kutularına göre (kutu kenarları θ(τ) ızgarası, log-ölçek)."""

    def __init__(self, grid: list[float]) -> None:
        self.edges = np.sqrt(np.array(grid[:-1]) * np.array(grid[1:]))      # [G−1] komşu ızgara noktalarının geometrik ortası
        self.grid = grid
        self.sum = np.zeros(len(grid))
        self.n = np.zeros(len(grid), dtype=np.int64)

    def add(self, tau: float, rate: float) -> None:
        i = int(np.searchsorted(self.edges, tau))
        self.sum[i] += rate
        self.n[i] += 1

    def result(self) -> dict[str, dict[str, float]]:
        return {f"{g:g}": {"rate": float(self.sum[i] / self.n[i]), "n": int(self.n[i])}
                for i, g in enumerate(self.grid) if self.n[i] > 0}


def _values(ag: Agent) -> list[float]:
    return [float(torch.as_tensor(getattr(ag, p)).mean()) for p in PARAMS]


@torch.no_grad()
def run_streams(cfg: dict[str, Any], s: Session, res: LIFReservoir, seed: int,
                configs: list[str]) -> dict[str, Any]:
    """Tüm konfigürasyonları aynı x, y akışıyla eğitir ve değerlendirir. Döndürür: config başına val/test tahmini
    ve C izleri; sabit akışın ateşleme oranı."""
    f5 = cfg["faz5"]
    washout, every = cfg["nhp"]["washout"], f5["trace_every"]
    fixed_cfgs = [c for c in configs if not c.startswith("C")]
    c_cfgs = [c for c in configs if c.startswith("C")]
    agents = {c: make_agent(cfg, c, s, res, seed) for c in configs}         # her ajan kendi rezervuar kopyası
    enc = SpikeEncoder.from_config(cfg, D_in=s.D_in)                        # sabit akış: tek kodlayıcı + rezervuar
    g0, tau0 = cfg["regulators"]["defaults"]["g"], cfg["regulators"]["defaults"]["tau"]
    N = res.N
    x, y = s.x, s.y
    tr = s.split("train")
    rate_fixed, n_rate = 0.0, 0
    bins = {c: TauBins(cfg["reservoir"]["calibration"]["tau_grid"]) for c in c_cfgs}
    trace = {c: {"train": [], "eval": [], "E_train": [], "E_eval": []} for c in c_cfgs}
    trials = {c: [0, 0] for c in c_cfgs}                                     # [deneme, kabul]
    sq = {c: [] for c in configs}                                            # tur başına train MSE

    def fixed_r(t: int) -> torch.Tensor:
        return res(enc(x[t: t + 1]), g=g0, tau=tau0)                         # [1, 2N]

    for epoch in range(f5["epochs"]):
        last = epoch == f5["epochs"] - 1
        res.reset_state()
        enc.reset_state()
        for ag in agents.values():
            ag.reset_state()                                                 # her tur yeni dizi (K6)
        err = {c: 0.0 for c in configs}
        for t in range(tr.start, tr.stop):
            r = fixed_r(t) if fixed_cfgs else None
            rc = {c: agents[c].perceive(x[t: t + 1]) for c in c_cfgs}
            if t - tr.start < washout:                                       # ısınma: öğrenme yok
                for c in c_cfgs:
                    agents[c].act(rc[c])
                continue
            yt = y[t: t + 1]
            x_next = x[t + 1: t + 2] if t + 1 < tr.stop else None             # x(t+1) yalnızca train içinden
            if last and r is not None:
                rate_fixed += float(r[0, :N].mean())
                n_rate += 1
            for c in fixed_cfgs:
                agents[c].act(r)
                err[c] += float(agents[c].learn(yt, x_next)["delta"].pow(2).mean())
            for c in c_cfgs:
                ag = agents[c]
                ag.act(rc[c])
                o = ag.learn(yt, x_next)
                err[c] += float(o["delta"].pow(2).mean())
                if "trial" in o:
                    trials[c][0] += 1
                    trials[c][1] += int(o["trial"]["accepted"])
                if last:
                    bins[c].add(float(torch.as_tensor(ag.tau).mean()), float(rc[c][0, :N].mean()))
                    if (t - tr.start) % every == 0:
                        trace[c]["train"].append(_values(ag))
                        if "E" in o:
                            trace[c]["E_train"].append(o["E"][0].tolist())
        n = tr.stop - tr.start - washout
        for c in configs:
            sq[c].append(err[c] / n)
            if not math.isfinite(sq[c][-1]):
                raise FloatingPointError(f"{s.name} seed {seed} {c}: eğitim ıraksadı")

    for ag in agents.values():
        ag.set_eval(True)                                                    # val + test: donuk, durum train'den sürer
    preds: dict[str, list[torch.Tensor]] = {c: [] for c in configs}
    for t in range(tr.stop, len(x)):
        x_next = x[t + 1: t + 2] if t + 1 < len(x) else None
        if fixed_cfgs:
            r = fixed_r(t)
            rate_fixed += float(r[0, :N].mean())
            n_rate += 1
            for c in fixed_cfgs:
                ag = agents[c]
                preds[c].append(ag.act(r))
                if ag.cerebellum is not None and x_next is not None:
                    ag.errors(y[t: t + 1], x_next)                           # duygu uzayı ve normlar (etiket yok)
        for c in c_cfgs:
            ag = agents[c]
            rc = ag.perceive(x[t: t + 1])
            preds[c].append(ag.act(rc))
            o = ag.errors(y[t: t + 1], x_next) if x_next is not None else {}
            bins[c].add(float(torch.as_tensor(ag.tau).mean()), float(rc[0, :N].mean()))
            if (t - tr.stop) % every == 0:
                trace[c]["eval"].append(_values(ag))
                if "E" in o:
                    trace[c]["E_eval"].append(o["E"][0].tolist())

    n_val = s.test_start - s.val_start
    out: dict[str, Any] = {"rate_fixed": rate_fixed / max(n_rate, 1), "configs": {}}
    for c in configs:
        P = torch.cat(preds[c])                                              # [n_val + n_test, D_out]
        out["configs"][c] = {"val": P[:n_val], "test": P[n_val:], "curve": sq[c]}
    for c in c_cfgs:
        out["configs"][c].update(
            tau_bins=bins[c].result(), n_trials=trials[c][0], n_accepted=trials[c][1],
            trace={k: np.array(v, dtype=np.float32) for k, v in trace[c].items()})
    return out


def job(args: tuple) -> dict[str, Any]:
    """Tek (oturum, seed) işi. Sonuç `jobs/<oturum>_seed<s>.json` varsa yeniden hesaplanmaz."""
    cfg, name, seed, run_dir = args
    done = Path(run_dir) / "jobs" / f"{name}_seed{seed}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    t0 = time.time()
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    if cfg_s["faz5"]["max_steps"]:
        s = shorten(s, cfg_s["faz5"]["max_steps"])
    res, _ = build_reservoir(cfg_s, s.D_in, seed, torch.device("cpu"), x_calib=s.x[s.split("train")])
    configs = cfg_s["faz5"]["configs"]
    out = run_streams(cfg_s, s, res, seed, configs)
    rho = cfg_s["reservoir"]["rho"]
    y_val, y_test = s.y[s.split("val")], s.y[s.split("test")]
    row: dict[str, Any] = {"session": name, "seed": seed, "n_steps": len(s.x), "firing_rate": out["rate_fixed"]}
    row["k7_ok"] = bool(rho / 2 <= row["firing_rate"] <= 2 * rho)
    for c, o in out["configs"].items():
        row[f"r2_{c}"] = r2_neurobench(o["test"], y_test)
        row[f"r2_{c}_val"] = r2_neurobench(o["val"], y_val)
        row[f"train_mse_{c}_last"] = o["curve"][-1]
        if c.startswith("C"):
            row[f"tau_bins_{c}"] = o["tau_bins"]
            row[f"k7_ok_{c}"] = all(rho / 2 <= b["rate"] <= 2 * rho for b in o["tau_bins"].values())
            row[f"n_trials_{c}"], row[f"n_accepted_{c}"] = o["n_trials"], o["n_accepted"]
            tr_ = o["trace"]["train"]
            if len(tr_):
                row[f"tau_median_train_{c}"] = float(np.median(tr_[:, PARAMS.index("tau")]))
                row[f"tau_median_eval_{c}"] = float(np.median(o["trace"]["eval"][:, PARAMS.index("tau")]))
    if "B" in out["configs"] and "B+Z" in out["configs"]:
        row["leak_free"] = bool(torch.equal(out["configs"]["B"]["test"], out["configs"]["B+Z"]["test"]))
    row["sec"] = time.time() - t0
    torch.save(out["configs"],
               Path(run_dir) / f"preds_{name}_seed{seed}.pt")
    done.write_text(json.dumps(row), encoding="utf-8")                      # en son: yarım iş "bitti" sayılmaz
    return row


def summarize(rows: list[dict[str, Any]], configs: list[str]) -> dict[str, Any]:
    """Oturum başına ortalama ± std (test ve val) ve eşli C − X farkları (seed başına fark, sonra ortalama)."""
    summary: dict[str, Any] = {}
    c_cfgs = [c for c in configs if c.startswith("C")]
    for name in dict.fromkeys(r["session"] for r in rows):
        rs = [r for r in rows if r["session"] == name]
        sm: dict[str, Any] = {}
        for split in ("", "_val"):
            for c in configs:
                v = [r[f"r2_{c}{split}"] for r in rs]
                sm[f"r2_{c}{split}"] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for c in c_cfgs:
                for ref in configs:
                    if ref.startswith("C"):
                        continue
                    d = [r[f"r2_{c}{split}"] - r[f"r2_{ref}{split}"] for r in rs]
                    sm[f"{c}_minus_{ref}{split}"] = {"mean": float(np.mean(d)), "std": float(np.std(d)),
                                                     "n_worse": int(sum(x < 0 for x in d)), "n": len(d)}
        sm["leak_free"] = all(r.get("leak_free", True) for r in rs)
        sm["k7_ok_fixed"] = all(r["k7_ok"] for r in rs)
        summary[name] = sm
    return summary


def table(summary: dict[str, Any], configs: list[str]) -> str:
    c_cfgs = [c for c in configs if c.startswith("C")]
    refs = [c for c in configs if not c.startswith("C")]

    def ms(d: dict[str, float]) -> str:
        return f"{d['mean']:.3f} ± {d['std']:.3f}"
    lines = ["## R² (test), ortalama ± std", "",
             "| Oturum | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for name, sm in summary.items():
        lines.append(f"| {name} | " + " | ".join(ms(sm[f"r2_{c}"]) for c in configs) + " |")
    for split, title in (("", "test"), ("_val", "val")):
        lines += ["", f"## Eşli fark ({title}): C − X, ortalama ± std (kötü seed / toplam)", "",
                  "| Oturum | " + " | ".join(f"{c} − {r}" for c in c_cfgs for r in refs) + " |",
                  "|---|" + "---|" * (len(c_cfgs) * len(refs))]
        for name, sm in summary.items():
            cells = [sm[f"{c}_minus_{r}{split}"] for c in c_cfgs for r in refs]
            lines.append(f"| {name} | " + " | ".join(f"{ms(d)} ({d['n_worse']}/{d['n']})" for d in cells) + " |")
    lines += ["", "leak_free (B = B+Z): " + ", ".join(f"{n}={sm['leak_free']}" for n, sm in summary.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--run", default=None, help="var olan koşu klasörü (sürdürmek için; config oradan okunur)")
    p.add_argument("--workers", type=int, default=14)
    a = p.parse_args()

    if a.run:
        run_dir = Path(a.run)
        cfg = load_config(run_dir / "config.yaml")                          # sürdürürken config değişmez (K8)
    else:
        cfg = apply_overrides(load_config(a.config), a.set)
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_faz5"
    files = a.files or cfg["data"]["files"]
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["seeds"]
    if not a.run:
        run_dir = RunLogger(cfg, seeds, a.config).dir
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)
    print(f"koşu: {run_dir}", flush=True)

    sizes = json.loads((Path(cfg["data"]["cache_dir"]) / "summary.json").read_text(encoding="utf-8"))
    jobs = sorted(((cfg, n, sd, str(run_dir)) for n in files for sd in seeds),
                  key=lambda j: -sizes[j[1]]["n_steps"])                     # uzun işler önce
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            print(f"{row['session']} seed {row['seed']}: " + "  ".join(
                f"{k}={v:.4f}" for k, v in row.items() if k.startswith("r2_") and "_val" not in k)
                + f"  ({row['sec'] / 60:.1f} dk)", flush=True)
    rows.sort(key=lambda r: (files.index(r["session"]), r["seed"]))
    configs = cfg["faz5"]["configs"]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i, **r}) + "\n" for i, r in enumerate(rows)), encoding="utf-8")
    summary = summarize(rows, configs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "table.md").write_text(table(summary, configs), encoding="utf-8")
    print("\n" + table(summary, configs))
    print(f"log: {run_dir}")


if __name__ == "__main__":
    main()
