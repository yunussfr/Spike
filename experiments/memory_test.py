"""Faz 1 bellek testi.

Rastgele bir girdi dizisi rezervuara verilir. Her τ için rezervuar çıktısından iki tür
hedef ridge regresyonla geri çıkarılır:

  item   — d adım önceki tek değer x(t−d). Bağımsız rastgele girdide tek bir öğenin sızıntılı
           bir toplamdaki payı τ büyüdükçe küçülür; bu hedef τ ile iyileşmek zorunda değildir.
  window — son d adımın ortalaması mean(x(t−d+1), …, x(t)). Uzun τ'nun sağlaması beklenen
           şey budur: daha uzun pencerelerin ortalaması daha büyük τ ile çıkarılabilmeli.

Okumalar: yalnızca f, r = [f, v] ve ideal referans `lin` (aynı kodlanmış girdi, aynı W_in ve
τ_i'lerle spike'sız doğrusal filtre bankası; kodlayıcının ve zaman sabitlerinin izin verdiği
üst sınır). Çıkış kriteri (PHASES.md Faz 1) window hedefinde [f, v] ile ölçülür.

Tüm τ değerleri aynı rezervuarda paralel koşar (batch boyutu = τ sayısı, aynı girdi).
Öznitelikler eğitim istatistikleriyle standartlaştırılır, λ doğrulama setinde seçilir,
R² test setinde raporlanır.

Kullanım:
    uv run python experiments/memory_test.py --config configs/memory.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from src.utils.config import apply_overrides, load_config, resolve_device
from src.utils.logging import RunLogger
from src.utils.metrics import r2_per_column, ridge_fit
from src.utils.seeding import set_seed

TARGET_LAGS = {"item": "delays", "window": "windows"}       # hedef → config'teki gecikme listesi


@torch.no_grad()
def collect_states(
    cfg: dict[str, Any], seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Döndürür: x [S, D_in], r [S, B, 2N], lin [S, B, N], ateşleme oranı [B], θ(τ) [B].

    B = τ sayısı. lin: ideal referans — aynı u, W_in, τ_i ve c_i ile spike'sız, rekürrentsiz
    doğrusal filtre bankası (T iç adımın kapalı biçimi).
    """
    mt = cfg["memory_test"]
    set_seed(seed)
    taus = torch.tensor(mt["taus"], device=device)                           # [B]
    B, D_in = len(taus), mt["D_in"]
    lo, hi = cfg["encoder"]["level"]["range"]

    reservoir = LIFReservoir.from_config(cfg, D_in=D_in, seed=seed, calibrate=False).to(device)
    reservoir.calibrate_from_config(cfg, D_in, seed)
    encoder = SpikeEncoder.from_config(cfg, D_in=D_in)
    g = cfg["regulators"]["defaults"]["g"]
    x = lo + (hi - lo) * torch.rand(mt["n_steps"], D_in, device=device)     # [S, D_in]

    T, dt = reservoir.T, reservoir.dt
    alpha_i = torch.exp(-dt / (taus.unsqueeze(1) * reservoir.tau_mult))     # [B, N]
    c = (1 - alpha_i) / (1 - torch.exp(-dt / taus)).unsqueeze(1)             # [B, N]
    decay = alpha_i ** T                                                     # [B, N]
    gain = c * (1 - decay) / (1 - alpha_i)                                   # [B, N] T adım sabit girdi
    v_lin = torch.zeros(B, reservoir.N, device=device)

    S, N = mt["n_steps"], reservoir.N
    r = torch.empty(S, B, 2 * N)                                             # [S, B, 2N] (CPU, önceden ayrılır)
    lin = torch.empty(S, B, N)                                               # [S, B, N]
    for t in range(S):
        u = encoder(x[t].expand(B, D_in))                                    # [B, T, C]
        r[t] = reservoir(u, g=g, tau=taus).cpu()                             # [B, 2N]
        v_lin = decay * v_lin + gain * (g * (u[:, 0] @ reservoir.W_in.T))    # [B, N]
        lin[t] = v_lin.cpu()
    rate = r[mt["washout"]:, :, :N].mean(dim=(0, 2))                         # [B]
    return x.cpu(), r, lin, rate, reservoir.theta(taus).tolist()


def make_targets(x: torch.Tensor, target: str, lags: list[int]):
    """Satır indislerinden hedef matrisi üreten fonksiyon döndürür: rows [n] → [n, L·D_in]."""
    x64 = x.double()                                                         # [S, D]
    if target == "item":
        return lambda rows: torch.cat([x64[rows - d] for d in lags], dim=1)
    if target == "window":
        cs = torch.cat([torch.zeros(1, x64.shape[1], dtype=x64.dtype), x64.cumsum(dim=0)])  # [S+1, D]
        return lambda rows: torch.cat([(cs[rows + 1] - cs[rows + 1 - d]) / d for d in lags], dim=1)
    raise ValueError(f"bilinmeyen hedef: {target}")


def memory_curve(
    cfg: dict[str, Any], x: torch.Tensor, features: torch.Tensor, target: str
) -> tuple[np.ndarray, np.ndarray]:
    """features: [S, B, F]. Döndürür (test R², doğrulama R²): her biri [B, L].

    R² girdi boyutları üzerinden ortalamadır. Doğrulama R²'si seçilen λ'nınkidir ve
    hiperparametre seçiminde (ör. γ_rec) test yerine bu kullanılır.

    Aynı satırlar tüm gecikmeler için kullanılır; hedefler [n, L·D_in] olarak birlikte çözülür.
    λ her (τ, gecikme) için doğrulama setinde ayrı seçilir.
    """
    mt = cfg["memory_test"]
    lags = mt[TARGET_LAGS[target]]
    L, D = len(lags), x.shape[1]
    max_lag = max(max(mt["delays"]), max(mt["windows"]))    # iki hedef aynı satırları kullanır
    start = mt["washout"] + max_lag
    S = len(x)
    n = S - start
    n_tr, n_va = int(n * mt["split"][0]), int(n * mt["split"][1])
    idx = torch.arange(start, S)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    trva = torch.cat([tr, va])
    targets = make_targets(x, target, lags)

    def per_lag(r2: torch.Tensor) -> torch.Tensor:                          # [L·D] → [L]
        return r2.reshape(L, D).mean(dim=1)

    out = np.zeros((features.shape[1], L))
    out_val = np.zeros((features.shape[1], L))
    for b in range(features.shape[1]):
        F = features[:, b].double()                                          # [S, F]
        mu, sd = F[tr].mean(dim=0), F[tr].std(dim=0).clamp(min=1e-8)
        Fz = (F - mu) / sd
        val = torch.stack([
            per_lag(r2_per_column(Fz[va], targets(va), ridge_fit(Fz[tr], targets(tr), lam)))
            for lam in mt["ridge_grid"]
        ])                                                                   # [n_lam, L]
        best = val.argmax(dim=0)                                             # [L]
        out_val[b] = val.max(dim=0).values.numpy()

        # Seçilen λ ile eğitim+doğrulama üzerinde yeniden uydur (standartlaştırma da trva'dan)
        mu, sd = F[trva].mean(dim=0), F[trva].std(dim=0).clamp(min=1e-8)
        Fz = (F - mu) / sd
        test = torch.stack([
            per_lag(r2_per_column(Fz[te], targets(te), ridge_fit(Fz[trva], targets(trva), lam)))
            for lam in mt["ridge_grid"]
        ])                                                                   # [n_lam, L]
        out[b] = test.gather(0, best.unsqueeze(0)).squeeze(0).numpy()
    return out, out_val


def memory_horizon(curve: np.ndarray, lags: list[int], threshold: float) -> np.ndarray:
    """Her τ için R² ≥ eşik olan en büyük gecikme/pencere (hiçbiri değilse 0). [B, L] → [B]."""
    ok = curve >= threshold
    return np.array([max((d for d, o in zip(lags, row) if o), default=0) for row in ok])


def exit_criterion(summary: dict[str, Any], mt: dict[str, Any]) -> dict[str, Any]:
    """PHASES.md Faz 1 çıkış kriteri, window hedefi ve [f, v] okuması üzerinden.

    1) τ_max'ın bellek ufku ≥ min_ratio × τ_min'in ufku (5 seed ortalaması).
    2) Uzun pencerelerde (d ≥ long_from) en iyi τ > kısa pencerelerde (d ≤ short_to) en iyi τ;
       en iyi τ, ilgili pencerelerin ortalama R²'sine göre seçilir.
    """
    ec_cfg = mt["exit_criterion"]
    s = summary["window"]["fv"]
    taus, windows = mt["taus"], mt["windows"]
    r2 = np.array(s["r2_mean"])                                              # [B, L]
    h = s["horizon_mean"]
    i_min, i_max = int(np.argmin(taus)), int(np.argmax(taus))
    ratio = h[i_max] / max(h[i_min], 1e-9)
    long_cols = [j for j, d in enumerate(windows) if d >= ec_cfg["long_from"]]
    short_cols = [j for j, d in enumerate(windows) if d <= ec_cfg["short_to"]]
    best_long = taus[int(r2[:, long_cols].mean(axis=1).argmax())]
    best_short = taus[int(r2[:, short_cols].mean(axis=1).argmax())]
    ratio_ok = bool(ratio >= ec_cfg["min_ratio"])
    best_ok = bool(best_long > best_short)
    return {"horizon_ratio": round(float(ratio), 3), "min_ratio": ec_cfg["min_ratio"],
            "ratio_ok": ratio_ok, "best_tau_short": best_short, "best_tau_long": best_long,
            "best_tau_ok": best_ok, "passed": ratio_ok and best_ok}


def plot(curves: dict[tuple[str, str], np.ndarray], cfg: dict[str, Any], path: str) -> None:
    """curves[(hedef, okuma)]: [n_seeds, B, L]. Satırlar hedef, sütunlar okuma + kapasite."""
    mt = cfg["memory_test"]
    taus = mt["taus"]
    titles = {"f": "yalnızca f", "fv": "r = [f, v]", "lin": "ideal doğrusal referans"}
    ylabels = {"item": "test R²  [x(t−d)]", "window": "test R²  [son d adım ortalaması]"}
    xlabels = {"item": "gecikme d (ortam adımı)", "window": "pencere uzunluğu d (ortam adımı)"}
    targets, readouts = mt["targets"], mt["readouts"]
    fig, axes = plt.subplots(len(targets), len(readouts) + 1,
                             figsize=(5.3 * (len(readouts) + 1), 4.3 * len(targets)), squeeze=False)
    for i, target in enumerate(targets):
        lags = mt[TARGET_LAGS[target]]
        for j, name in enumerate(readouts):
            ax = axes[i, j]
            c = curves[(target, name)]
            mean, std = c.mean(axis=0), c.std(axis=0)                        # [B, L]
            for b, tau in enumerate(taus):
                ax.errorbar(lags, mean[b], yerr=std[b], marker="o", capsize=3, label=f"τ = {tau:g} ms")
            ax.axhline(mt["r2_threshold"], color="gray", ls="--", lw=1)
            ax.set(xscale="log", xlabel=xlabels[target], ylabel=ylabels[target],
                   title=f"{target} — {titles[name]}", ylim=(-0.05, 1.0))
            ax.legend()
        ax = axes[i, -1]
        for name in readouts:
            h = np.stack([memory_horizon(s, lags, mt["r2_threshold"]) for s in curves[(target, name)]])
            ax.errorbar(taus, h.mean(axis=0), yerr=h.std(axis=0), marker="o", capsize=3, label=name)
        ax.set(xscale="log", yscale="symlog", xlabel="τ (ms)",
               ylabel=f"R² ≥ {mt['r2_threshold']} olan en büyük d", title=f"{target} — bellek ufku")
        ax.legend()
    n_seeds = next(iter(curves.values())).shape[0]
    fig.suptitle(f"Faz 1 bellek testi — ortalama ± std, {n_seeds} seed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


Curves = dict[tuple[str, str], list[np.ndarray]]


def load_run(run_dir: Path, keys: list[tuple[str, str]], mt: dict[str, Any]):
    """Bir koşunun metrics.jsonl'ünden seed başına eğrileri geri kurar."""
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    seeds = list(dict.fromkeys(rec["step"] for rec in records))
    curves: Curves = {k: [] for k in keys}
    val_curves: Curves = {k: [] for k in keys}
    rates, thetas = [], []
    for seed in seeds:
        recs = [rec for rec in records if rec["step"] == seed]
        for target, name in keys:
            lags = mt[TARGET_LAGS[target]]
            rows = {rec["tau"]: rec for rec in recs if rec["target"] == target and rec["readout"] == name}
            curves[(target, name)].append(np.array([[rows[t][f"r2_d{d}"] for d in lags] for t in mt["taus"]]))
            val_curves[(target, name)].append(
                np.array([[rows[t][f"val_r2_d{d}"] for d in lags] for t in mt["taus"]]))
        first = {rec["tau"]: rec for rec in recs if rec["target"] == keys[0][0] and rec["readout"] == keys[0][1]}
        rates.append(np.array([first[t]["firing_rate"] for t in mt["taus"]]))
        thetas.append([first[t]["theta"] for t in mt["taus"]])
    return seeds, curves, val_curves, rates, thetas


def report(
    cfg: dict[str, Any], keys: list[tuple[str, str]], curves: Curves, val_curves: Curves,
    rates: list[np.ndarray], thetas: list[list[float]], logger: RunLogger, extra: dict[str, Any],
) -> None:
    """Seed'ler üzerinden özet, grafik ve konsol tablosu."""
    mt = cfg["memory_test"]
    arrays = {k: np.stack(c) for k, c in curves.items()}                    # [n_seeds, B, L]
    summary: dict[str, Any] = {
        **extra,
        "taus": mt["taus"],
        "delays": mt["delays"],
        "windows": mt["windows"],
        "firing_rate_mean": np.stack(rates).mean(axis=0).round(5).tolist(),
        "theta_mean": np.array(thetas).mean(axis=0).round(4).tolist(),
    }
    for (target, name), c in arrays.items():
        lags = mt[TARGET_LAGS[target]]
        horizons = np.stack([memory_horizon(s, lags, mt["r2_threshold"]) for s in c])  # [n_seeds, B]
        val = np.stack(val_curves[(target, name)])
        summary.setdefault(target, {})[name] = {
            "r2_mean": c.mean(axis=0).round(4).tolist(),
            "r2_std": c.std(axis=0).round(4).tolist(),
            "capacity_mean": c.clip(min=0).sum(axis=2).mean(axis=0).round(4).tolist(),
            "capacity_std": c.clip(min=0).sum(axis=2).std(axis=0).round(4).tolist(),
            "horizon_per_seed": horizons.tolist(),
            "horizon_mean": horizons.mean(axis=0).round(2).tolist(),
            "best_tau_per_lag": [mt["taus"][i] for i in c.mean(axis=0).argmax(axis=0)],
            "val_capacity_mean": val.clip(min=0).sum(axis=2).mean(axis=0).round(4).tolist(),
        }
    summary["exit_criterion"] = exit_criterion(summary, mt)
    logger.save_json("summary.json", summary)
    plot(arrays, cfg, str(logger.path("memory.png")))

    # Konsol çıktısı ASCII: Windows konsolu (cp1254) Yunan harflerini yazamıyor
    for target, name in keys:
        s = summary[target][name]
        lags = mt[TARGET_LAGS[target]]
        print(f"\n[{target} / {name}]  tau ms | " + " | ".join(f"d={d:>3}" for d in lags) + " | ufuk")
        for b, tau in enumerate(mt["taus"]):
            row = " | ".join(f"{m:5.2f}" for m in s["r2_mean"][b])
            print(f"      {tau:6g} | {row} | {s['horizon_mean'][b]:g}")
    ec = summary["exit_criterion"]
    print(f"\nCikis kriteri (window / fv): ufuk tau_max/tau_min = {ec['horizon_ratio']:.2f} "
          f"(>= {ec['min_ratio']:g}: {ec['ratio_ok']}), en iyi tau kisa={ec['best_tau_short']:g} "
          f"uzun={ec['best_tau_long']:g} ({ec['best_tau_ok']}) -> {'GECTI' if ec['passed'] else 'KALDI'}")
    print(f"\nlog: {logger.dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="koşu için config")
    parser.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER",
                        help="config geçersiz kılma, ör. reservoir.rec_gain=3 experiment.seeds=[100]")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU iş parçacığı sayısı")
    parser.add_argument("--aggregate", nargs="+", metavar="KOŞU_KLASÖRÜ",
                        help="ayrı koşuları (ör. paralel seed grupları) tek özette birleştir; yeni koşu yapmaz")
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.aggregate:
        aggregate([Path(p) for p in args.aggregate])
        return
    if args.config is None:
        parser.error("--config ya da --aggregate gerekli")

    cfg = apply_overrides(load_config(args.config), args.set)
    mt = cfg["memory_test"]
    seeds = cfg["experiment"]["seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seeds, args.config)
    rho = cfg["reservoir"]["rho"]
    keys = [(t, name) for t in mt["targets"] for name in mt["readouts"]]

    curves: Curves = {k: [] for k in keys}
    val_curves: Curves = {k: [] for k in keys}
    rates, thetas = [], []
    for seed in seeds:
        x, r, lin, rate, theta = collect_states(cfg, seed, device)
        N = r.shape[-1] // 2
        views = {"f": r[:, :, :N], "fv": r, "lin": lin}
        for target, name in keys:
            curve, val_curve = memory_curve(cfg, x, views[name], target)    # [B, L] ×2
            curves[(target, name)].append(curve)
            val_curves[(target, name)].append(val_curve)
            lags = mt[TARGET_LAGS[target]]
            for b, tau in enumerate(mt["taus"]):
                logger.log(seed, target=target, readout=name, tau=tau, firing_rate=rate[b].item(),
                           theta=theta[b],
                           **{f"r2_d{d}": float(v) for d, v in zip(lags, curve[b])},
                           **{f"val_r2_d{d}": float(v) for d, v in zip(lags, val_curve[b])})
        del x, r, lin, views
        rates.append(rate.numpy())
        thetas.append(theta)
        print(f"seed {seed}: rate={[round(v, 4) for v in rate.tolist()]} theta={[round(v, 2) for v in theta]}",
              flush=True)
        if ((rate < rho / 2) | (rate > 2 * rho)).any():
            print("  UYARI (K7): atesleme orani [rho/2, 2*rho] disinda")

    report(cfg, keys, curves, val_curves, rates, thetas, logger, extra={})


def aggregate(run_dirs: list[Path]) -> None:
    """Aynı config'le (seed'ler hariç) koşulmuş klasörleri birleştirir."""
    cfgs = [load_config(d / "config.yaml") for d in run_dirs]
    strip = lambda c: {**c, "experiment": {k: v for k, v in c["experiment"].items() if k != "seeds"}}
    if any(strip(c) != strip(cfgs[0]) for c in cfgs[1:]):
        raise ValueError("birleştirilen koşuların config'leri (seed'ler hariç) aynı olmalı")
    cfg = cfgs[0]
    mt = cfg["memory_test"]
    keys = [(t, name) for t in mt["targets"] for name in mt["readouts"]]

    all_seeds: list[int] = []
    curves: Curves = {k: [] for k in keys}
    val_curves: Curves = {k: [] for k in keys}
    rates, thetas = [], []
    for d in run_dirs:
        s, c, vc, rt, th = load_run(d, keys, mt)
        all_seeds += s
        for k in keys:
            curves[k] += c[k]
            val_curves[k] += vc[k]
        rates += rt
        thetas += th
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError(f"aynı seed birden fazla koşuda: {all_seeds}")
    cfg = {**cfg, "experiment": {**cfg["experiment"], "seeds": all_seeds}}
    logger = RunLogger(cfg, all_seeds, "aggregate")
    report(cfg, keys, curves, val_curves, rates, thetas, logger,
           extra={"aggregated_from": [str(d) for d in run_dirs]})


if __name__ == "__main__":
    main()
