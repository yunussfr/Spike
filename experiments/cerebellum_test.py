"""Faz 2: beyincik bloğu ile Mackey-Glass tek adımlık tahmin.

Akış (her seed için):
  1. Seri x(0..S−1) kodlanır ve rezervuardan geçirilir → r(t) = [f, v]. Regülatörler kapalı
     (K5), rezervuar sabit (K1) ve bu fazda a(t) = 0 olduğu için r(t) öğrenmeden bağımsızdır;
     bir kez hesaplanıp turlar boyunca yeniden kullanılır.
  2. Beyincik x̂(t+1) = W_c·[r(t), a(t)] tahmin eder; e = x(t+1) − x̂(t+1) ile yerel kuralla öğrenir.
     Eğitim dizisi üzerinde sabit sayıda tur atılır; test sırasında ağırlıklar dondurulur.
     Ek olarak "test sırasında da öğrenme açık" (önce tahmin, sonra güncelleme) raporlanır.
  3. Karşılaştırmalar: "önceki değeri tekrarla" taban çizgisi ve aynı r(t) üzerinde çevrimdışı
     ridge (λ eğitimin son val_frac kısmında seçilir; üst sınır referansı).

Kullanım:
    uv run python experiments/cerebellum_test.py --config configs/chaotic.yaml
"""

import argparse
import copy
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.cerebellum import Cerebellum
from src.encoder import SpikeEncoder
from src.reservoir import LIFReservoir
from src.utils.config import apply_overrides, load_config, resolve_device
from src.utils.logging import RunLogger
from src.utils.metrics import mse, nrmse, ridge_fit, ridge_predict
from src.utils.seeding import set_seed


def load_series(cfg: dict[str, Any]) -> torch.Tensor:
    """NeuroBench Mackey-Glass serisi: [S, 1], S = n_train + n_test + 1 (son nokta yalnızca hedef)."""
    d = cfg["data"]
    a = np.load(d["file"], allow_pickle=False)
    S = d["n_train"] + d["n_test"] + 1
    a = a[d["start_offset"]: d["start_offset"] + S]
    if len(a) < S:
        raise ValueError(f"seri çok kısa: {len(a)} < {S}")
    return torch.tensor(a, dtype=torch.float32).unsqueeze(1)


def resolve_level_range(cfg: dict[str, Any], x: torch.Tensor) -> dict[str, Any]:
    """`encoder.level.range: train` → eğitim bölümünün [min, max]'ı (test verisine bakılmaz)."""
    cfg = copy.deepcopy(cfg)
    if cfg["encoder"]["level"]["range"] == "train":
        tr = x[: cfg["data"]["n_train"]]
        cfg["encoder"]["level"]["range"] = [float(tr.min()), float(tr.max())]
    return cfg


@torch.no_grad()
def collect_features(cfg: dict[str, Any], x: torch.Tensor, seed: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """x: [S, 1] → r: [S, 2N] (r[t], x(0..t)'den hesaplanır), ortalama ateşleme oranı."""
    set_seed(seed)
    reservoir = LIFReservoir.from_config(cfg, D_in=x.shape[1], seed=seed, calibrate=False).to(device)
    reservoir.calibrate_from_config(cfg, x.shape[1], seed)
    encoder = SpikeEncoder.from_config(cfg, D_in=x.shape[1])
    g, tau = cfg["regulators"]["defaults"]["g"], cfg["regulators"]["defaults"]["tau"]
    r = torch.empty(len(x), 2 * reservoir.N)
    for t in range(len(x)):
        r[t] = reservoir(encoder(x[t: t + 1].to(device)), g=g, tau=tau)[0].cpu()
    rate = r[cfg["chaotic_test"]["washout"]:, : reservoir.N].mean().item()
    return r, rate


def run_cerebellum(
    cfg: dict[str, Any], rule: str, r: torch.Tensor, x: torch.Tensor,
    train_rows: torch.Tensor, test_rows: torch.Tensor,
) -> dict[str, Any]:
    """Yerel kuralla eğitir; dondurulmuş ve çevrimiçi test tahminlerini döndürür."""
    ct = cfg["chaotic_test"]
    c = copy.deepcopy(cfg)
    c["cerebellum"]["rule"] = rule
    eta = cfg["regulators"]["defaults"]["eta"]
    cb = Cerebellum.from_config(c, n_features=r.shape[1], D_a=1, D_pred=1)
    a0 = torch.zeros(1, 1)                                                   # Faz 2: a(t) = 0

    curve = []
    diverged = False
    for _ in range(ct["epochs"]):
        cb.reset_state()
        sq = 0.0
        for t in train_rows.tolist():
            x_hat = cb(r[t: t + 1], a0)                                      # x(t+1) görülmez
            e = x[t + 1: t + 2] - x_hat                                      # x(t+1) şimdi gelir
            cb.update(e, eta)
            sq += e.pow(2).item()
        curve.append(sq / len(train_rows))
        if not math.isfinite(curve[-1]) or curve[-1] > 1e6:
            diverged = True
            break

    def predict(learn: bool) -> torch.Tensor:
        state = (cb.W_c.clone(), cb.mu.clone(), cb.var.clone(), cb.n_seen)
        preds = []
        for t in test_rows.tolist():
            x_hat = cb(r[t: t + 1], a0)
            preds.append(x_hat)
            if learn:
                cb.update(x[t + 1: t + 2] - x_hat, eta)
        cb.W_c, cb.mu, cb.var, cb.n_seen = state
        return torch.cat(preds)                                              # [n_test, 1]

    if diverged:
        nan = torch.full((len(test_rows), 1), float("nan"))
        return {"diverged": True, "curve": curve, "frozen": nan, "online": nan}
    return {"diverged": False, "curve": curve, "frozen": predict(False), "online": predict(True)}


def ridge_reference(cfg: dict[str, Any], r: torch.Tensor, x: torch.Tensor,
                    train_rows: torch.Tensor, test_rows: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Aynı r(t) üzerinde çevrimdışı ridge. λ eğitimin son val_frac kısmında seçilir."""
    n_val = int(len(train_rows) * cfg["data"]["val_frac"])
    fit_rows, val_rows = train_rows[:-n_val], train_rows[-n_val:]
    R = r.double()
    Y = x.double()

    def standardized(rows_stat: torch.Tensor) -> torch.Tensor:
        mu, sd = R[rows_stat].mean(dim=0), R[rows_stat].std(dim=0).clamp(min=1e-8)
        return (R - mu) / sd

    Z = standardized(fit_rows)
    val_err = [mse(ridge_predict(Z[val_rows], ridge_fit(Z[fit_rows], Y[fit_rows + 1], lam)), Y[val_rows + 1])
               for lam in cfg["chaotic_test"]["ridge_grid"]]
    lam = cfg["chaotic_test"]["ridge_grid"][int(np.argmin(val_err))]
    Z = standardized(train_rows)
    W = ridge_fit(Z[train_rows], Y[train_rows + 1], lam)
    return ridge_predict(Z[test_rows], W).float(), lam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="ANAHTAR=DEĞER")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = apply_overrides(load_config(args.config), args.set)
    x = load_series(cfg)                                                     # [S, 1]
    cfg = resolve_level_range(cfg, x)
    ct, d = cfg["chaotic_test"], cfg["data"]
    seeds = cfg["experiment"]["seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seeds, args.config)
    rho = cfg["reservoir"]["rho"]

    train_rows = torch.arange(ct["washout"], d["n_train"])                  # hedef x(t+1), t+1 ≤ n_train
    test_rows = torch.arange(d["n_train"], d["n_train"] + d["n_test"])
    y_test = x[test_rows + 1]                                                # [n_test, 1]
    repeat = x[test_rows]                                                    # x̂(t+1) = x(t)
    mse_repeat = mse(repeat, y_test)

    results: dict[str, list[float]] = {}
    curves: dict[str, list[list[float]]] = {rule: [] for rule in ct["rules"]}
    example: dict[str, torch.Tensor] = {}

    def add(key: str, value: float) -> None:
        results.setdefault(key, []).append(value)

    for seed in seeds:
        r, rate = collect_features(cfg, x, seed, device)
        ridge_pred, lam = ridge_reference(cfg, r, x, train_rows, test_rows)
        row: dict[str, Any] = {"firing_rate": rate, "ridge_lambda": lam,
                               "mse_repeat": mse_repeat, "mse_ridge": mse(ridge_pred, y_test),
                               "nrmse_ridge": nrmse(ridge_pred, y_test)}
        for rule in ct["rules"]:
            out = run_cerebellum(cfg, rule, r, x, train_rows, test_rows)
            curves[rule].append(out["curve"])
            row[f"{rule}_diverged"] = out["diverged"]
            for mode in ("frozen", "online"):
                row[f"mse_{rule}_{mode}"] = mse(out[mode], y_test)
                row[f"nrmse_{rule}_{mode}"] = nrmse(out[mode], y_test)
            if seed == seeds[0]:
                example[rule] = out["frozen"]
        if seed == seeds[0]:
            example["ridge"] = ridge_pred
        for k, v in row.items():
            if not isinstance(v, bool):
                add(k, float(v))
        logger.log(seed, **{k: (float(v) if not isinstance(v, bool) else v) for k, v in row.items()})
        print(f"seed {seed}: rate={rate:.4f}  " + "  ".join(
            f"{k}={v:.5f}" for k, v in row.items() if k.startswith("mse_")), flush=True)
        if not (rho / 2 <= rate <= 2 * rho):
            print("  UYARI (K7): atesleme orani [rho/2, 2*rho] disinda")

    summary: dict[str, Any] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in results.items()}
    ec = ct["exit_criterion"]
    rule = "normalized" if "normalized" in ct["rules"] else ct["rules"][0]
    m_local = summary[f"mse_{rule}_frozen"]["mean"]
    m_ridge = summary["mse_ridge"]["mean"]
    crit = {
        "rule": rule,
        "mse_vs_repeat": m_local / mse_repeat,
        "gap_vs_ridge": (m_local - m_ridge) / m_ridge,
    }
    crit["repeat_ok"] = bool(crit["mse_vs_repeat"] <= ec["max_mse_vs_repeat"])
    crit["ridge_ok"] = bool(crit["gap_vs_ridge"] < ec["max_gap_vs_ridge"])
    crit["passed"] = crit["repeat_ok"] and crit["ridge_ok"]
    summary["exit_criterion"] = crit
    logger.save_json("summary.json", summary)
    plot(cfg, x, test_rows, example, curves, logger.path("chaotic.png"))

    print("\nTest MSE (5 seed ortalama +- std):")
    for k in sorted(k for k in summary if k.startswith("mse_")):
        print(f"  {k:28s} {summary[k]['mean']:.6f} +- {summary[k]['std']:.6f}")
    print(f"\nCikis kriteri ({rule}, donmus): MSE/tekrar = {crit['mse_vs_repeat']:.3f} "
          f"(<= {ec['max_mse_vs_repeat']}: {crit['repeat_ok']}), ridge farki = {crit['gap_vs_ridge']:+.1%} "
          f"(< {ec['max_gap_vs_ridge']:.0%}: {crit['ridge_ok']}) -> {'GECTI' if crit['passed'] else 'KALDI'}")
    print(f"\nlog: {logger.dir}")


def plot(cfg: dict[str, Any], x: torch.Tensor, test_rows: torch.Tensor, example: dict[str, torch.Tensor],
         curves: dict[str, list[list[float]]], path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.3))
    t = test_rows.numpy()
    ax1.plot(t + 1, x[test_rows + 1, 0].numpy(), color="black", lw=1.5, label="gerçek x(t+1)")
    for name, pred in example.items():
        if torch.isfinite(pred).all():
            ax1.plot(t + 1, pred[:, 0].numpy(), lw=1, label=name)
    ax1.set(xlabel="zaman adımı", ylabel="x", title="Test bölümü, tek adımlık tahmin (ilk seed)")
    ax1.legend()
    for rule, cs in curves.items():
        for i, c in enumerate(cs):
            ax2.plot(np.arange(1, len(c) + 1), c, color=f"C{list(curves).index(rule)}", alpha=0.6,
                     label=rule if i == 0 else None)
    ax2.set(yscale="log", xlabel="tur (epoch)", ylabel="eğitim MSE", title="Beyincik öğrenme eğrisi")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
