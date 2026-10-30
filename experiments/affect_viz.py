"""Faz 4: duygu uzayı görselleştirmesi ve nicel kontroller (bir nhp_motor koşu klasörü üzerinden).

Her (oturum, seed) için `preds_*.pt` içindeki "B+Z" izlerini okur:
  1. Baş hatası eğitim boyunca düşüyor mu? Standart ölçekte baş MSE'si: train'in ilk %10'u vs son %10'u.
  2. z'de iyi/kötü anlar ayrışıyor mu? Val+test'te q'nun üst çeyreği "iyi", alt çeyreği "kötü".
     z'den q'ya doğrusal okuma train izinde kurulur (ridge, yalnızca analiz); val+test'te
     bu skorla iyi/kötü ayrımının AUC'si raporlanır (0,5 = ayrışma yok).
  3. Görseller (ilk seed): z'nin 3B saçılımı (renk = q), baş tahmini vs hedef zaman grafikleri,
     kayan baş MSE eğrisi.

Kullanım:
    uv run python experiments/affect_viz.py --run runs/<koşu klasörü> [--seed 0]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HEADS = ("q", "u", "s")
NAMES = {"q": "q = −‖δ‖² (performans)", "u": "u = ‖e‖² (belirsizlik)", "s": "s (eğilim)"}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney AUC: rastgele bir 'iyi' anın skoru rastgele bir 'kötü' anınkinden büyük olma olasılığı."""
    order = score.argsort()
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def analyse(a: dict) -> dict:
    tr, ev = a["train"], a["eval"]
    err = tr["head_err"].numpy() ** 2                                        # [n, 3]
    n = len(err)
    w = max(1, n // 10)
    out = {}
    for i, h in enumerate(HEADS):
        first, last = float(err[:w, i].mean()), float(err[-w:, i].mean())
        out[f"head_mse_{h}_first10"] = first
        out[f"head_mse_{h}_last10"] = last
        out[f"head_mse_{h}_eval"] = float((ev["head_err"][:, i].numpy() ** 2).mean())
        hc = a["head_curve"][:, i]                                           # [epochs] tur başına MSE
        out[f"head_mse_{h}_epoch_first"] = float(hc[0])
        out[f"head_mse_{h}_epoch_last"] = float(hc[-1])
        # Ölçüt: son turun hatası ilk turunkinden düşük (tek turluk koşuda tur içi ilk/son %10)
        out[f"head_declines_{h}"] = bool(hc[-1] < hc[0]) if len(hc) > 1 else bool(last < first)
    # z → q doğrusal okuma (train izinde), iyi/kötü AUC (eval izinde)
    Z = np.c_[tr["z"].numpy(), np.ones(n)]
    q = tr["targets"][:, 0].numpy()
    w_q = np.linalg.lstsq(Z, q, rcond=None)[0]
    Ze = np.c_[ev["z"].numpy(), np.ones(len(ev["z"]))]
    qe = ev["targets"][:, 0].numpy()
    lo, hi = np.quantile(qe, [0.25, 0.75])
    mask = (qe <= lo) | (qe >= hi)
    out["auc_good_bad_from_z"] = auc((Ze @ w_q)[mask], (qe[mask] >= hi).astype(int))
    out["z_std"] = [float(v) for v in ev["z"].std(dim=0)]
    return out


def plot(a: dict, title: str, path: Path) -> None:
    tr, ev = a["train"], a["eval"]
    fig = plt.figure(figsize=(18, 10))
    # 1) z 3B saçılım, renk = q (eval)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    z, q = ev["z"].numpy(), ev["targets"][:, 0].numpy()
    lo, hi = np.quantile(q, [0.02, 0.98])
    sc = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=np.clip(q, lo, hi), cmap="viridis", s=2, alpha=0.6)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="q (yüksek = iyi performans)")
    ax.set(xlabel="z₁", ylabel="z₂", zlabel="z₃", title="z(t), val+test (renk = q)")
    # 2) tur başına baş MSE (train, standart ölçek)
    ax = fig.add_subplot(2, 3, 2)
    hc = a["head_curve"].numpy()
    for i, h in enumerate(HEADS):
        ax.plot(np.arange(1, len(hc) + 1), hc[:, i], marker="o", label=h)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="ortalama tahmini (≈1)")
    ax.set(xlabel="tur (epoch)", ylabel="baş MSE (standart ölçek)", title="Baş hatası eğitim boyunca")
    ax.legend()
    # 3) iyi/kötü anlar: z₁–z₂ düzleminde
    ax = fig.add_subplot(2, 3, 3)
    ql, qh = np.quantile(q, [0.25, 0.75])
    ax.scatter(z[q <= ql, 0], z[q <= ql, 1], s=2, alpha=0.4, c="C3", label="kötü (q alt çeyrek)")
    ax.scatter(z[q >= qh, 0], z[q >= qh, 1], s=2, alpha=0.4, c="C2", label="iyi (q üst çeyrek)")
    ax.set(xlabel="z₁", ylabel="z₂", title="İyi ve kötü anlar")
    ax.legend(markerscale=5)
    # 4–6) baş tahmini vs hedef (eval başından 1500 adım)
    m = min(1500, len(ev["z"]))
    for i, h in enumerate(HEADS):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.plot(ev["targets"][:m, i].numpy(), color="black", lw=0.8, label="hedef")
        ax.plot(ev["heads"][:m, i].numpy(), color=f"C{i}", lw=0.8, label="baş tahmini")
        ax.set(xlabel="val adımı", title=NAMES[h])
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=None, help="görsel için seed (varsayılan: ilk bulunan)")
    a = p.parse_args()
    run = Path(a.run)
    results, plotted = [], set()
    for f in sorted(run.glob("preds_*_seed*.pt")):
        name, seed = f.stem[len("preds_"):].rsplit("_seed", 1)
        d = torch.load(f)
        if "B+Z" not in d.get("affect", {}):
            continue
        aff = d["affect"]["B+Z"]
        row = {"session": name, "seed": int(seed), **analyse(aff)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if name not in plotted and (a.seed is None or int(seed) == a.seed):
            plot(aff, f"Duygu uzayı — {name}, seed {seed}", run / f"affect_{name}_seed{seed}.png")
            plotted.add(name)
    summary = {}
    for name in dict.fromkeys(r["session"] for r in results):
        rs = [r for r in results if r["session"] == name]
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))}
                         for k in rs[0] if isinstance(rs[0][k], float)}
        summary[name].update({f"all_seeds_{k}": all(r[k] for r in rs) for k in rs[0] if k.startswith("head_declines")})
    (run / "affect_summary.json").write_text(json.dumps({"rows": results, "summary": summary}, indent=2,
                                                        ensure_ascii=False), encoding="utf-8")
    print(f"özet: {run / 'affect_summary.json'}")


if __name__ == "__main__":
    main()
