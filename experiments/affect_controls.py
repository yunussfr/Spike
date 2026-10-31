"""Faz 4 kontrolleri: z'deki iyi/kötü ayrımı gerçekten öğrenmeden mi geliyor?

`affect_viz.py`'deki AUC (z → q doğrusal okuma train'de, iyi/kötü AUC eval'de) aynı yöntemle
dört farklı girdiden hesaplanır ve karşılaştırılır:
  - `trained_z`  : eğitilmiş z (affect_viz ile aynı sayı, izden okunur).
  - `random_z`   : eğitilmemiş z = tanh(W_z⁰ · φ̃(r)); W_z⁰ aynı seed'in başlangıç ağırlığı (`lr_z = 0`).
                   q etiketi değişmez: K3 gereği δ duygu uzayından bağımsızdır.
  - `speed`      : yalnızca hareket hızı |y(t)| (ve |y|²). "Kötü an = hızlı hareket" açıklamasını sınar.
  - `trained_z+speed`: z hızın ötesinde bilgi taşıyor mu?
  - `r_ridge`    : tüm r (2N boyut) üzerinden ridge → tavan: bilgi rezervuarda ne kadar var?
                   (λ, q'nun val MSE'siyle seçilir; yalnızca analiz.)

Eğitilmemiş z için duygu uzayının standartlaştırıcısı r üzerinde yeniden yürütülür. Oran
`affect.norm_rate` sabit bir EMA olduğundan (1/n_seen > rate yalnızca ilk adımlarda) iki geçiş
yeterlidir: birinci geçiş ısıtır, ikinci geçiş (eğitimin son turuna karşılık gelir) z'yi kaydeder;
eval'de istatistikler donuktur (KR-017). Böylece 12 turu yeniden koşmaya gerek kalmaz.

Kullanım:
    uv run python experiments/affect_controls.py --run runs/<koşu klasörü> [--config configs/nhp_motor.yaml]
Çıktı: `<run>/affect_controls.json` (satır = oturum × seed, oturum özeti).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.affect_viz import auc  # noqa: E402
from experiments.nhp_full import _CachedReservoir, _chunks  # noqa: E402
from experiments.nhp_motor import make_agent  # noqa: E402
from src.batched import BatchedStandardizer  # noqa: E402
from src.data.primate import load_session, prepare  # noqa: E402
from src.utils.config import load_config  # noqa: E402

METHODS = ("trained_z", "random_z", "speed", "trained_z+speed", "r_ridge")


def auc_from(X_tr: np.ndarray, q_tr: np.ndarray, X_ev: np.ndarray, q_ev: np.ndarray) -> float:
    """affect_viz ile aynı ölçüt: [X, 1] → q doğrusal okuma (train), q'nun alt/üst çeyreği (eval)."""
    w = np.linalg.lstsq(np.c_[X_tr, np.ones(len(X_tr))], q_tr, rcond=None)[0]
    score = np.c_[X_ev, np.ones(len(X_ev))] @ w
    lo, hi = np.quantile(q_ev, [0.25, 0.75])
    mask = (q_ev <= lo) | (q_ev >= hi)
    return auc(score[mask], (q_ev[mask] >= hi).astype(int))


@torch.no_grad()
def random_z(cfg: dict, s, rs: list[np.ndarray], seeds: list[int], tr0: int, tr1: int, ev0: int, ev1: int,
             chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Eğitilmemiş z izleri. → train [M, tr1−tr0, k], eval [M, ev1−ev0, k]."""
    N = rs[0].shape[1] // 2
    agents = [make_agent(cfg, "B+Z", s, _CachedReservoir(N), sd) for sd in seeds]
    norm = BatchedStandardizer.from_singles([a.affect.norm for a in agents], torch.device("cpu"))
    W_z = torch.stack([a.affect.W_z for a in agents])                        # [M, k, F+1]

    def block(a: int, b: int) -> torch.Tensor:                               # [M, b−a, 2N]
        return torch.from_numpy(np.stack([np.asarray(r[a:b]) for r in rs]))

    def run(a0: int, a1: int, record: bool) -> torch.Tensor | None:
        out = []
        for a, b in _chunks(a0, a1, chunk):
            R = block(a, b)
            for i in range(b - a):
                phi = norm(R[:, i])                                          # [M, F+1]
                if record:
                    out.append(torch.tanh(torch.einsum("mf,mkf->mk", phi, W_z)))
        return torch.stack(out, dim=1) if record else None

    run(tr0, tr1, record=False)                                              # ısıtma geçişi
    z_tr = run(tr0, tr1, record=True)                                        # "son tur"
    norm.frozen = True                                                       # eval: donuk (KR-017)
    z_ev = run(ev0, ev1, record=True)
    return z_tr, z_ev


def ridge_auc(r: np.ndarray, tr: slice, val: slice, ev: slice, q_tr: np.ndarray, q_ev: np.ndarray,
              n_val: int, grid: list[float], chunk: int) -> tuple[float, float]:
    """Tüm r'den q'ya ridge (standartlaştırılmış r). λ val'de seçilir. → (eval AUC, λ)."""
    F = r.shape[1]
    s1 = np.zeros(F)
    s2 = np.zeros(F)
    for a, b in _chunks(tr.start, tr.stop, chunk):
        X = np.asarray(r[a:b], dtype=np.float64)
        s1 += X.sum(0)
        s2 += (X ** 2).sum(0)
    n = tr.stop - tr.start
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-8

    def Z(a: int, b: int) -> np.ndarray:
        X = (np.asarray(r[a:b], dtype=np.float64) - mu) / sd
        return np.c_[X, np.ones(len(X))]

    G = np.zeros((F + 1, F + 1))
    B = np.zeros(F + 1)
    for a, b in _chunks(tr.start, tr.stop, chunk):
        X = Z(a, b)
        G += X.T @ X
        B += X.T @ q_tr[a - tr.start: b - tr.start]
    Zev = Z(ev.start, ev.stop)
    best = None
    for lam in grid:
        reg = lam * np.eye(F + 1)
        reg[-1, -1] = 0.0                                                    # bias düzenlenmez
        w = np.linalg.solve(G + reg, B)
        mse = float(((Zev[:n_val] @ w - q_ev[:n_val]) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, w)
    score = Zev @ best[2]
    lo, hi = np.quantile(q_ev, [0.25, 0.75])
    mask = (q_ev <= lo) | (q_ev >= hi)
    return auc(score[mask], (q_ev[mask] >= hi).astype(int)), best[1]


def session_controls(cfg: dict, run: Path, name: str, seeds: list[int]) -> list[dict]:
    cfg_s, s = prepare(cfg, load_session(cfg, name))
    chunk = cfg_s["nhp"].get("chunk", 4096)
    traces = [torch.load(run / f"preds_{name}_seed{sd}.pt")["affect"]["B+Z"] for sd in seeds]
    t0 = traces[0]
    tr0, ev0 = t0["train_start"], t0["eval_start"]
    tr1, ev1 = tr0 + len(t0["train"]["z"]), ev0 + len(t0["eval"]["z"])
    n_val = s.test_start - s.val_start
    rs = [np.load(run / "r" / f"{name}_seed{sd}.npy", mmap_mode="r") for sd in seeds]
    z_tr_rand, z_ev_rand = random_z(cfg_s, s, rs, seeds, tr0, tr1, ev0, ev1, chunk)

    sp = s.y.norm(dim=-1).numpy()                                            # |y(t)|  [n]
    speed_tr = np.c_[sp[tr0:tr1], sp[tr0:tr1] ** 2]
    speed_ev = np.c_[sp[ev0:ev1], sp[ev0:ev1] ** 2]
    rows = []
    for m, sd in enumerate(seeds):
        a = traces[m]
        q_tr, q_ev = a["train"]["targets"][:, 0].numpy(), a["eval"]["targets"][:, 0].numpy()
        z_tr, z_ev = a["train"]["z"].numpy(), a["eval"]["z"].numpy()
        row = {"session": name, "seed": sd,
               "trained_z": auc_from(z_tr, q_tr, z_ev, q_ev),
               "random_z": auc_from(z_tr_rand[m].numpy(), q_tr, z_ev_rand[m].numpy(), q_ev),
               "speed": auc_from(speed_tr, q_tr, speed_ev, q_ev),
               "trained_z+speed": auc_from(np.c_[z_tr, speed_tr], q_tr, np.c_[z_ev, speed_ev], q_ev),
               "random_z_std": [float(v) for v in z_ev_rand[m].std(dim=0)]}
        row["r_ridge"], row["r_ridge_lambda"] = ridge_auc(
            rs[m], slice(tr0, tr1), slice(ev0, ev0 + n_val), slice(ev0, ev1), q_tr, q_ev, n_val,
            cfg_s["nhp"]["ridge_grid"], chunk)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k != "random_z_std"}, ensure_ascii=False), flush=True)
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="configs/nhp_motor.yaml")
    p.add_argument("--files", nargs="*", default=None)
    a = p.parse_args()
    run = Path(a.run)
    cfg = load_config(a.config)
    files = a.files or cfg["data"]["files"]
    seeds = cfg["experiment"]["seeds"]
    torch.set_num_threads(4)
    rows = [r for name in files for r in session_controls(cfg, run, name, seeds)]
    summary = {name: {k: {"mean": float(np.mean([r[k] for r in rows if r["session"] == name])),
                          "std": float(np.std([r[k] for r in rows if r["session"] == name]))}
                      for k in METHODS} for name in files}
    (run / "affect_controls.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2,
                                                         ensure_ascii=False), encoding="utf-8")
    print("\n| Oturum | " + " | ".join(METHODS) + " |\n|---|" + "---|" * len(METHODS))
    for name, sm in summary.items():
        print(f"| {name} | " + " | ".join(f"{sm[k]['mean']:.3f} ± {sm[k]['std']:.3f}" for k in METHODS) + " |")
    print(f"\nözet: {run / 'affect_controls.json'}")


if __name__ == "__main__":
    main()
