"""Tavan ve kayıp ölçümü (KR-033): kayıp rezervuarda mı, okuma kuralında mı, mimaride mi?

Aynı veri, aynı 20 ms ızgara ve aynı 80/10/10 bölme (KR-012, KR-018), test R² (NeuroBench `R2`):
  (a) wiener  — son H kutunun ham sayımları üzerinde ridge (BCI standart taban çizgisi). Deterministik.
  (b) gru     — ham sayımlar (train'de standartlaştırılmış) üzerinde BPTT ile eğitilen GRU. 5 seed.
  (c) ridge_r — rezervuar r = [f, v] üzerinde ridge; Faz 3 tam koşusundan (`ceiling.ref_run`) okunur.
  nlms        — bizim yerel okumamız (A, 12 tur); aynı koşudan.
Okuma: (a) > (c) → rezervuar bilgi kaybettiriyor; (c) > nlms → okuma kuralı zayıf; (b) ≫ hepsi → mimari.

Tüm seçimler (λ, H, gizli boyut, erken durdurma) val'de yapılır; test yalnızca raporlanır.
GRU tavan referansıdır, ajanın parçası değildir (ARCHITECTURE.md K2 notu). Değerlendirmede GRU tüm
dosyayı tek akış olarak, nedensel biçimde görür (rezervuar gibi, K6).

Kullanım:
    uv run python experiments/ceiling.py --config configs/ceiling.yaml [--files ...] [--seeds 0 1] [--skip-gru]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import r2_neurobench, ridge_reference  # noqa: E402
from src.data.primate import Session, load_session  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

STREAM_CHUNK = 16384                      # değerlendirme akışının parça uzunluğu (yalnızca cuDNN sınırı; sonucu değiştirmez)


# --- (a) Wiener filtresi -----------------------------------------------------------------

def history(x: torch.Tensor, H: int) -> torch.Tensor:
    """x: [n, D] → [n, H·D]; satır t = [x(t), x(t−1), …, x(t−H+1)], başta sıfır dolgu (nedensel)."""
    n, D = x.shape
    out = torch.zeros(n, H * D)
    for h in range(H):
        out[h:, h * D:(h + 1) * D] = x[: n - h]
    return out


def wiener(cfg: dict[str, Any], s: Session) -> list[dict[str, Any]]:
    """Her H için ridge (λ val'de). Döndürür: H başına val/test R² satırları."""
    nh, wc = cfg["nhp"], cfg["ceiling"]["wiener"]
    val, test = s.split("val"), s.split("test")
    rows = []
    for H in wc["history"]:
        X = history(s.x, H)                                                  # [n, H·D]
        washout = max(nh["washout"], H)
        pred_va, lam = ridge_reference(X, s.y, s.split("train"), val, val, nh["ridge_grid"], washout)
        pred_te, _ = ridge_reference(X, s.y, s.split("train"), val, test, [lam], washout)
        rows.append({"H": H, "lambda": lam, "r2_val": r2_neurobench(pred_va, s.y[val]),
                     "r2_test": r2_neurobench(pred_te, s.y[test])})
        del X
    return rows


# --- (b) GRU (BPTT) ----------------------------------------------------------------------

class GRUDecoder(nn.Module):
    def __init__(self, D_in: int, hidden: int, num_layers: int, input_dropout: float, D_out: int) -> None:
        super().__init__()
        self.drop = nn.Dropout(input_dropout)
        self.gru = nn.GRU(D_in, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, D_out)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, D_in] → ŷ: [B, L, D_out], h: [layers, B, hidden]"""
        z, h = self.gru(self.drop(x), h)
        return self.out(z), h


@torch.no_grad()
def stream_predict(model: GRUDecoder, x: torch.Tensor, stop: int) -> torch.Tensor:
    """Dizinin başından `stop`'a kadar tek akış (sıfır başlangıç durumu). x: [n, D] → [stop, D_out]

    cuDNN çok uzun diziyi tek çağrıda desteklemediği için parça parça koşulur; gizli durum taşınır (sonuç aynı).
    """
    model.eval()
    h, ys = None, []
    for i in range(0, stop, STREAM_CHUNK):
        y, h = model(x[None, i: min(i + STREAM_CHUNK, stop)], h)
        ys.append(y[0])
    return torch.cat(ys)


def train_gru(cfg: dict[str, Any], s: Session, seed: int, hidden: int,
              device: torch.device) -> tuple[GRUDecoder, float, int]:
    """TBPTT: train ardışık `streams` akışa bölünür, gizli durum akış içinde taşınır. → model (en iyi val), val R², tur"""
    gc = cfg["ceiling"]["gru"]
    tr, va = s.split("train"), s.split("val")
    mu_x, sd_x = s.x[tr].mean(0), s.x[tr].std(0).clamp(min=1e-6)
    mu_y, sd_y = s.y[tr].mean(0), s.y[tr].std(0)
    xs = ((s.x - mu_x) / sd_x).to(device)                                   # [n, D]
    ys = ((s.y - mu_y) / sd_y).to(device)                                   # [n, 2]

    set_seed(seed)
    model = GRUDecoder(s.D_in, hidden, gc["num_layers"], gc["input_dropout"], s.y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    B, L = gc["streams"], gc["chunk"]
    n_tr = tr.stop

    best, best_state, best_ep, bad = -float("inf"), None, 0, 0
    for ep in range(gc["max_epochs"]):
        model.train()
        off = int(torch.randint(0, L, (1,), generator=gen))                  # parça sınırları her turda kayar
        per = (n_tr - off) // B
        n_ch = per // L
        xb = xs[off: off + B * per].view(B, per, -1)                         # [B, per, D]
        yb = ys[off: off + B * per].view(B, per, -1)
        h = None
        for c in range(n_ch):
            sl = slice(c * L, (c + 1) * L)
            pred, h = model(xb[:, sl], h)
            loss = ((pred - yb[:, sl]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gc["grad_clip"])
            opt.step()
            h = h.detach()
        p = stream_predict(model, xs, va.stop)[va.start: va.stop] * sd_y.to(device) + mu_y.to(device)
        r2 = r2_neurobench(p, s.y[va])
        if r2 > best:
            best, best_ep, bad = r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= gc["patience"]:
                break
    model.load_state_dict(best_state)
    model.norm = (mu_x, sd_x, mu_y, sd_y)                                    # değerlendirme için
    return model, best, best_ep


def gru(cfg: dict[str, Any], s: Session, seed: int, device: torch.device) -> dict[str, Any]:
    """Gizli boyut val'de seçilir; seçilen modelin test R²'si raporlanır."""
    t0 = time.time()
    cands = []
    for hidden in cfg["ceiling"]["gru"]["hidden_grid"]:
        model, r2v, ep = train_gru(cfg, s, seed, hidden, device)
        cands.append((r2v, hidden, ep, model))
    r2v, hidden, ep, model = max(cands, key=lambda c: c[0])
    mu_x, sd_x, mu_y, sd_y = model.norm
    xs = ((s.x - mu_x) / sd_x).to(device)
    te = s.split("test")
    p = stream_predict(model, xs, te.stop)[te.start: te.stop].cpu() * sd_y + mu_y
    return {"hidden": hidden, "best_epoch": ep, "r2_val": r2v, "r2_test": r2_neurobench(p, s.y[te]),
            "r2_val_by_hidden": {str(c[1]): c[0] for c in cands}, "sec": time.time() - t0}


# --- (c) ve NLMS: Faz 3 tam koşusundan ---------------------------------------------------

def reference_rows(ref_run: Path) -> dict[str, dict[str, list[float]]]:
    """metrics.jsonl → oturum başına {"ridge_r": [...], "nlms": [...]} (seed sırasıyla)."""
    out: dict[str, dict[str, list[float]]] = {}
    for line in (ref_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        d = out.setdefault(m["session"], {"ridge_r": [], "nlms": [], "seeds": []})
        d["ridge_r"].append(m["r2_ridge"])
        d["nlms"].append(m["r2_A"])
        d["seeds"].append(m["seed"])
    return out


def fmt(v: list[float]) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def write_table(logger: RunLogger, cfg: dict[str, Any], results: dict[str, Any]) -> None:
    hr = cfg["ceiling"]["wiener"]["report_h"]
    lines = ["| Oturum | (a) Wiener H=%d | Wiener (val'de H) | (b) GRU | (c) ridge-r | NLMS (A) | (a)−(c) | (c)−NLMS | (b)−max |" % hr,
             "|---|---|---|---|---|---|---|---|---|"]
    by_monkey: dict[str, dict[str, list[float]]] = {}
    for name, r in results.items():
        wh = next(w for w in r["wiener"] if w["H"] == hr)["r2_test"]
        wsel = max(r["wiener"], key=lambda w: w["r2_val"])
        g = [x["r2_test"] for x in r["gru"]]
        c, nl = r["ref"].get("ridge_r", [float("nan")]), r["ref"].get("nlms", [float("nan")])
        best_other = max(wh, np.mean(c), np.mean(nl))
        gm = np.mean(g) if g else float("nan")
        lines.append(f"| {name} | {wh:.3f} | {wsel['r2_test']:.3f} (H={wsel['H']}) | "
                     f"{fmt(g) if g else '—'} | {fmt(c)} | {fmt(nl)} | {wh - np.mean(c):+.3f} | "
                     f"{np.mean(c) - np.mean(nl):+.3f} | {gm - best_other:+.3f} |")
        mk = by_monkey.setdefault(name.split("_")[0], {"a": [], "b": [], "c": [], "n": []})
        mk["a"].append(wh); mk["b"].append(gm); mk["c"].append(float(np.mean(c))); mk["n"].append(float(np.mean(nl)))
    lines += ["", "| Maymun | (a) Wiener | (b) GRU | (c) ridge-r | NLMS |", "|---|---|---|---|---|"]
    for mk, d in by_monkey.items():
        lines.append(f"| {mk} | {np.mean(d['a']):.3f} | {np.mean(d['b']):.3f} | {np.mean(d['c']):.3f} | {np.mean(d['n']):.3f} |")
    lines += ["", "Test R² (NeuroBench `R2`), 20 ms ızgara, 80/10/10 bölme. GRU, (c) ve NLMS 5 seed ortalama ± std; "
              "Wiener deterministik. (c) ve NLMS: `ceiling.ref_run` (Faz 3 tam koşusu)."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--skip-gru", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="config geçersiz kılmaları: a.b=değer")
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set)
    files = a.files or cfg["data"]["files"]
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["seeds"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seeds, a.config)
    ref = reference_rows(Path(cfg["ceiling"]["ref_run"]))

    results: dict[str, Any] = {}
    step = 0
    for name in files:
        s = load_session(cfg, name)                                          # ham sayımlar (ölçeksiz)
        t0 = time.time()
        wr = wiener(cfg, s)
        print(json.dumps({"session": name, "wiener": wr, "sec": time.time() - t0}), flush=True)
        gr = []
        for seed in ([] if a.skip_gru else seeds):
            row = gru(cfg, s, seed, device)
            gr.append({"seed": seed, **row})
            print(json.dumps({"session": name, "seed": seed, "gru": row}), flush=True)
            logger.log(step, session=name, seed=seed, kind="gru", **{k: v for k, v in row.items()})
            step += 1
        for w in wr:
            logger.log(step, session=name, kind="wiener", **w)
            step += 1
        results[name] = {"wiener": wr, "gru": gr, "ref": ref.get(name, {})}
        logger.save_json("results.json", results)
    write_table(logger, cfg, results)
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
