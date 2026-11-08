"""Dinamik duygu izleri (V5 çıkış kriteri) ve elektrot kaybı toparlanma eğrileri (V6), experiments/v2_v6.py koşusundan.

    traces:   oturum başına bir sütun; satırlar sensör (s_sens, s_drift, a, λ_norm), iç arousal (a_int, u),
              valens / dominans, η, α_anchor, konsolidasyon / fren
    recovery: kayıp öncesi ve sonrası kayan pencere R² (koşul başına), kayıp anı dikey çizgi

Kullanım:
    uv run python scripts/plot_v2_traces.py traces --run runs/<v6> --scenario crossday --seed 100 \
        --sessions indy_20160622_01 indy_20170131_02 --conds D C_D P_D --out docs/figures/v5_traces_indy.png
    uv run python scripts/plot_v2_traces.py recovery --run runs/<v6> --seed 0 --sessions loco_20170210_03 \
        --conds A Bm_D P_D D --out docs/figures/v6_recovery_loco.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROWS = [("sensör", [("s_sens", "beyincik (izleme) / ε²₀"), ("s_drift", "girdi sapması / δ₀")], "log"),
        ("sensör arousal ve λ_norm", [("a", "a (sensör)"), ("lam_n", "λ_norm / λ_surp")], None),
        ("kontrol arousal'ı ve uyuşmazlık", [("a_int", "a (kontrol)"), ("u", "başlık uyuşmazlığı / u₀")], None),
        ("valens / dominans", [("val", "valens"), ("dom", "dominans")], None),
        ("η (okuma adımı)", [("eta", "η")], "log"),
        ("α_anchor", [("alpha", "α")], "log"),
        ("konsolidasyon / fren", [("cons", "c"), ("brake", "b")], None)]


def traces(a: argparse.Namespace) -> None:
    cols = len(a.sessions)
    fig, ax = plt.subplots(len(ROWS), cols, figsize=(5.5 * cols, 2.0 * len(ROWS)), sharex="col", squeeze=False)
    for j, se in enumerate(a.sessions):
        st = torch.load(Path(a.run) / f"trace_{a.scenario}_{se}_seed{a.seed}.pt", weights_only=False)
        tr = st["trace"]
        for i, (title, keys, scale) in enumerate(ROWS):
            axx = ax[i, j]
            for c in a.conds:
                if c not in tr:
                    continue
                d = dict(tr[c])
                lam = d["lam"].numpy()
                d["lam_n"] = torch.tensor(lam / max(float(lam.max()), 1e-12))
                for k, lab in keys:
                    if k not in d:
                        continue
                    v = d[k].numpy()
                    if scale == "log":
                        v = np.maximum(v, 1e-8)
                    axx.plot(np.arange(len(v)), v, lw=0.9, label=f"{c}: {lab}" if len(a.conds) > 1 else lab)
                if title.startswith("sensör"):
                    break                                                    # sensör tüm koşullarda aynı
            if scale == "log":
                axx.set_yscale("log")
            axx.set_title(f"{se} — {title}" if i == 0 else title, fontsize=8)
            axx.tick_params(labelsize=7)
            axx.legend(fontsize=6, loc="upper right")
        ax[-1, j].set_xlabel("zaman (s)", fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


def recovery(a: argparse.Namespace) -> None:
    cols = len(a.sessions)
    fig, ax = plt.subplots(1, cols, figsize=(5.5 * cols, 3.2), squeeze=False)
    for j, se in enumerate(a.sessions):
        st = torch.load(Path(a.run) / f"trace_electrode_{se}_seed{a.seed}.pt", weights_only=False)
        dt_s, t_drop = a.dt_ms / 1000.0, st["t_drop"]
        for c in a.conds:
            if c in st.get("rolling", {}):
                s, r = st["rolling"][c]
                ax[0, j].plot((s + a.window) * dt_s, r, lw=1.0, label=c)
        ax[0, j].axvline(t_drop * dt_s, color="k", ls="--", lw=0.8)
        ax[0, j].set_ylim(-1.0, 1.0)                                         # durgun pencerelerde R² ≪ −1 olabilir
        ax[0, j].set_title(f"{se}: kayan R² (pencere sonu), seed {a.seed}", fontsize=8)
        ax[0, j].set_xlabel("zaman (s)", fontsize=8)
        ax[0, j].legend(fontsize=7)
        ax[0, j].tick_params(labelsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["traces", "recovery"])
    p.add_argument("--run", required=True)
    p.add_argument("--scenario", default="crossday")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--conds", nargs="+", default=["D"])
    p.add_argument("--dt_ms", type=float, default=4.0)
    p.add_argument("--window", type=int, default=1250, help="kayan R² pencere uzunluğu (adım), konu ekseni için")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    traces(a) if a.mode == "traces" else recovery(a)


if __name__ == "__main__":
    main()
