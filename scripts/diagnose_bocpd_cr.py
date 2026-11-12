"""BOCPD CR teşhisi, üretim koduyla (src/v2/changepoint.py, §12.16, KR-050): kaydedilmiş blok dizileri üzerinde
kaynak (x, ar1, cer_mon) × Day-0'a dönüş p_0 karşılaştırması.

Girdi: `scripts/diagnose_cer_bocpd.py --save_blocks` çıktısı blocks_<primat>_seed<s>.pt. Kaynak eşlemesi:
x → S0s_x, ar1 → S0s_x (AR(1) dönüşümü fit_score_ref'te), cer_mon → S_M1_h1. Yalnız okur; test verisi yok.

Kullanım:
    uv run python scripts/diagnose_bocpd_cr.py --config configs/v2_v7.yaml --blocks runs/<diag>/blocks_*.pt --p0 0.5 0.9
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.v2.changepoint import BocpdSpec, bocpd_series, fit_score_ref  # noqa: E402

MAP = {"x": "S0s_x", "ar1": "S0s_x", "cer_mon": "S_M1_h1"}


def first(a: torch.Tensor, at: int = 0) -> int:
    i = (a[at:] > 0.5).nonzero()
    return int(i[0, 0]) if len(i) else -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--blocks", nargs="+", required=True)
    p.add_argument("--sources", nargs="*", default=["x", "ar1", "cer_mon"])
    p.add_argument("--p0", nargs="*", type=float, default=[0.5, 0.9])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cfg = load_config(a.config)
    bc = cfg["bocpd"]
    grid = sorted(float(v) for v in bc["temper_grid"])
    target, eps, floor = float(bc["calib_false_alarm"]), float(bc["score_eps"]), float(bc["var_floor"])
    lines = ["| primat | seed | kaynak | p_0 | τ_T | kalibre | YA ayrılmış | YA Day-0 val | kayıp tespiti (s) | Day-k tespit oranı | "
             "Day-k ilk tespit (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in a.blocks:
        B = torch.load(path)
        blk_s, days, drop = B["blk_s"], B["days"], B["drop_blk"]
        for src in a.sources:
            blocks = {k: v.float() for k, v in B["src"][MAP[src]].items()}
            ref = fit_score_ref(blocks["stat"], src, eps, floor)
            for p0 in a.p0:
                sp0 = replace(BocpdSpec.from_config(cfg, source=src), block=1, p0=p0)
                tau, fa_h = None, None
                for t in grid:
                    fa = float((bocpd_series(blocks["hold"], ref, replace(sp0, temper=t)) > 0.5).float().mean())
                    if fa > target:
                        break
                    tau, fa_h = t, fa
                ok = tau is not None
                if not ok:
                    tau = grid[0]
                    fa_h = float((bocpd_series(blocks["hold"], ref, replace(sp0, temper=tau)) > 0.5).float().mean())
                sp = replace(sp0, temper=tau)
                a0 = bocpd_series(blocks["val0"], ref, sp)
                ad = bocpd_series(blocks["drop"], ref, sp)
                fb = first(ad, drop)
                det, t1 = [], []
                for n in days:
                    ak = bocpd_series(blocks[n], ref, sp)
                    det.append(f"{float((ak > 0.5).float().mean()):.2f}")
                    f = first(ak)
                    t1.append(f"{f * blk_s:.1f}" if f >= 0 else "−")
                lines.append(f"| {B['primate']} | {B['seed']} | {src} | {p0:g} | {tau:g} | {'evet' if ok else 'hayır'} | "
                             f"{fa_h:.3f} | {float((a0 > 0.5).float().mean()):.3f} | "
                             f"{fb * blk_s if fb >= 0 else -1:.1f} | {' / '.join(det)} | {' / '.join(t1)} |")
                print(lines[-1], flush=True)
    out = Path(a.out) if a.out else Path(a.blocks[0]).parent / "bocpd_cr.md"
    out.write_text("\n".join(["# BOCPD CR (üretim kodu): kaynak × p_0", ""] + lines), encoding="utf-8")


if __name__ == "__main__":
    main()
