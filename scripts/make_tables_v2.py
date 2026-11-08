"""Faz V6 sonuç tabloları: koşu klasörlerini (experiments/v2_v6.py) birleştirir; senaryo × koşul tabloları, eşli
karşılaştırmalar (kesin Wilcoxon + Holm), toparlanma süresi ve alt-popülasyon ateşleme oranı (KV7).

Birim: (senaryo, oturum, seed). Ana iddia (§12.9): D > B (eşlenmiş), D > C (karıştırılmış), D > P (sensörlü
yeniden ölçekleme + eşlenmiş sabit öğrenme); Holm düzeltmesi bu üç karşılaştırma üzerinde, grup başına.

Kullanım:
    uv run python scripts/make_tables_v2.py --runs runs/<v6 koşu 1> runs/<v6 koşu 2> ... --out runs/v6_summary
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v2.stats import holm, wilcoxon_signed_rank  # noqa: E402

NEUROBENCH = {"indy": {"SNN": 0.593, "ANN": 0.593}, "loco": {"SNN": 0.568, "ANN": 0.558}}   # arXiv 2304.04640, Tablo 4
BAND = (0.02, 0.08)                                                          # KV7 homeostatik bant

Row = dict[str, Any]
GROUPS: dict[str, tuple[Callable[[Row], bool], str, bool]] = {   # ad: (seçici, metrik, büyük mü iyi)
    "Day-k": (lambda r: r["scenario"] == "crossday" and r["day"] > 0, "r2", True),
    "tüm günler": (lambda r: r["scenario"] == "crossday", "r2", True),
    "Day-0": (lambda r: r["scenario"] == "crossday" and r["day"] == 0, "r2", True),
    "oturum içi": (lambda r: r["scenario"] == "within", "r2", True),
    "elektrot: kayıp sonrası R²": (lambda r: r["scenario"] == "electrode", "r2_post", True),
    "elektrot: toparlanma s": (lambda r: r["scenario"] == "electrode", "rec_s", False),
}


def load(runs: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for r in runs:
        rows += json.loads((r / "results.json").read_text(encoding="utf-8"))
    return rows


def key(r: Row) -> tuple:
    return r["scenario"], r["session"], r["seed"]


def ms(v: list[float]) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "—"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--main", default="D")
    p.add_argument("--claims", nargs="*", default=None, help="Holm'a giren karşılaştırmalar (varsayılan Bm_D C_D P_D)")
    p.add_argument("--extra", nargs="*", default=["Pm3", "Pm", "B", "R5", "O"])
    a = p.parse_args()
    runs = [Path(r) for r in a.runs]
    rows = load(runs)
    out = Path(a.out) if a.out else runs[0]
    out.mkdir(parents=True, exist_ok=True)
    D = a.main
    claims = a.claims if a.claims is not None else [f"Bm_{D}", f"C_{D}", f"P_{D}"]
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    by = {(key(r), r["cond"]): r for r in rows}
    split = json.loads((runs[0] / "inputs.json").read_text(encoding="utf-8")).get("split", "?")
    L: list[str] = [f"# Faz V6 sonuçları ({split})", "",
                    f"Koşu klasörleri: {', '.join(str(r) for r in runs)}; seed {seeds}; birim (oturum, seed).", ""]

    # 1. özet
    L += ["## Özet: koşul × grup (ort ± std)", "", "| Koşul | " + " | ".join(GROUPS) + " |", "|" + "---|" * (len(GROUPS) + 1)]
    for c in conds:
        L.append(f"| {c} | " + " | ".join(ms([r[m] for r in rows if r["cond"] == c and f(r) and m in r])
                                         for f, m, _ in GROUPS.values()) + " |")

    # 2. eşli karşılaştırmalar
    L += ["", f"## Eşli karşılaştırmalar: {D} − X", "",
          f"Fark ortalaması (toparlanmada X − {D}: pozitif = {D} daha hızlı), {D}'nin iyi olduğu birim / n, kesin Wilcoxon p "
          f"(iki yönlü); Holm yalnızca {', '.join(claims)} üzerinde, grup başına.", "",
          "| Grup | X | fark | kazanma | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for g, (f, m, hi) in GROUPS.items():
        stats = []
        for x in claims + [c for c in a.extra if c not in claims]:
            ks = sorted({key(r) for r in rows if f(r) and r["cond"] == D and (key(r), x) in by})
            if not ks:
                continue
            d = np.array([(by[(k, D)][m] - by[(k, x)][m]) * (1 if hi else -1) for k in ks])
            _, pv, _ = wilcoxon_signed_rank(d)
            stats.append((x, d, pv))
        ph = dict(zip([s[0] for s in stats if s[0] in claims], holm([s[2] for s in stats if s[0] in claims])))
        for x, d, pv in stats:
            L.append(f"| {g} | {x} | {d.mean():+.3f} | {int((d > 0).sum())}/{len(d)} | {pv:.4f} | "
                     f"{ph[x]:.4f} |" if x in ph else f"| {g} | {x} | {d.mean():+.3f} | {int((d > 0).sum())}/{len(d)} | "
                     f"{pv:.4f} | — |")

    # 3. oturum × koşul
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        m = "r2_post" if sc == "electrode" else "r2"
        L += ["", f"## Oturum × koşul: {sc} ({m}, seed ort ± std)", "", "| Oturum | " + " | ".join(conds) + " |",
              "|" + "---|" * (len(conds) + 1)]
        for se in dict.fromkeys(r["session"] for r in rows if r["scenario"] == sc):
            L.append(f"| {se} | " + " | ".join(ms([r[m] for r in rows if r["scenario"] == sc and r["session"] == se
                                                  and r["cond"] == c]) for c in conds) + " |")
        if sc == "electrode":
            L += ["", "| Oturum | " + " | ".join(f"{c} toparlanma s (sansürlü)" for c in conds) + " |",
                  "|" + "---|" * (len(conds) + 1)]
            for se in dict.fromkeys(r["session"] for r in rows if r["scenario"] == sc):
                cells = []
                for c in conds:
                    rs = [r for r in rows if r["scenario"] == sc and r["session"] == se and r["cond"] == c]
                    cells.append(f"{np.mean([r['rec_s'] for r in rs]):.1f} ({sum(r['rec_censored'] for r in rs)}/{len(rs)})"
                                 if rs else "—")
                L.append(f"| {se} | " + " | ".join(cells) + " |")

    # 4. primat ortalaması ve NeuroBench
    L += ["", "## Primat ortalaması (oturum içi, A ve D) ve NeuroBench taban çizgisi", "",
          "| Primat | A | D | NeuroBench SNN | NeuroBench ANN |", "|---|---|---|---|---|"]
    for pm in sorted({r["primate"] for r in rows}):
        f = lambda c: [r["r2"] for r in rows if r["primate"] == pm and r["scenario"] == "within" and r["cond"] == c]   # noqa: E731
        L.append(f"| {pm} | {ms(f('A'))} | {ms(f(D))} | {NEUROBENCH[pm]['SNN']:.3f} | {NEUROBENCH[pm]['ANN']:.3f} |")

    # 5. KV7
    L += ["", f"## Ateşleme oranı (KV7, bant {BAND[0]}–{BAND[1]})", "", "| Primat | Koşul | hızlı | yavaş | bant |",
          "|---|---|---|---|---|"]
    for pm in sorted({r["primate"] for r in rows}):
        for c in ("A", D):
            rs = [r for r in rows if r["primate"] == pm and r["cond"] == c]
            if rs:
                fa, sl = np.mean([r["rate_fast"] for r in rs]), np.mean([r["rate_slow"] for r in rs])
                ok = all(BAND[0] <= v <= BAND[1] for v in (fa, sl))
                L.append(f"| {pm} | {c} | {fa:.4f} | {sl:.4f} | {'✓' if ok else '✗'} |")
    L += ["", "NeuroBench: arXiv 2304.04640 Tablo 4 (4 ms, 75/25 bölme); biz 4 ms, 80/10/10 (KR-035)."]
    text = "\n".join(L) + "\n"
    (out / "table_v6.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
