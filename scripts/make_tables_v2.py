"""Faz V6 / V7 sonuç tabloları: koşu klasörlerini (experiments/v2_v6.py, v2_v7.py) birleştirir; senaryo × koşul
tabloları, eşli karşılaştırmalar, toparlanma süresi ve alt-popülasyon ateşleme oranı (KV7).

İstatistik (§12.9, KR-049): birincil birim **oturumdur**. Seed'ler oturum içinde ortalanır; oturum düzeyinde kesin
Wilcoxon (ulaşılabilir en küçük p ile) ve oturum → seed küme bootstrap'lı %95 GA. (Oturum, seed) birimli Wilcoxon
yalnızca ek bilgidir (seed'ler aynı verinin tekrarıdır, bağımsız değildir). Holm iki ailede: birincil iddialar
(`--claims`) ve tüm karşılaştırmalar. Toparlanma süresi sansürlü: Kaplan–Meier medyanı ve sansürlü sayı.

Kullanım:
    uv run python scripts/make_tables_v2.py --runs runs/<koşu 1> runs/<koşu 2> ... --out runs/v6_summary \\
        [--export results/v6]   # birim satırları + tablo: depoya girecek özet
    # V8 (KR-051): S ve M koşuları birlikte; koşul adları `<koşul>@<omurga>` olur, ek bölüm oturum eşli M − S
    uv run python scripts/make_tables_v2.py --runs runs/<S koşusu> runs/<M koşusu> --backbones --main D7@M \\
        --claims D7@S A@M --extra A@S --title "Faz V8" --export results/v8/<protokol>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v2.stats import (cluster_bootstrap_ci, holm, kaplan_meier_median, min_exact_p,  # noqa: E402
                          session_means, wilcoxon_signed_rank)

N_BOOT, ALPHA, BOOT_SEED = 10000, 0.05, 0                                   # rapor ayarı (veriye bakmaz)

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


def km(rs: list[Row]) -> str:
    """Toparlanma: Kaplan–Meier medyanı (sansürlü / n)."""
    if not rs:
        return "—"
    med = kaplan_meier_median([r["rec_s"] for r in rs], [bool(r["rec_censored"]) for r in rs])
    return f"KM med {med:.1f} ({sum(bool(r['rec_censored']) for r in rs)}/{len(rs)} sansürlü)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--main", default="D")
    p.add_argument("--claims", nargs="*", default=None, help="Holm'a giren karşılaştırmalar (varsayılan Bm_D C_D P_D)")
    p.add_argument("--extra", nargs="*", default=["Pm3", "Pm", "B", "R5", "O"])
    p.add_argument("--title", default="Faz V6")
    p.add_argument("--export", default=None, help="birim satırları (rows.json) ve tablonun depoya kopyalanacağı klasör")
    p.add_argument("--backbones", action="store_true", help="V8: koşul adına omurga kolu eklenir (A@S, D7@M) + M − S bölümü")
    a = p.parse_args()
    runs = [Path(r) for r in a.runs]
    rows = load(runs)
    if a.backbones:
        for r in rows:
            r["cond"] = f"{r['cond']}@{r.get('backbone', 'S')}"
    out = Path(a.out) if a.out else runs[0]
    out.mkdir(parents=True, exist_ok=True)
    D = a.main
    claims = a.claims if a.claims is not None else [f"Bm_{D}", f"C_{D}", f"P_{D}"]
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    by = {(key(r), r["cond"]): r for r in rows}
    split = json.loads((runs[0] / "inputs.json").read_text(encoding="utf-8")).get("split", "?")
    L: list[str] = [f"# {a.title} sonuçları ({split})", "",
                    f"Koşu klasörleri: {', '.join(r.name for r in runs)}; seed {seeds}. Birincil birim oturum "
                    f"(seed ortalaması); (oturum, seed) birimleri ek bilgi.", ""]

    # 1. özet (birim ortalaması; toparlanma Kaplan–Meier)
    L += ["## Özet: koşul × grup (ort ± std; toparlanmada Kaplan–Meier medyanı)", "",
          "| Koşul | " + " | ".join(GROUPS) + " |", "|" + "---|" * (len(GROUPS) + 1)]
    for c in conds:
        cells = []
        for f, m, _ in GROUPS.values():
            rs = [r for r in rows if r["cond"] == c and f(r) and m in r]
            cells.append(km(rs) if m == "rec_s" else ms([r[m] for r in rs]))
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    # 2. eşli karşılaştırmalar: oturum birimli (birincil) ve (oturum, seed) birimli (ek)
    L += ["", f"## Eşli karşılaştırmalar: {D} − X", "",
          f"Fark: oturum ortalamalarının ortalaması (toparlanmada X − {D}: pozitif = {D} daha hızlı). "
          f"**Oturum düzeyi:** {D}'nin iyi olduğu oturum / n, kesin Wilcoxon p, bu n'de ulaşılabilir en küçük p, "
          f"küme bootstrap %95 GA (oturum → seed, {N_BOOT} tekrar). Holm: birincil aile ({', '.join(claims)}) ve tüm aile. "
          f"**Ek (oturum, seed):** kazanma ve p; seed'ler bağımsız olmadığı için güveni olduğundan yüksek gösterir.", "",
          "| Grup | X | fark | %95 GA | oturum kazanma | p (oturum) | en küçük p | p Holm (birincil) | p Holm (tümü) "
          "| kazanma (oturum, seed) | p (oturum, seed) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, (f, m, hi) in GROUPS.items():
        stats = []
        for x in claims + [c for c in a.extra if c not in claims]:
            ks = sorted({key(r) for r in rows if f(r) and r["cond"] == D and (key(r), x) in by and m in r})
            if not ks:
                continue
            sign = 1.0 if hi else -1.0
            units = [(k[1], (by[(k, D)][m] - by[(k, x)][m]) * sign) for k in ks]   # (oturum, fark)
            d = np.array([u[1] for u in units])
            ds = np.array(list(session_means(units).values()))
            _, p_s, _ = wilcoxon_signed_rank(ds)
            _, p_u, _ = wilcoxon_signed_rank(d)
            est, lo, up = cluster_bootstrap_ci(units, N_BOOT, ALPHA, BOOT_SEED)
            stats.append((x, d, ds, p_s, p_u, est, lo, up))
        prim = [t for t in stats if t[0] in claims]
        ph = dict(zip([t[0] for t in prim], holm([t[3] for t in prim])))
        pa = dict(zip([t[0] for t in stats], holm([t[3] for t in stats])))
        for x, d, ds, p_s, p_u, est, lo, up in stats:
            hp = f"{ph[x]:.3f}" if x in ph else "—"
            L.append(f"| {g} | {x} | {est:+.3f} | [{lo:+.3f}, {up:+.3f}] | {int((ds > 0).sum())}/{len(ds)} | {p_s:.3f} | "
                     f"{min_exact_p(len(ds)):.3f} | {hp} | {pa[x]:.3f} | {int((d > 0).sum())}/{len(d)} | {p_u:.4f} |")

    # 2b. V8: omurga kolu M − S, aynı koşul, oturum eşli (birincil V8 iddiası A ve D7'de; Holm tüm koşullarda)
    if a.backbones:
        base = list(dict.fromkeys(c.rsplit("@", 1)[0] for c in conds))
        pairs = [b for b in base if f"{b}@M" in conds and f"{b}@S" in conds]
        L += ["", "## Omurga kolu: M − S (aynı koşul, oturum eşli)", "",
              "M: ÖN + Day-0 omurgası; S: yalnız Day-0 (§12.21). Fark oturum ortalamalarının ortalaması; pozitif = M iyi "
              "(toparlanmada S − M). `etkileşim`: (D − A)_M − (D − A)_S, yani uyumun kazancının omurgaya göre değişimi.", "",
              "| Grup | Koşul | fark | %95 GA | oturum kazanma | p (oturum) | en küçük p | p Holm (grup içi) |",
              "|---|---|---|---|---|---|---|---|"]
        Dm = D.rsplit("@", 1)[0]
        for g, (f, m, hi) in GROUPS.items():
            sign = 1.0 if hi else -1.0
            stats = []
            items = [(b, [(b, "M", 1.0), (b, "S", -1.0)]) for b in pairs]
            if "A" in pairs and Dm in pairs and Dm != "A":
                items.append((f"etkileşim {Dm}−A", [(Dm, "M", 1.0), ("A", "M", -1.0), (Dm, "S", -1.0), ("A", "S", 1.0)]))
            for name, terms in items:
                ks = sorted({key(r) for r in rows if f(r) and m in r}
                            & set.intersection(*[{key(r) for r in rows if r["cond"] == f"{b}@{k}" and f(r) and m in r}
                                                 for b, k, _ in terms]))
                if not ks:
                    continue
                units = [(k_[1], sign * sum(w * by[(k_, f"{b}@{k}")][m] for b, k, w in terms)) for k_ in ks]
                ds = np.array(list(session_means(units).values()))
                _, p_s, _ = wilcoxon_signed_rank(ds)
                est, lo, up = cluster_bootstrap_ci(units, N_BOOT, ALPHA, BOOT_SEED)
                stats.append((name, ds, p_s, est, lo, up))
            hp = holm([s_[2] for s_ in stats]) if stats else []
            for (name, ds, p_s, est, lo, up), h in zip(stats, hp):
                L.append(f"| {g} | {name} | {est:+.3f} | [{lo:+.3f}, {up:+.3f}] | {int((ds > 0).sum())}/{len(ds)} | "
                         f"{p_s:.3f} | {min_exact_p(len(ds)):.3f} | {h:.3f} |")

    # 3. oturum × koşul
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        m = "r2_post" if sc == "electrode" else "r2"
        L += ["", f"## Oturum × koşul: {sc} ({m}, seed ort ± std)", "", "| Oturum | " + " | ".join(conds) + " |",
              "|" + "---|" * (len(conds) + 1)]
        for se in dict.fromkeys(r["session"] for r in rows if r["scenario"] == sc):
            L.append(f"| {se} | " + " | ".join(ms([r[m] for r in rows if r["scenario"] == sc and r["session"] == se
                                                  and r["cond"] == c]) for c in conds) + " |")
        if sc == "electrode":
            L += ["", "| Oturum | " + " | ".join(f"{c} toparlanma s" for c in conds) + " |",
                  "|" + "---|" * (len(conds) + 1)]
            for se in dict.fromkeys(r["session"] for r in rows if r["scenario"] == sc):
                cells = [km([r for r in rows if r["scenario"] == sc and r["session"] == se and r["cond"] == c])
                         for c in conds]
                L.append(f"| {se} | " + " | ".join(cells) + " |")

    # 4. primat ortalaması ve NeuroBench
    L += ["", "## Primat ortalaması (oturum içi, A ve D) ve NeuroBench taban çizgisi", "",
          "| Primat | A | D | NeuroBench SNN | NeuroBench ANN |", "|---|---|---|---|---|"]
    for pm in sorted({r["primate"] for r in rows}):
        f = lambda c: [r["r2"] for r in rows if r["primate"] == pm and r["scenario"] == "within" and r["cond"] == c]   # noqa: E731
        nb = NEUROBENCH.get(pm, {"SNN": float("nan"), "ANN": float("nan")})          # Dryad: NeuroBench tabanı yok
        L.append(f"| {pm} | {ms(f('A'))} | {ms(f(D))} | {nb['SNN']:.3f} | {nb['ANN']:.3f} |")

    # 5. KV7
    L += ["", f"## Ateşleme oranı (KV7, bant {BAND[0]}–{BAND[1]})", "", "| Primat | Koşul | hızlı | yavaş | bant |",
          "|---|---|---|---|---|"]
    for pm in sorted({r["primate"] for r in rows}):
        for c in dict.fromkeys(["A", D] + ([f"A@{k}" for k in ("S", "M")] if a.backbones else [])):
            rs = [r for r in rows if r["primate"] == pm and r["cond"] == c]
            if rs:
                fa, sl = np.mean([r["rate_fast"] for r in rs]), np.mean([r["rate_slow"] for r in rs])
                ok = all(BAND[0] <= v <= BAND[1] for v in (fa, sl))
                L.append(f"| {pm} | {c} | {fa:.4f} | {sl:.4f} | {'✓' if ok else '✗'} |")
    L += ["", "NeuroBench: arXiv 2304.04640 Tablo 4 (4 ms, 75/25 bölme); biz 4 ms, 80/10/10 (KR-035)."]
    text = "\n".join(L) + "\n"
    (out / "table_v6.md").write_text(text, encoding="utf-8")
    if a.export:                                                             # depoya girecek özet (runs/ git dışı)
        ex = Path(a.export)
        ex.mkdir(parents=True, exist_ok=True)
        (ex / "table.md").write_text(text, encoding="utf-8")
        (ex / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
