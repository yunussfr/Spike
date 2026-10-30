"""Faz 3 sonuç tablosu: 6 oturum × A/B (ort ± std, 5 seed), B − A, ridge, taban çizgileri.

NeuroBench taban çizgileri makaleden alınır (arXiv 2304.04640, Tablo 4); makale R²'yi maymun
başına 3 oturumun ortalaması olarak verir. Oturum başına değer yayımlanmadığı için tabloda
yalnızca maymun ortalaması satırında karşılaştırılır.

Kullanım:
    uv run python scripts/make_tables.py --run runs/<nhp_motor koşusu>
"""

import argparse
import json
from pathlib import Path

import numpy as np

NEUROBENCH = {  # arXiv 2304.04640, Tablo 4 (R², 3 oturum ortalaması)
    "indy": {"SNN": 0.593, "ANN": 0.593},
    "loco": {"SNN": 0.568, "ANN": 0.558},
}


def fmt(v: list[float]) -> str:
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    configs = [k[3:] for k in rows[0] if k.startswith("r2_") and not k.endswith(("_val", "_own"))
               and k not in ("r2_ridge", "r2_mean_baseline")]
    sessions = list(dict.fromkeys(r["session"] for r in rows))

    head = ["Oturum"] + configs + (["B − A"] if {"A", "B"} <= set(configs) else []) + \
        ["Ridge (üst sınır)", "Ortalama tabanı", "Ateşleme oranı", "K7"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    per_animal: dict[str, dict[str, list[float]]] = {}
    for name in sessions:
        rs = [r for r in rows if r["session"] == name]
        cells = [f"{name} ({len(rs)} seed)"] + [fmt([r[f"r2_{c}"] for r in rs]) for c in configs]
        if {"A", "B"} <= set(configs):
            cells.append(fmt([r["r2_B"] - r["r2_A"] for r in rs]))
        cells += [fmt([r["r2_ridge"] for r in rs]) if "r2_ridge" in rs[0] else "—",
                  fmt([r["r2_mean_baseline"] for r in rs]),
                  f"{np.mean([r['firing_rate'] for r in rs]):.4f}",
                  "✓" if all(r["k7_ok"] for r in rs) else "✗"]
        lines.append("| " + " | ".join(cells) + " |")
        animal = name.split("_")[0]
        for c in configs + ["ridge"]:
            key = f"r2_{c}"
            if key in rs[0]:                                                   # seed ortalaması, oturum başına bir değer
                per_animal.setdefault(animal, {}).setdefault(c, []).append(float(np.mean([r[key] for r in rs])))

    lines += ["", "**Maymun ortalaması (3 oturum) ve NeuroBench taban çizgisi**", ""]
    head2 = ["Maymun"] + configs + ["Ridge", "NeuroBench SNN", "NeuroBench ANN"]
    lines += ["| " + " | ".join(head2) + " |", "|" + "---|" * len(head2)]
    for animal, d in per_animal.items():
        cells = [animal] + [f"{np.mean(d[c]):.3f}" for c in configs] + \
            [f"{np.mean(d['ridge']):.3f}" if "ridge" in d else "—",
             f"{NEUROBENCH[animal]['SNN']:.3f}", f"{NEUROBENCH[animal]['ANN']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Kaynak: NeuroBench, arXiv 2304.04640, Tablo 4. Fark: NeuroBench 4 ms adımla ve 75/25 bölmeyle "
              "raporlar; biz 20 ms adım ve NeuroBench kodunun 80/10/10 bölmesini kullanırız (KR-012, KR-018). "
              "K7 ✗: θ(τ) düzgün rastgele girdiyle kalibre edildiği için gerçek veride oran ρ/2'nin altında (KR-014)."]
    text = "\n".join(lines)
    (run / "table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
