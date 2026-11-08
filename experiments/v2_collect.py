"""V3 rapor koşusu seed/aşama başına ayrı süreçlerde koşunca (CUDA Graph çöküşü, KR-042) sonuçları birleştirir.

Her klasörün results.json'u okunur; (primat, seed, aşama, oturum) başına zamanca en son satır tutulur.
    uv run python experiments/v2_collect.py --config configs/v2_pretrain.yaml --runs runs/<...> [runs/<...> ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None, help="tablonun yazılacağı .md dosyası")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    h = load_config(a.config)["backbone"]["homeostasis"]
    rows: dict[tuple, dict] = {}
    for d in sorted(a.runs):                                                 # klasör adı zaman damgalı → sıralı
        for r in json.loads((Path(d) / "results.json").read_text(encoding="utf-8")):
            rows[(r["primate"], r["seed"], r["stage"], r["session"])] = r
    groups: dict[tuple[str, str], list[dict]] = {}
    for (pr, seed, stg, n), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], kv[0][1])):
        groups.setdefault((n, stg), []).append(r)
    lines = ["| Oturum | Aşama | test R² | val R² | n seed | oran hızlı / yavaş (val) |", "|---|---|---|---|---|---|"]
    for (n, stg), rs in groups.items():
        t = np.array([r["test_r2"] for r in rs])
        v = np.array([r["val_r2"] for r in rs])
        rf = np.mean([r["val_rate_fast"] for r in rs])
        rsl = np.mean([r["val_rate_slow"] for r in rs])
        bad = "" if h["f_min"] <= rf <= h["f_max"] and h["f_min"] <= rsl <= h["f_max"] else " ✗"
        lines.append(f"| {n} | {stg} | {t.mean():.3f} ± {t.std():.3f} | {v.mean():.3f} ± {v.std():.3f} | {len(rs)} "
                     f"| {rf:.3f} / {rsl:.3f}{bad} |")
    seeds = sorted({k[1] for k in rows})
    lines += ["", f"Seed {seeds}. day0: Day-0'da eğitilen donuk model; Day-k satırları günler arası (v2-A). "
              f"✗: alt-popülasyon oranı [{h['f_min']}, {h['f_max']}] dışında (KV7)."]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
