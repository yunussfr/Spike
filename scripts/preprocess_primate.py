"""İndirilen primat oturumlarını 20 ms ızgaraya işler, önbelleğe yazar ve teşhis özeti çıkarır.

Kullanım:
    uv run python scripts/preprocess_primate.py --config configs/nhp_motor.yaml [--files indy_20170131_02]

Özet `data/primate/cache/summary.json`'a yazılır: kanal sayısı, süre, bölme boyları,
ortalama sayım, train'de hiç ateşlemeyen kanal sayısı, hız istatistikleri.
"""

import argparse
import json
from pathlib import Path

from src.data.primate import _cache_path, build_cache, channel_scale, load_session
from src.utils.config import load_config

EXPECTED_CHANNELS = {"indy": 96, "loco": 192}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    out_path = Path(d["cache_dir"]) / "summary.json"
    summary = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    for name in args.files or d["files"]:
        if not (Path(d["dir"]) / f"{name}.mat").exists():
            print(f"[yok] {name}.mat — önce scripts/download_primate.py")
            continue
        if args.rebuild or not _cache_path(d, name).exists():
            print(f"[işleniyor] {name}", flush=True)
            build_cache(d, name)
        s = load_session(cfg, name)
        tr = s.x[s.split("train")]
        expected = EXPECTED_CHANNELS[name.split("_")[0]]
        info = {
            "D_in": s.D_in,
            "n_steps": len(s.x),
            "minutes": len(s.x) * d["stride"] / 60,
            "n_train": s.val_start, "n_val": s.test_start - s.val_start, "n_test": len(s.x) - s.test_start,
            "mean_count": float(tr.mean()),
            "max_count": float(tr.max()),
            "silent_channels_train": int((tr.sum(dim=0) == 0).sum()),
            "scale_min": float(channel_scale(tr, d["scale_percentile"]).min()),
            "scale_max": float(channel_scale(tr, d["scale_percentile"]).max()),
            "vel_std": [float(v) for v in s.y[s.split("train")].std(dim=0)],
        }
        problems = []
        if s.D_in != expected:
            problems.append(f"kanal sayısı {s.D_in}, beklenen {expected}")
        if min(info["n_train"], info["n_val"], info["n_test"]) <= 0:
            problems.append("boş bölme")
        info["problems"] = problems
        summary[name] = info
        print(f"{name}: {json.dumps(info, ensure_ascii=False)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"özet: {out_path}")


if __name__ == "__main__":
    main()
