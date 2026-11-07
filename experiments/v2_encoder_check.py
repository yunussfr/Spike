"""Faz V1 çıkış kriteri: kodlanmış girdi (SpikeEncoderV2) üzerinde Wiener R², ham sayım Wiener'inden kötü değil.

Aynı Wiener (experiments/v2_ceiling.py: nedensel kutu toplamları, λ ve geçmiş val'de), aynı bölme; tek fark
girdi: ham sayım x ya da x_norm. Kodlayıcı istatistikleri yalnızca train'den. Test R² raporlanır.

Kullanım:
    uv run python experiments/v2_encoder_check.py --config configs/v2_ceiling.yaml [--files ...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.v2_ceiling import wiener  # noqa: E402
from src.data.primate import Session  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.data import check_time, load_v2_session  # noqa: E402
from src.v2.encoder import SpikeEncoderV2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = "v2_encoder_check"
    dt = check_time(cfg)
    logger = RunLogger(cfg, cfg["experiment"]["seed"], a.config)
    lines = ["| Oturum | ham sayım (test, seçilen geçmiş) | x_norm (test, seçilen geçmiş) | fark |", "|---|---|---|---|"]
    for name in a.files or cfg["data"]["files"]:
        s = load_v2_session(cfg, name)
        enc = SpikeEncoderV2.from_config(cfg, s.D_in)
        enc.fit(s.x[s.split("train")])
        se = Session(name, enc.normalize(s.x), s.y, s.val_start, s.test_start)
        res = {}
        for kind, sess in (("raw", s), ("encoded", se)):
            rows = wiener(cfg, sess, dt)
            best = max(rows, key=lambda r: r["r2_val"])
            res[kind] = best
            logger.log(0, session=name, kind=kind, rows=rows)
        print(json.dumps({"session": name, **res}), flush=True)
        d = res["encoded"]["r2_test"] - res["raw"]["r2_test"]
        lines.append(f"| {name} | {res['raw']['r2_test']:.3f} ({res['raw']['history_ms']:.0f} ms) | "
                     f"{res['encoded']['r2_test']:.3f} ({res['encoded']['history_ms']:.0f} ms) | {d:+.3f} |")
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
