"""Oturum önbelleklerini üretir (4 ms sayım + hız `<ad>_<etiket>.pt`, görev durumu `_task.pt`); .mat diskte olmalı.

V8 Colab akışında (notebooks/v8_veri.ipynb) indirilen her oturum için bir kez çalışır; yerelde de kullanılabilir.

Kullanım:
    uv run python scripts/build_cache.py --config configs/v2_v8.yaml --names indy_20160419_01 [...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.primate import load_task_state  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.v2.data import load_v2_session  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--names", nargs="+", required=True)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(a.config)
    for n in a.names:
        s = load_v2_session(cfg, n)                                          # önbellek yoksa NeuroBench'le üretir
        pos, _ = load_task_state(cfg, n)                                     # imleç / hedef (ReFIT, §12.15)
        print(f"[tamam] {n}: x {tuple(s.x.shape)}, y {tuple(s.y.shape)}, görev {tuple(pos.shape)}, "
              f"val {s.val_start}, test {s.test_start}", flush=True)


if __name__ == "__main__":
    main()
