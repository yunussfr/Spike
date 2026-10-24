"""Deney giriş noktası.

Kullanım:
    uv run python experiments/run.py --config configs/base.yaml [--seed 3]

Faz 0: config okuyup boş bir döngü çalıştırır ve log klasörü oluşturur.
"""

import argparse

from src.utils.config import load_config, resolve_device
from src.utils.logging import RunLogger
from src.utils.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="config'teki experiment.seed yerine")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, seed, args.config)

    for step in range(cfg["experiment"]["n_steps"]):
        logger.log(step, placeholder=0.0)

    print(f"device={device}  log={logger.dir}")


if __name__ == "__main__":
    main()
