"""Faz 0 altyapı testleri."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import load_config
from src.utils.logging import RunLogger
from src.utils.seeding import set_seed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def test_base_config_loads() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg["reservoir"]["N"] == 1000
    assert isinstance(cfg["regulators"]["bounds"]["eta"][0], float)


def test_all_configs_load() -> None:
    for path in CONFIGS.glob("*.yaml"):
        load_config(path)


def test_string_scientific_notation_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1.0e3]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_run_logger_writes_files(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    logger = RunLogger(cfg, 7, "configs/base.yaml")
    logger.log(0, loss=1.5)
    assert (logger.dir / "config.yaml").exists()
    meta = json.loads((logger.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7 and "commit" in meta["git"]
    lines = (logger.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5}
