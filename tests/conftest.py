"""Paylaşılan test nesneleri. θ(τ) kalibrasyonu ~1 dk sürdüğü için oturum başına bir kez yapılır."""

from pathlib import Path

import pytest

from src.reservoir import LIFReservoir
from src.utils.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
D_IN = 5


@pytest.fixture(scope="session")
def calibrated_reservoir() -> LIFReservoir:
    return LIFReservoir.from_config(CFG, D_in=D_IN, seed=0)
