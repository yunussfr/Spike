"""YAML config yükleme (K8: tüm hiperparametreler config'ten gelir)."""

import copy
import re
from pathlib import Path
from typing import Any

import torch
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_NUMBER_LIKE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _check_numbers(node: Any, where: str) -> None:
    """PyYAML'ın string okuduğu bilimsel gösterimleri (`1e-4`, `1.0e3`) yakalar."""
    if isinstance(node, dict):
        for k, v in node.items():
            _check_numbers(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_numbers(v, f"{where}[{i}]")
    elif isinstance(node, str) and _NUMBER_LIKE.match(node):
        raise ValueError(f"{where} = '{node}' sayı değil string okundu; `1.0e-4` / `1.0e+3` biçimini kullan")


def load_config(path: str | Path) -> dict[str, Any]:
    """Config'i okur. `base: <dosya>` anahtarı varsa o dosyayı (aynı klasörden) devralır."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    _check_numbers(cfg, path.name)
    base_name = cfg.pop("base", None)
    if base_name is not None:
        cfg = _deep_merge(load_config(path.parent / base_name), cfg)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """`a.b.c=değer` biçimindeki komut satırı geçersiz kılmalarını uygular (değer YAML olarak okunur).

    Sonuç config'i koşu klasörüne kaydedildiği için tekrarlanabilirlik (K8) korunur.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"config'te olmayan anahtar: {key}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_device(name: str) -> torch.device:
    """`auto` → CUDA varsa cuda, yoksa cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
