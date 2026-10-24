"""Koşu kaydı (K8): config, seed, git hash ve metrikler tek bir klasörde."""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def git_info() -> dict[str, Any]:
    """Mevcut commit hash'i ve çalışma ağacının kirli olup olmadığı."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),   # henüz commit yoksa None
        "dirty": bool(status) if status is not None else None,
    }


class RunLogger:
    """`<log_dir>/<zaman>_<ad>_<seed etiketi>/` klasörünü açar ve içine yazar.

    Klasör içeriği:
      config.yaml   — birleştirilmiş (base dahil) config
      meta.json     — seed, git bilgisi, zaman, sürümler
      metrics.jsonl — her satır bir `log()` çağrısı
    """

    def __init__(self, cfg: dict[str, Any], seed: int | list[int], config_path: str | Path) -> None:
        exp = cfg["experiment"]
        seed_tag = f"seed{seed}" if isinstance(seed, int) else "seeds" + "-".join(map(str, seed))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(exp["log_dir"]) / f"{stamp}_{exp['name']}_{seed_tag}"

        self.dir = base
        suffix = 1
        while self.dir.exists():
            self.dir = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
        self.dir.mkdir(parents=True)

        with (self.dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        self.save_json("meta.json", {
            "seed": seed,
            "config_path": str(config_path),
            "git": git_info(),
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "torch": torch.__version__,
        })
        self._metrics_path = self.dir / "metrics.jsonl"
        self._metrics_path.touch()

    def log(self, step: int, **metrics: float) -> None:
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def save_json(self, name: str, obj: Any) -> None:
        with (self.dir / name).open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def path(self, name: str) -> Path:
        return self.dir / name
