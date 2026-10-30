"""NeuroBench primat motor tahmini verisini indirir (Zenodo 583331, 6 oturum, ≈4,3 GB).

Kullanım:
    uv run python scripts/download_primate.py --config configs/nhp_motor.yaml

md5 doğrulaması NeuroBench'in `check_integrity`'si ile yapılır; doğru dosya varsa atlanır.
"""

import argparse
from pathlib import Path

from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.datasets.utils import check_integrity, download_url

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    d = load_config(args.config)["data"]
    root = Path(d["dir"])
    root.mkdir(parents=True, exist_ok=True)

    md5s = PrimateReaching.md5s
    for name in d["files"]:
        fname = name if name.endswith(".mat") else name + ".mat"
        path = root / fname
        if check_integrity(str(path), md5s[fname]):
            print(f"[var] {fname}")
            continue
        url = PrimateReaching.url + fname
        print(f"[indiriliyor] {url}", flush=True)
        download_url(url, str(path), md5=md5s[fname])
        if not check_integrity(str(path), md5s[fname]):
            raise RuntimeError(f"md5 tutmadı: {fname}")
        print(f"[tamam] {fname}  {path.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
