"""Primat motor tahmini verisini indirir (Zenodo 583331, O'Doherty vd.).

`data.files`: NeuroBench'in 6 oturumu (≈4,3 GB, md5 NeuroBench tablosundan).
`data.tune_files`: V7 ayar oturumları (KR-049; test dışı), md5 `data.md5`'ten.

Kullanım:
    uv run python scripts/download_primate.py --config configs/v2_base.yaml [--only tune]

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
    parser.add_argument("--only", choices=["all", "test", "tune"], default="all")
    args = parser.parse_args()
    d = load_config(args.config)["data"]
    root = Path(d["dir"])
    root.mkdir(parents=True, exist_ok=True)

    names = ([] if args.only == "tune" else list(d["files"])) + ([] if args.only == "test" else list(d.get("tune_files", [])))
    md5s = {**PrimateReaching.md5s, **{f"{k}.mat": v for k, v in (d.get("md5") or {}).items()}}
    for name in names:
        fname = name if name.endswith(".mat") else name + ".mat"
        if fname not in md5s:
            raise KeyError(f"{fname} için md5 yok (data.md5'e ekle)")
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
