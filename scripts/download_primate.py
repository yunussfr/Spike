"""Primat motor tahmini verisini indirir (Zenodo 583331, O'Doherty vd.).

`data.files`: NeuroBench'in 6 oturumu (≈4,3 GB, md5 NeuroBench tablosundan).
`data.tune_files`: V7 ayar oturumları (KR-049; test dışı), md5 `data.md5`'ten.
`data.pre_files` / `data.ext_files`: V8 ÖN ve EK oturumları (KR-051, §12.21), md5 `data.md5`'ten.

Kullanım:
    uv run python scripts/download_primate.py --config configs/v2_base.yaml [--only tune]
    uv run python scripts/download_primate.py --config configs/v2_v8.yaml --only pre     # V8: ÖN (Indy 7, Loco 2)
    uv run python scripts/download_primate.py --config configs/v2_v8.yaml --only ext     # V8: EK test
    uv run python scripts/download_primate.py --config configs/v2_v8.yaml --names indy_20160419_01   # tek dosya

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
    parser.add_argument("--only", choices=["all", "test", "tune", "pre", "ext"], default="all")
    parser.add_argument("--names", nargs="*", default=None, help="yalnız bu oturumlar (config listelerinden birinde olmalı)")
    args = parser.parse_args()
    d = load_config(args.config)["data"]
    root = Path(d["dir"])
    root.mkdir(parents=True, exist_ok=True)

    groups = {"test": "files", "tune": "tune_files", "pre": "pre_files", "ext": "ext_files"}
    keys = list(groups.values()) if args.only == "all" else [groups[args.only]]
    names = list(dict.fromkeys(n for k in keys for n in d.get(k, [])))
    if args.names is not None:
        known = {n for k in groups.values() for n in d.get(k, [])}
        bad = [n for n in args.names if n not in known]
        if bad:
            raise KeyError(f"config listelerinde olmayan oturum: {bad}")
        names = list(args.names)
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
