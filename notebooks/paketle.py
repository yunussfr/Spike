"""Colab paketi: kod ve veri önbelleğini iki zip'e koyar (notebooks/paket/). Drive'a yüklenecek dosyalar bunlardır.

    spike_kod.zip   — src/, experiments/, configs/, scripts/, tests/, pyproject.toml, KOD_SURUMU.json  (birkaç MB)
    spike_veri.zip  — data/primate/cache/ altındaki NeuroBench oturumlarının önbelleği (b4_s4)          (~325 MB)

KOD_SURUMU.json: git commit, çalışma ağacı kirli mi, değişen dosyalar ve kod zip'inin SHA-256'sı. Colab'da git yok;
defter bu dosyayı her koşu klasörüne kopyalar (KV8: kod sürümü kaydı).

Kullanım (depo kökünden):
    uv run python notebooks/paketle.py            # ikisi
    uv run python notebooks/paketle.py --yalniz-kod   # yalnız kod (veri bir kez yüklenir)
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parents[1]
KOD = ["src", "experiments", "configs", "scripts", "tests", "pyproject.toml"]
ATLA = {"__pycache__", ".pytest_cache"}


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=KOK, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kod_dosyalari() -> list[Path]:
    out = []
    for k in KOD:
        p = KOK / k
        if p.is_file():
            out.append(p)
        else:
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not (ATLA & set(f.parts)) and f.suffix != ".pyc"]
    return out


def veri_dosyalari() -> list[Path]:
    """Rapor eğitiminin okuduğu oturumlar: configs/v2_base.yaml `primates` (test protokolü), 4 ms önbellek."""
    cfg = yaml.safe_load((KOK / "configs/v2_base.yaml").read_text(encoding="utf-8"))
    d = cfg["data"]
    tag = f"b{round(d['bin_width'] * 1000)}_s{round(d['stride'] * 1000)}_r{d['train_ratio']}"
    names = [n for ns in cfg["primates"].values() for n in ns]
    files = [KOK / d["cache_dir"] / f"{n}_{tag}.pt" for n in names]
    eksik = [f.name for f in files if not f.exists()]
    if eksik:
        sys.exit(f"Önbellek eksik: {eksik}. Önce yerelde bir kez yükleyin (ör. pytest ya da indirme betiği).")
    return files


def yaz(zip_yolu: Path, files: list[Path], ek: dict[str, str] | None = None) -> None:
    zip_yolu.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_yolu, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(KOK).as_posix())
        for ad, icerik in (ek or {}).items():
            z.writestr(ad, icerik)
    print(f"[tamam] {zip_yolu.relative_to(KOK)}  {zip_yolu.stat().st_size / 1e6:.1f} MB  ({len(files)} dosya)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yalniz-kod", action="store_true")
    p.add_argument("--cikis", default=str(KOK / "notebooks" / "paket"))
    a = p.parse_args()
    out = Path(a.cikis)
    kod = kod_dosyalari()
    surum = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
             "degisen": (git("status", "--porcelain") or "").splitlines(),
             "paket_zamani": datetime.now().isoformat(timespec="seconds"),
             "dosya_sha256": {f.relative_to(KOK).as_posix(): sha256(f) for f in kod}}
    yaz(out / "spike_kod.zip", kod, {"KOD_SURUMU.json": json.dumps(surum, ensure_ascii=False, indent=1)})
    if not a.yalniz_kod:
        yaz(out / "spike_veri.zip", veri_dosyalari())
    print(f"\nDrive'a yükle: {out}  →  Drive'ım/Spike_colab/")


if __name__ == "__main__":
    main()
