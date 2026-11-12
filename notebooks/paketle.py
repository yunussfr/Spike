"""Colab paketi: kod ve veri önbelleğini iki zip'e koyar (notebooks/paket/). Drive'a yüklenecek dosyalar bunlardır.

    spike_kod.zip   — src/, experiments/, configs/, scripts/, tests/, pyproject.toml, KOD_SURUMU.json  (birkaç MB)
    spike_veri.zip  — data/primate/cache/ altındaki NeuroBench oturumlarının önbelleği (b4_s4)          (~325 MB)

KOD_SURUMU.json: git commit, çalışma ağacı kirli mi, değişen dosyalar ve kod zip'inin SHA-256'sı. Colab'da git yok;
defter bu dosyayı her koşu klasörüne kopyalar (KV8: kod sürümü kaydı).

Kullanım (depo kökünden):
    uv run python notebooks/paketle.py            # ikisi
    uv run python notebooks/paketle.py --yalniz-kod   # yalnız kod (veri bir kez yüklenir)
    uv run python notebooks/paketle.py --config configs/v2_v8.yaml --roller   # V8: tüm rol oturumlarının yerelde
        # bulunan önbellekleri (+ görev durumu `_task.pt`); eksikleri notebooks/v8_veri.ipynb Colab'da üretir
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


def veri_dosyalari(config: str = "configs/v2_base.yaml", roller: bool = False) -> list[Path]:
    """Rapor eğitiminin okuduğu oturumlar: config'in `primates`'i (test protokolü), 4 ms önbellek.

    roller = True (V8): tüm rol oturumları (ÖN, Day-0, AYAR, TEST, EK); yerelde olmayanlar atlanır ve listelenir
    (Colab'da notebooks/v8_veri.ipynb üretir), görev durumu önbelleği (`_task.pt`) varsa eklenir."""
    sys.path.insert(0, str(KOK))
    from src.utils.config import load_config
    from src.v2.data import check_roles, session_roles

    cfg = load_config(KOK / config)
    d = cfg["data"]
    tag = f"b{round(d['bin_width'] * 1000)}_s{round(d['stride'] * 1000)}_r{d['train_ratio']}"
    if roller:
        check_roles(cfg)
        names = [n for p in cfg["primates"] for n in session_roles(cfg, p).all()]
    else:
        names = [n for ns in cfg["primates"].values() for n in ns]
    files = [KOK / d["cache_dir"] / f"{n}_{tag}.pt" for n in names]
    eksik = [f.name for f in files if not f.exists()]
    if eksik and not roller:
        sys.exit(f"Önbellek eksik: {eksik}. Önce yerelde bir kez yükleyin (ör. pytest ya da indirme betiği).")
    if eksik:
        print(f"[bilgi] yerelde yok, Colab'da üretilecek ({len(eksik)}): {', '.join(eksik)}")
    files = [f for f in files if f.exists()]
    if roller:
        files += [t for f in files if (t := f.with_name(f.stem + "_task.pt")).exists()]
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
    p.add_argument("--config", default="configs/v2_base.yaml", help="veri paketinin oturumları bu config'ten")
    p.add_argument("--roller", action="store_true", help="V8: tüm rol oturumları (eksikler Colab'da üretilir)")
    a = p.parse_args()
    out = Path(a.cikis)
    kod = kod_dosyalari()
    surum = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
             "degisen": (git("status", "--porcelain") or "").splitlines(),
             "paket_zamani": datetime.now().isoformat(timespec="seconds"),
             "dosya_sha256": {f.relative_to(KOK).as_posix(): sha256(f) for f in kod}}
    yaz(out / "spike_kod.zip", kod, {"KOD_SURUMU.json": json.dumps(surum, ensure_ascii=False, indent=1)})
    if not a.yalniz_kod:
        yaz(out / "spike_veri.zip", veri_dosyalari(a.config, a.roller))
    print(f"\nDrive'a yükle: {out}  →  Drive'ım/Spike_colab/")


if __name__ == "__main__":
    main()
