"""Spike v2 veri katmanı (ARCHITECTURE.md §12.8, docs/PhasesV2.md Faz V0).

v1 yükleyicisini (`src/data/primate.py`) Δt = `time.dt_ms` ızgarasında kullanır: bir veri adımı = bir omurga
adımı (KV6). Bölme v1 ile aynı NeuroBench 80/10/10 bölmesidir (KR-035). Burada:
  - zaman ayarlarının tutarlılığı (dt_ms = bin_width = stride),
  - çevrimdışı eğitim pencereleri (yalnızca train diliminden),
  - günler arası protokol (primat başına tarih sırası, ilk oturum Day-0).
"""

from datetime import date
from typing import Any

import torch

from src.data.primate import Session, load_session


def check_time(cfg: dict[str, Any]) -> float:
    """dt_ms, data.bin_width ve data.stride aynı olmalı (bir veri adımı = bir omurga adımı). → dt_ms"""
    dt = float(cfg["time"]["dt_ms"])
    d = cfg["data"]
    for key in ("bin_width", "stride"):
        if abs(d[key] * 1000.0 - dt) > 1e-9:
            raise ValueError(f"data.{key} = {d[key]} s, time.dt_ms = {dt} ms ile uyuşmuyor")
    return dt


def ms_to_steps(ms: float, dt_ms: float) -> int:
    """Süreyi adım sayısına çevirir; tam katı değilse hata (sessiz yuvarlama yok)."""
    n = ms / dt_ms
    if abs(n - round(n)) > 1e-9 or round(n) < 1:
        raise ValueError(f"{ms} ms, Δt = {dt_ms} ms'nin pozitif tam katı değil")
    return int(round(n))


def load_v2_session(cfg: dict[str, Any], name: str) -> Session:
    """Oturumu Δt ızgarasında yükler (önbellek yoksa NeuroBench'ten üretir). x: [n, C] ham sayım, y: [n, 2]."""
    check_time(cfg)
    return load_session(cfg, name)


def train_window_starts(session: Session, cfg: dict[str, Any]) -> torch.Tensor:
    """Train dilimindeki pencerelerin başlangıç indeksleri. Pencere [s, s + L) tamamen train içinde kalır. → [W]"""
    dt = check_time(cfg)
    L = ms_to_steps(cfg["windows"]["length_ms"], dt)
    S = ms_to_steps(cfg["windows"]["stride_ms"], dt)
    last = session.val_start - L                                             # son geçerli başlangıç
    if last < 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.arange(0, last + 1, S, dtype=torch.long)


def gather_windows(x: torch.Tensor, starts: torch.Tensor, L: int) -> torch.Tensor:
    """x: [n, D], starts: [W] → [W, L, D] (kopya)."""
    idx = starts[:, None] + torch.arange(L, device=starts.device)[None, :]   # [W, L]
    return x[idx]


def session_date(name: str) -> date:
    """`indy_20160622_01` → 2016-06-22."""
    s = name.split("_")[1]
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def check_tune_disjoint(cfg: dict[str, Any]) -> None:
    """Ayar oturumları (`tune_primates`) test oturumlarından (`primates`) ayrı olmalı (KV6, KR-049).

    Ayar listesinin Day-0'ı yoktur: Day-0 modeli test protokolündekiyle aynıdır (primates[p][0], yalnız train'i)."""
    tune = cfg.get("tune_primates") or {}
    test = {n for names in cfg["primates"].values() for n in names}
    clash = sorted({n for names in tune.values() for n in names} & test)
    if clash:
        raise ValueError(f"ayar oturumları test oturumlarıyla çakışıyor: {clash}")
    unknown = [p for p in tune if p not in cfg["primates"]]
    if unknown:
        raise ValueError(f"tune_primates'te primates'te olmayan primat: {unknown}")


def crossday_sessions(cfg: dict[str, Any], primate: str, tune: bool = False) -> tuple[str, list[str]]:
    """Primatın oturumları tarih sırasıyla; ilki Day-0, kalanlar Day-k. → (day0, [day1, …])

    tune = True: Day-0 aynı, Day-k'ler ayar oturumları (`tune_primates`, test dışı; KR-049)."""
    names = list(cfg["primates"][primate])
    dates = [session_date(n) for n in names]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError(f"primates.{primate} tarih sırasında değil: {names}")
    missing = [n for n in names if n not in cfg["data"]["files"]]
    if missing:
        raise ValueError(f"primates.{primate} data.files'ta olmayan oturum içeriyor: {missing}")
    if not tune:
        return names[0], names[1:]
    check_tune_disjoint(cfg)
    tn = list((cfg.get("tune_primates") or {}).get(primate, []))
    td = [session_date(n) for n in tn]
    if not tn or td != sorted(td) or min(td) <= dates[0]:
        raise ValueError(f"tune_primates.{primate} boş, tarih sırasında değil ya da Day-0'dan önce: {tn}")
    missing = [n for n in tn if n not in cfg["data"].get("tune_files", [])]
    if missing:
        raise ValueError(f"tune_primates.{primate} data.tune_files'ta olmayan oturum içeriyor: {missing}")
    return names[0], tn
