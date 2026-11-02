"""NeuroBench primat motor tahmini verisi (Zenodo 583331) → tek sürekli zaman dizisi.

NeuroBench `PrimateReaching` ham spike'ları 4 ms çözünürlükte açar ve `bin_width` penceresinde
sayar. Biz örnekleri `stride` aralığında (varsayılan 20 ms = bir ortam adımı, T·dt) alırız:
  x: [n, D_in]  kanal başına spike sayımı (indy 96, loco 192 kanal)
  y: [n, 2]     parmak ucu hızı (cursor_pos'un gradyanı; x, y bileşeni)
Bölme NeuroBench'inkiyle aynıdır: hedef segmentlerinin ilk %80'i train, sonraki yarısı val,
kalanı test; zaman sırası korunur. Rezervuar tüm dosyayı tek dizi olarak görür (K6).

Ön işleme (ARCHITECTURE.md §3.1, `range: train_per_channel`): her kanal, train bölmesindeki
sayımlarının `scale_percentile` yüzdeliğine bölünür; böylece kodlayıcının seviye aralığı
[0, 1] olur. Ölçek yalnızca train'den hesaplanır, val/test'e bakılmaz.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SAMPLING_RATE = 0.004                                                       # s, NeuroBench'in ham çözünürlüğü


@dataclass
class Session:
    name: str
    x: torch.Tensor                       # [n, D_in] sayımlar (ham ya da ölçekli)
    y: torch.Tensor                       # [n, 2] hız
    val_start: int                        # train = [0, val_start)
    test_start: int                       # val = [val_start, test_start), test = [test_start, n)

    @property
    def D_in(self) -> int:
        return self.x.shape[1]

    def split(self, part: str) -> slice:
        return {"train": slice(0, self.val_start), "val": slice(self.val_start, self.test_start),
                "test": slice(self.test_start, len(self.x))}[part]


def _cache_path(d: dict[str, Any], name: str) -> Path:
    tag = f"b{round(d['bin_width'] * 1000)}_s{round(d['stride'] * 1000)}_r{d['train_ratio']}"
    return Path(d["cache_dir"]) / f"{name}_{tag}.pt"


def build_cache(d: dict[str, Any], name: str) -> Path:
    """NeuroBench yükleyicisiyle .mat'ı işler ve sonucu önbelleğe yazar (yavaş adım, bir kez)."""
    from neurobench.datasets.primate_reaching import PrimateReaching

    ds = PrimateReaching(file_path=d["dir"], filename=name, num_steps=1, train_ratio=d["train_ratio"],
                         bin_width=d["bin_width"], stride=d["stride"], download=False)
    step = int(round(d["stride"] / SAMPLING_RATE))
    n4 = min(ds.samples.shape[1], ds.labels.shape[1])                       # NeuroBench aynı indeksle eşler; fazla sütun atılır
    idx = np.arange(0, n4, step)                                             # ortak 20 ms ızgarası
    x = ds.samples[:, idx].T.contiguous().to(torch.uint8 if ds.samples.max() < 256 else torch.int16)
    y = ds.labels[:, idx].T.contiguous().float()
    # NeuroBench bölmesi segment indeksleriyle tanımlı; sınırları kendi ızgaramıza çeviririz.
    if not (max(ds.ind_train) < min(ds.ind_val) and max(ds.ind_val) < min(ds.ind_test)):
        raise ValueError(f"{name}: train/val/test zaman sırasında ardışık değil")
    val_start = int(np.searchsorted(idx, min(ds.ind_val)))
    test_start = int(np.searchsorted(idx, min(ds.ind_test)))
    path = _cache_path(d, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "val_start": val_start, "test_start": test_start,
                "n_raw": n4, "first_train": int(min(ds.ind_train))}, path)
    return path


def load_session(cfg: dict[str, Any], name: str) -> Session:
    """Önbellekten oturumu yükler (yoksa üretir); gecikmeyi uygular. x ham sayımdır."""
    d = cfg["data"]
    path = _cache_path(d, name)
    if not path.exists():
        build_cache(d, name)
    c = torch.load(path)
    x, y = c["x"].float(), c["y"]
    delay = d["biological_delay"]
    if delay > 0:                                                            # x(t) → y(t + delay)
        x, y = x[:-delay], y[delay:]
    return Session(name=name, x=x, y=y, val_start=c["val_start"], test_start=c["test_start"])


def channel_scale(x_train: torch.Tensor, percentile: float) -> torch.Tensor:
    """Kanal başına ölçek: train sayımlarının yüzdeliği; hiç ateşlemeyen kanalda max ya da 1. [n, D] → [D]"""
    q = torch.from_numpy(np.percentile(x_train.numpy(), percentile, axis=0)).float()  # [D] (torch.quantile büyük girdide hata verir)
    mx = x_train.max(dim=0).values
    return torch.where(q > 0, q, torch.where(mx > 0, mx, torch.ones_like(mx)))


def prepare(cfg: dict[str, Any], session: Session) -> tuple[dict[str, Any], Session]:
    """Kodlayıcı aralığını çözer. `train_per_channel`: x kanal başına ölçeklenir, aralık [0, 1]."""
    cfg = copy.deepcopy(cfg)
    rng = cfg["encoder"]["level"]["range"]
    x_tr = session.x[session.split("train")]
    if rng == "train_per_channel":
        scale = channel_scale(x_tr, cfg["data"]["scale_percentile"])
        session = Session(session.name, session.x / scale, session.y, session.val_start, session.test_start)
        cfg["encoder"]["level"]["range"] = [0.0, 1.0]
    elif rng == "train":
        cfg["encoder"]["level"]["range"] = [float(x_tr.min()), float(x_tr.max())]
    return cfg, session
