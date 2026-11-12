"""GRU tavan tabanı (BPTT), 4 ms ızgarada (Faz V0, KR-037). v1 `experiments/ceiling.py`'den taşındı (KR-049).

Ham sayımlar (train'de standartlaştırılmış) üzerinde BPTT ile eğitilen GRU; gizli boyut ve erken durdurma val'de.
Değerlendirmede GRU tüm dosyayı tek akış olarak, nedensel biçimde görür.
"""

import copy
import time
from typing import Any

import torch
from torch import nn

from src.data.primate import Session
from src.utils.seeding import set_seed
from src.v2.metrics import r2_neurobench

STREAM_CHUNK = 16384                      # değerlendirme akışının parça uzunluğu (yalnızca cuDNN sınırı; sonucu değiştirmez)


class GRUDecoder(nn.Module):
    def __init__(self, D_in: int, hidden: int, num_layers: int, input_dropout: float, D_out: int) -> None:
        super().__init__()
        self.drop = nn.Dropout(input_dropout)
        self.gru = nn.GRU(D_in, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, D_out)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, D_in] → ŷ: [B, L, D_out], h: [layers, B, hidden]"""
        z, h = self.gru(self.drop(x), h)
        return self.out(z), h


@torch.no_grad()
def stream_predict(model: GRUDecoder, x: torch.Tensor, stop: int) -> torch.Tensor:
    """Dizinin başından `stop`'a kadar tek akış (sıfır başlangıç durumu). x: [n, D] → [stop, D_out]

    cuDNN çok uzun diziyi tek çağrıda desteklemediği için parça parça koşulur; gizli durum taşınır (sonuç aynı).
    """
    model.eval()
    h, ys = None, []
    for i in range(0, stop, STREAM_CHUNK):
        y, h = model(x[None, i: min(i + STREAM_CHUNK, stop)], h)
        ys.append(y[0])
    return torch.cat(ys)


def train_gru(cfg: dict[str, Any], s: Session, seed: int, hidden: int,
              device: torch.device) -> tuple[GRUDecoder, float, int]:
    """TBPTT: train ardışık `streams` akışa bölünür, gizli durum akış içinde taşınır. → model (en iyi val), val R², tur"""
    gc = cfg["ceiling"]["gru"]
    tr, va = s.split("train"), s.split("val")
    mu_x, sd_x = s.x[tr].mean(0), s.x[tr].std(0).clamp(min=1e-6)
    mu_y, sd_y = s.y[tr].mean(0), s.y[tr].std(0)
    xs = ((s.x - mu_x) / sd_x).to(device)                                   # [n, D]
    ys = ((s.y - mu_y) / sd_y).to(device)                                   # [n, 2]

    set_seed(seed)
    model = GRUDecoder(s.D_in, hidden, gc["num_layers"], gc["input_dropout"], s.y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    B, L = gc["streams"], gc["chunk"]
    n_tr = tr.stop

    best, best_state, best_ep, bad = -float("inf"), None, 0, 0
    for ep in range(gc["max_epochs"]):
        model.train()
        off = int(torch.randint(0, L, (1,), generator=gen))                  # parça sınırları her turda kayar
        per = (n_tr - off) // B
        n_ch = per // L
        xb = xs[off: off + B * per].view(B, per, -1)                         # [B, per, D]
        yb = ys[off: off + B * per].view(B, per, -1)
        h = None
        for c in range(n_ch):
            sl = slice(c * L, (c + 1) * L)
            pred, h = model(xb[:, sl], h)
            loss = ((pred - yb[:, sl]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gc["grad_clip"])
            opt.step()
            h = h.detach()
        p = stream_predict(model, xs, va.stop)[va.start: va.stop] * sd_y.to(device) + mu_y.to(device)
        r2 = r2_neurobench(p, s.y[va])
        if r2 > best:
            best, best_ep, bad = r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= gc["patience"]:
                break
    model.load_state_dict(best_state)
    model.norm = (mu_x, sd_x, mu_y, sd_y)                                    # değerlendirme için
    return model, best, best_ep


def gru(cfg: dict[str, Any], s: Session, seed: int, device: torch.device) -> dict[str, Any]:
    """Gizli boyut val'de seçilir; seçilen modelin test R²'si raporlanır."""
    t0 = time.time()
    cands = []
    for hidden in cfg["ceiling"]["gru"]["hidden_grid"]:
        model, r2v, ep = train_gru(cfg, s, seed, hidden, device)
        cands.append((r2v, hidden, ep, model))
    r2v, hidden, ep, model = max(cands, key=lambda c: c[0])
    mu_x, sd_x, mu_y, sd_y = model.norm
    xs = ((s.x - mu_x) / sd_x).to(device)
    te = s.split("test")
    p = stream_predict(model, xs, te.stop)[te.start: te.stop].cpu() * sd_y + mu_y
    return {"hidden": hidden, "best_epoch": ep, "r2_val": r2v, "r2_test": r2_neurobench(p, s.y[te]),
            "r2_val_by_hidden": {str(c[1]): c[0] for c in cands}, "sec": time.time() - t0}


# --- (c) ve NLMS: Faz 3 tam koşusundan ---------------------------------------------------
