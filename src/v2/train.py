"""Çevrimdışı eğitim döngüsü (ARCHITECTURE.md §12.8): train pencereleri, AdamW + cosine, RMSE + λ·L_homeo.

Model seçimi val R² ile (her epoch akış değerlendirmesi, `stream_predict`); test bu modülde hiç görülmez.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from src.data.primate import Session
from src.v2.backbone import homeostatic_loss
from src.v2.data import gather_windows, ms_to_steps, train_window_starts
from src.v2.encoder import SpikeEncoderV2
from src.v2.model import V2Model, stream_predict

Metric = Callable[[torch.Tensor, torch.Tensor], float]                    # (tahmin [n, D], hedef [n, D]) → skor


@dataclass
class Prepared:
    """Bir oturumun cihazdaki kodlanmış girdisi ve standartlaştırılmış hedefi."""
    session: Session
    x: torch.Tensor                       # [n, C] x_norm
    y: torch.Tensor                       # [n, D] standart hedef
    y_mu: torch.Tensor                    # [D]
    y_sd: torch.Tensor                    # [D]
    starts: torch.Tensor                  # [W] train pencere başlangıçları


def prepare_session(cfg: dict[str, Any], s: Session, device: torch.device) -> Prepared:
    tr = s.split("train")
    enc = SpikeEncoderV2.from_config(cfg, s.D_in)
    enc.fit(s.x[tr])                                                         # KV6: yalnızca train
    if cfg["train"]["target_norm"]:
        mu, sd = s.y[tr].mean(0), s.y[tr].std(0)
    else:
        mu, sd = torch.zeros(s.y.shape[1]), torch.ones(s.y.shape[1])
    return Prepared(s, enc.normalize(s.x).to(device), ((s.y - mu) / sd).to(device), mu.to(device), sd.to(device),
                    train_window_starts(s, cfg))


@dataclass
class TrainResult:
    best_val: float
    best_epoch: int
    epochs_run: int
    sec_per_epoch: float
    state: dict[str, torch.Tensor]
    history: list[dict[str, float]] = field(default_factory=list)


def evaluate(model: V2Model, p: Prepared, part: str, cfg: dict[str, Any], metric: Metric) -> dict[str, float]:
    """Dilimin akış tahmini → skor (ham hız ölçeğinde) ve alt-popülasyon ateşleme oranları."""
    t = cfg["train"]
    sl = p.session.split(part)
    burnin = ms_to_steps(t["eval_burnin_ms"], cfg["time"]["dt_ms"])
    pred, rate = stream_predict(model, p.x, sl.start, sl.stop, t["eval_streams"], burnin)
    y_hat = pred * p.y_sd + p.y_mu
    y = p.y[sl] * p.y_sd + p.y_mu
    h = model.backbone.half
    return {"r2": metric(y_hat.cpu(), y.cpu()), "rate_fast": float(rate[:h].mean()), "rate_slow": float(rate[h:].mean()),
            "rate_min_neuron": float(rate.min()), "rate_max_neuron": float(rate.max())}


class TrainStep:
    """Bir eğitim adımı (ileri + BPTT + kırpma + AdamW). CUDA'da `cuda_graph: true` iken tüm adım bir kez
    CUDA Graph olarak yakalanır ve her batch'te yeniden oynatılır (Windows'ta çekirdek başlatma yükü ~25× azalır).

    Yakalanan grafik sabit batch boyutu ister; bu yüzden epoch'un son eksik batch'i atlanır (`drop_last`).
    Dropout her oynatmada yeni rastgele maske üretir (grafik-güvenli RNG).
    """

    def __init__(self, model: V2Model, cfg: dict[str, Any], epochs: int, B: int, L: int, C: int, D: int,
                 device: torch.device) -> None:
        t, h = cfg["train"], cfg["backbone"]["homeostasis"]
        self.model, self.B, self.clip = model, B, t["grad_clip"]
        self.f_min, self.f_max = h["f_min"], h["f_max"]
        self.lam = h["loss_weight"] if model.backbone.spiking else 0.0      # KV7: tau_gru muaf
        self.graph = bool(t["cuda_graph"]) and device.type == "cuda"
        lr = torch.tensor(t["lr"], device=device) if self.graph else t["lr"]
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"], capturable=self.graph)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.sx = torch.zeros(B, L, C, device=device)                       # [B, L, C] sabit girdi tamponu
        self.sy = torch.zeros(B, L, D, device=device)
        self.g: torch.cuda.CUDAGraph | None = None
        self.n_warm = t["graph_warmup"]

    def _eager(self) -> torch.Tensor:
        self.model.reset_state(self.B)                                       # her pencere yeni dizi (KV6)
        pred, rate = self.model(self.sx)
        loss = torch.sqrt(((pred - self.sy) ** 2).mean() + 1e-6)
        if self.lam:
            loss = loss + self.lam * homeostatic_loss(rate, self.f_min, self.f_max)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        return loss.detach()

    def __call__(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        self.sx.copy_(xb)
        self.sy.copy_(yb)
        if not self.graph:
            self.opt.zero_grad(set_to_none=True)
            return float(self._eager())
        if self.g is None and self.n_warm > 0:                               # ısınma adımları gerçek eğitim adımıdır
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.opt.zero_grad(set_to_none=True)
                loss = self._eager()
            torch.cuda.current_stream().wait_stream(s)
            self.n_warm -= 1
            return float(loss)
        if self.g is None:
            self.opt.zero_grad(set_to_none=True)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.static_loss = self._eager()
        self.g.replay()
        return float(self.static_loss)


def train_model(model: V2Model, p: Prepared, cfg: dict[str, Any], seed: int, metric: Metric,
                epochs: int | None = None, log: Callable[[dict[str, float]], None] | None = None) -> TrainResult:
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    L = ms_to_steps(cfg["windows"]["length_ms"], cfg["time"]["dt_ms"])
    B = min(t["batch"], len(p.starts))
    trainer = TrainStep(model, cfg, epochs, B, L, p.x.shape[1], p.y.shape[1], p.x.device)
    gen = torch.Generator().manual_seed(seed)

    best, best_ep, best_state, bad, hist, sec = -float("inf"), -1, None, 0, [], []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = p.starts[torch.randperm(len(p.starts), generator=gen)]
        tot, nb = 0.0, 0
        for i in range(0, len(order) - B + 1, B):                            # drop_last: sabit batch (TrainStep)
            st = order[i: i + B].to(p.x.device)
            tot += trainer(gather_windows(p.x, st, L), gather_windows(p.y, st, L))   # [B, L, C], [B, L, D]
            nb += 1
        trainer.sched.step()
        train_sec = time.time() - t0
        ev = evaluate(model, p, "val", cfg, metric)
        row = {"epoch": ep, "loss": tot / nb, "val_r2": ev["r2"], "train_sec": train_sec,
               "eval_sec": time.time() - t0 - train_sec, **{f"val_{k}": v for k, v in ev.items() if k != "r2"}}
        hist.append(row)
        sec.append(train_sec)
        if log:
            log(row)
        if ev["r2"] > best:
            best, best_ep, bad = ev["r2"], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    model.load_state_dict(best_state)
    return TrainResult(best, best_ep, len(hist), sum(sec) / len(sec), best_state, hist)
