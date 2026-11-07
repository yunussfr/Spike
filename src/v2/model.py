"""Spike v2 çevrimdışı model: omurga + okuma (Faz V2). Girdi kodlanmış x_norm'dur (SpikeEncoderV2).

Beyincik, duygu uzayı ve çevrimiçi güncellemeler sonraki fazlarda (V4–V5) bağlanır.
"""

from typing import Any

import torch
from torch import nn

from src.v2.backbone import TemporalBackbone
from src.v2.decision import MultiHeadReadout


class V2Model(nn.Module):
    def __init__(self, cfg: dict[str, Any], input_dim: int, out_dim: int, mode: str | None = None) -> None:
        super().__init__()
        self.backbone = TemporalBackbone(input_dim, cfg["backbone"], cfg["time"]["dt_ms"], mode)
        self.readout = MultiHeadReadout.from_config(cfg, cfg["backbone"]["N"], out_dim)

    def reset_state(self, batch_size: int = 1) -> None:
        self.backbone.reset_state(batch_size)
        self.readout.reset_state(batch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mevcut durumdan devam eder (sıfırlamaz, KV6). x: [B, L, C] → v̂: [B, L, D_out], nöron başına oran: [N]"""
        ys = []
        spike_sum = torch.zeros(self.backbone.N, device=x.device)
        for t in range(x.shape[1]):
            s = self.backbone.step(x[:, t])
            y, _ = self.readout.step(s)
            ys.append(y)
            spike_sum = spike_sum + s.mean(0)
        return torch.stack(ys, 1), spike_sum / x.shape[1]


@torch.no_grad()
def stream_predict(model: V2Model, x: torch.Tensor, start: int, stop: int, n_streams: int,
                   burnin: int, chunk: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """[start, stop) aralığının nedensel tahmini, `n_streams` paralel akışla.

    Aralık ardışık parçalara bölünür; her akış kendi parçasından `burnin` adım önce (ya da dizi başında)
    sıfır durumla başlar ve parça boyunca kesintisiz akar. Yalnızca parça içi tahminler döner; burn-in
    skora girmez. x: [n, C] (cihazda) → v̂: [stop − start, D_out], skorlanan adımlarda nöron başına oran: [N]

    chunk > 0 ve CUDA → `chunk` adımlık CUDA Graph parçaları (sonuç aynı; akış sonrası sıfır dolgu skora girmez).
    """
    model.eval()
    L = stop - start
    seg = -(-L // n_streams)                                                 # tavan bölme
    starts = [start + i * seg for i in range(n_streams) if start + i * seg < stop]
    B = len(starts)
    begin = [max(0, s - burnin) for s in starts]
    pre = [s - b for s, b in zip(starts, begin)]
    total = max(p + min(seg, stop - s) for p, s in zip(pre, starts))
    C = x.shape[1]
    lens = [min(seg, stop - s) for s in starts]
    graph = chunk > 0 and x.device.type == "cuda"
    padded = -(-total // chunk) * chunk if graph else total
    # akış başına girdi [B, padded, C] ve skor maskesi [B, padded]; parça sonundan sonrası sıfır dolgu (skora girmez)
    xb = torch.zeros(B, padded, C, device=x.device)
    mask = torch.zeros(B, padded, device=x.device)
    for i, (b, s) in enumerate(zip(begin, starts)):
        xb[i, : s + lens[i] - b] = x[b: s + lens[i]]
        mask[i, pre[i]: pre[i] + lens[i]] = 1.0
    model.reset_state(B)
    rate_sum = torch.zeros(model.backbone.N, device=x.device)

    def step(xt: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
        s = model.backbone.step(xt)                                          # [B, N]
        y, _ = model.readout.step(s)
        rate_sum.add_(mt @ s)                                                # yalnızca skorlanan adımlar
        return y                                                             # [B, D]

    if graph:
        D = model.readout.weight.shape[1]
        sx = torch.zeros(B, chunk, C, device=x.device)
        sm = torch.zeros(B, chunk, device=x.device)
        bb, ro = model.backbone, model.readout
        init = [bb.v.clone(), bb.i_syn.clone(), bb.s.clone(), ro.traces.clone()]
        bb.v, bb.i_syn, bb.s, ro.traces = init
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                        # ısınma (maske 0 → oran değişmez)
            step(sx[:, 0], sm[:, 0])
        torch.cuda.current_stream().wait_stream(side)
        bb.v, bb.i_syn, bb.s, ro.traces = init
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ys = torch.stack([step(sx[:, i], sm[:, i]) for i in range(chunk)], 1)   # [B, chunk, D]
        final = [bb.v, bb.i_syn, bb.s, ro.traces]
        y = torch.empty(B, padded, D, device=x.device)
        for c in range(0, padded, chunk):
            sx.copy_(xb[:, c: c + chunk])
            sm.copy_(mask[:, c: c + chunk])
            g.replay()
            y[:, c: c + chunk] = ys
            for a_, b_ in zip(init, final):
                a_.copy_(b_)
        model.reset_state(B)
    else:
        y = torch.stack([step(xb[:, t], mask[:, t]) for t in range(total)], 1)   # [B, total, D]
    pred = torch.cat([y[i, pre[i]: pre[i] + lens[i]] for i in range(B)])
    return pred, rate_sum / mask.sum()
