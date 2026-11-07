"""Aşama 3: kesintisiz akışta çevrimiçi çapalı NLMS (ARCHITECTURE.md §12.7, §12.8; KV1, KV2, KV6).

Omurga donuk (yalnızca isteğe bağlı homeostatik ϑ değişir). Aynı akış üzerinde E koşul (η değerleri) birlikte
yürütülür: omurga ve sızıntılı izler koşullar arasında ortaktır (g = 1 iken girdileri aynıdır), her koşulun kendi
başlık ağırlıkları vardır. Etiket d adım gecikmeyle gelir (`label_delay`, KR-041). Her adımda (prequential):
  1. v̂_t = (1/K) Σ_k W^(k)·r^(k)_t + b^(k)            → kaydedilir (y_t ve sonrası kullanılmadı)
  2. etiket y_{t−d} gelir: e = y_{t−d} − v̂_{t−d}      (v̂_{t−d}: o anki ağırlıklarla yapılan tahmin)
  3. W^(k) ← (1 − α)·W^(k) + α·W_0^(k) + η·e·r^(k)_{t−d}ᵀ / (‖r^(k)_{t−d}‖² + ε)
d = 0 → anında etiket. Anında etiket 4 ms'de dejenere: y_{t−1}'i tekrarlamak bile R² ≈ 0,999 (KR-041).
Bias b^(k) donuk kalır (spesifikasyondaki gibi; §12.10 nokta 5, KR-040). Hata standart hedef ölçeğindedir.
Burn-in adımlarında (maske 0) omurga ve izler ilerler; güncelleme, homeostaz ve oran sayımı yoktur.

CUDA'da akış `chunk` adımlık parçalar halinde CUDA Graph olarak yakalanıp oynatılır (çekirdek başlatma yükü);
sonuç adım adım (eager) yolla aynıdır (`tests/v2/test_online.py`).
"""

from dataclasses import dataclass
from typing import Any

import torch

from src.v2.data import ms_to_steps
from src.v2.model import V2Model


@dataclass
class StreamResult:
    pred: torch.Tensor                    # [E, L, D] standart ölçekte tahminler (skorlanan adımlar)
    rate: torch.Tensor                    # [N] skorlanan adımlarda nöron başına oran
    dw_rel: torch.Tensor                  # [E] akış sonunda ‖W − W_0‖ / ‖W_0‖
    theta: torch.Tensor                   # [N] akış sonu ϑ


class _Stream:
    def __init__(self, model: V2Model, etas: list[float], anchor_decay: float, eps: float, homeostasis: bool,
                 delay: int, device: torch.device) -> None:
        ro, bb = model.readout, model.backbone
        self.m, self.ro, self.bb = model, ro, bb
        E = len(etas)
        self.W0 = ro.w_anchor                                                # [K, D, N]
        self.W = self.W0.unsqueeze(0).repeat(E, 1, 1, 1).clone()             # [E, K, D, N]
        self.b = ro.b_anchor.unsqueeze(0)                                    # [1, K, D] donuk
        self.eta = torch.tensor(etas, device=device).view(E, 1, 1, 1)
        self.alpha, self.eps, self.homeo, self.d = anchor_decay, eps, homeostasis, delay
        self.rate = torch.zeros(bb.N, device=device)
        if delay > 0:                                                        # halka tamponlar: t−d anındaki iz, tahmin, etiket, maske
            D = self.W0.shape[1]
            self.buf_r = torch.zeros(delay, ro.K, bb.N, device=device)       # [d, K, N]
            self.buf_v = torch.zeros(delay, E, D, device=device)             # [d, E, D]
            self.buf_y = torch.zeros(delay, D, device=device)                # [d, D]
            self.buf_m = torch.zeros(delay, device=device)                   # [d]

    def step(self, xt: torch.Tensor, yt: torch.Tensor, mt: torch.Tensor, j: int) -> torch.Tensor:
        """xt: [1, C], yt: [D], mt: [] skor maskesi (0 burn-in / dolgu, 1 skorlanan), j = (t − başlangıç) mod d → v̂: [E, D]"""
        bb, ro = self.bb, self.ro
        s = bb.step(xt)                                                      # [1, N]
        ro.traces = ro.kappa * ro.traces + (1.0 - ro.kappa) * s.unsqueeze(0)  # [K, 1, N]
        r = ro.traces[:, 0]                                                  # [K, N]
        v = (torch.einsum("ekdn,kn->ekd", self.W, r) + self.b).mean(1)       # [E, D]  1. tahmin
        if self.d > 0:                                                       # 2. gecikmeli etiket: t−d anının kaydı
            r_old, m_old = self.buf_r[j].clone(), self.buf_m[j].clone()
            err = (self.buf_y[j] - self.buf_v[j]) * m_old                    # [E, D]
            self.buf_r[j] = r
            self.buf_v[j] = v
            self.buf_y[j] = yt
            self.buf_m[j] = mt
        else:
            r_old, m_old = r, mt
            err = (yt.unsqueeze(0) - v) * mt
        st = r_old / (r_old.pow(2).sum(-1, keepdim=True) + self.eps)         # [K, N]
        self.W.sub_((self.alpha * m_old) * (self.W - self.W0))               # 3. çapaya çekme (etiket gelince)
        self.W.add_(self.eta * torch.einsum("ed,kn->ekdn", err, st))         #    NLMS
        self.rate.add_(mt * s[0])
        if self.homeo:                                                       # KV1: yerel eşik adaptasyonu (maskeli)
            d = (1.0 - bb.rate_decay) * mt
            bb.rate_ema.mul_(1.0 - d).add_(d * s[0])
            bb.theta.add_(bb.homeo_lr * mt * (bb.rate_ema - bb.target_rate)).clamp_(bb.theta_min, bb.theta_max)
        return v


def _state(m: V2Model) -> list[torch.Tensor]:
    return [m.backbone.v, m.backbone.i_syn, m.backbone.s, m.readout.traces]


def _set_state(m: V2Model, st: list[torch.Tensor]) -> None:
    m.backbone.v, m.backbone.i_syn, m.backbone.s, m.readout.traces = st


@torch.no_grad()
def stream_adapt(model: V2Model, x: torch.Tensor, y: torch.Tensor, start: int, stop: int, burnin: int,
                 etas: list[float], anchor_decay: float, eps: float, homeostasis: bool,
                 chunk: int = 0, label_delay: int = 0) -> StreamResult:
    """x: [n, C], y: [n, D] standart hedef (cihazda). [start, stop) skorlanır; akış max(0, start − burnin)'de başlar.

    chunk > 0 ve CUDA → CUDA Graph parçaları (chunk, label_delay'in katı olmalı); aksi halde adım adım.
    """
    model.eval()
    if not bool(model.readout.anchored):
        raise RuntimeError("çapa kaydedilmemiş: önce readout.save_anchor() (Aşama 2)")
    d = label_delay
    if chunk > 0 and d > 0 and chunk % d:
        raise ValueError(f"graph_chunk ({chunk}) etiket gecikmesinin ({d}) katı olmalı")
    bb, dev = model.backbone, x.device
    theta0, rate_ema0 = bb.theta.clone(), bb.rate_ema.clone()
    st = _Stream(model, etas, anchor_decay, eps, homeostasis, d, dev)
    model.reset_state(1)
    begin = max(0, start - burnin)
    total = stop - begin
    E, D = len(etas), y.shape[1]
    out = torch.empty(E, total, D, device=dev)
    mask = torch.zeros(total, device=dev)
    mask[start - begin:] = 1.0

    if chunk > 0 and dev.type == "cuda":
        sx = torch.zeros(chunk, x.shape[1], device=dev)                      # [L, C] sabit tamponlar
        sy = torch.zeros(chunk, D, device=dev)
        sm = torch.zeros(chunk, device=dev)
        init = [t.clone() for t in _state(model)]                            # parça başı durumu (sabit adres)
        _set_state(model, init)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                        # ısınma (maske 0: durum dışında etkisiz)
            st.step(sx[0:1], sy[0], sm[0], 0)
        torch.cuda.current_stream().wait_stream(side)
        _set_state(model, init)                                              # ısınma durumu atılır; init sıfır kaldı
        if d > 0:
            for buf in (st.buf_r, st.buf_v, st.buf_y, st.buf_m):
                buf.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            vs = torch.stack([st.step(sx[i: i + 1], sy[i], sm[i], i % d if d else 0) for i in range(chunk)], 1)   # [E, L, D]
        final = _state(model)
        n_full = total // chunk * chunk
        for c in range(0, n_full, chunk):
            sx.copy_(x[begin + c: begin + c + chunk])
            sy.copy_(y[begin + c: begin + c + chunk])
            sm.copy_(mask[c: c + chunk])
            g.replay()
            out[:, c: c + chunk] = vs
            for a_, b_ in zip(init, final):                                  # sonraki parça bu parçanın sonundan
                a_.copy_(b_)
        rest = range(n_full, total)                                          # eksik son parça adım adım (dolgu adımı yok)
    else:
        rest = range(total)
    for i in rest:
        out[:, i] = st.step(x[begin + i: begin + i + 1], y[begin + i], mask[i], i % d if d else 0)

    W, W0 = st.W, st.W0
    dw = (W - W0).flatten(1).norm(dim=1) / W0.flatten().norm()
    theta_end = bb.theta.clone()
    bb.theta.copy_(theta0)                                                   # akış sonrası model aynı kalır
    bb.rate_ema.copy_(rate_ema0)
    model.reset_state(1)
    return StreamResult(out[:, start - begin:], st.rate / (stop - start), dw, theta_end)


def stream_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    o = cfg["online"]
    return {"anchor_decay": o["anchor_decay"], "eps": o["nlms_eps"], "chunk": o["graph_chunk"],
            "label_delay": ms_to_steps(o["label_delay_ms"], cfg["time"]["dt_ms"]) if o["label_delay_ms"] else 0}
