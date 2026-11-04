"""Faz V0 hesap bütçesi: bir eğitim batch'inin (ileri + BPTT + AdamW) süresi → epoch ve protokol süresi.

Omurga henüz yazılmadı (Faz V2); burada ölçüm için en küçük prototip adımlar kullanılır:
  plif     — I = α·I + W_in·x + W_rec·s;  V = β·V + (1−β)·I − ϑ·s;  s = H(V − ϑ) (hızlı-sigmoid surrogate)
  gru_loop — adım adım `nn.GRUCell` (spiking_gru / tau_gru'nun Python döngüsü maliyetine yakın)
Pencere sayıları gerçek önbelleklerden (train dilimi) hesaplanır; Δt = 8 ms için 8 ms önbelleği gerekmez,
pencere sayısı süreye bağlıdır (aynı train süresi, aynı pencere süresi).

Kullanım:
    uv run python experiments/v2_bench.py --config configs/v2_base.yaml
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.data import load_v2_session, ms_to_steps, train_window_starts  # noqa: E402


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        (v,) = ctx.saved_tensors
        return g / (1.0 + 10.0 * v.abs()) ** 2


class Proto(nn.Module):
    def __init__(self, mode: str, C: int, N: int) -> None:
        super().__init__()
        self.mode, self.N = mode, N
        if mode == "plif":
            self.w_in = nn.Linear(C, N)
            self.w_rec = nn.Linear(N, N, bias=False)
            self.lt_mem = nn.Parameter(torch.zeros(N))
            self.lt_syn = nn.Parameter(torch.zeros(N))
        else:
            self.cell = nn.GRUCell(C, N)
        self.out = nn.Linear(N, 2)

    def forward(self, x: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, C] → v̂: [B, L, 2], ortalama aktivite: [B, L, N]"""
        B, L, _ = x.shape
        h = x.new_zeros(B, self.N)
        ys, ss = [], []
        if self.mode == "plif":
            beta = torch.exp(-dt / (4.0 + 396.0 * torch.sigmoid(self.lt_mem)))
            alpha = torch.exp(-dt / (4.0 + 96.0 * torch.sigmoid(self.lt_syn)))
            I = torch.zeros_like(h)
            s = torch.zeros_like(h)
            for t in range(L):
                I = alpha * I + self.w_in(x[:, t]) + self.w_rec(s)
                h = beta * h + (1 - beta) * I - s
                s = FastSigmoidSpike.apply(h - 1.0)
                ss.append(s)
                ys.append(self.out(s))
        else:
            for t in range(L):
                h = self.cell(x[:, t], h)
                ss.append(h)
                ys.append(self.out(h))
        return torch.stack(ys, 1), torch.stack(ss, 1)


def time_batch(mode: str, C: int, L: int, bc: dict[str, Any], dt: float, device: torch.device) -> dict[str, float]:
    torch.manual_seed(0)
    model = Proto(mode, C, bc["N"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = (torch.rand(bc["batch"], L, C, device=device) < 0.05).float()
    y = torch.randn(bc["batch"], L, 2, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(bc["warmup"] + bc["iters"]):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred, s = model(x, dt)
        loss = torch.sqrt(((pred - y) ** 2).mean() + 1e-6) + ((s.mean((0, 1)) - 0.05) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= bc["warmup"]:
            times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return {"sec_per_batch": sum(times) / len(times), "peak_gib": mem}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    bc = cfg["bench"]
    device = resolve_device(cfg["experiment"]["device"])
    logger = RunLogger(cfg, cfg["experiment"]["seed"], a.config)

    # train pencere sayıları (4 ms önbelleğinden; süre Δt'den bağımsız)
    windows: dict[str, int] = {}
    C_of: dict[str, int] = {}
    for primate, names in cfg["primates"].items():
        for name in names:
            s = load_v2_session(cfg, name)
            windows[name] = len(train_window_starts(s, cfg))
            C_of[primate] = s.D_in

    rows = []
    for dt in bc["dt_ms"]:
        L = ms_to_steps(cfg["windows"]["length_ms"], dt)
        for mode in bc["modes"]:
            for primate, names in cfg["primates"].items():
                r = time_batch(mode, C_of[primate], L, bc, dt, device)
                pre_b = math.ceil(sum(windows[n] for n in names) / bc["batch"])
                ft_b = [math.ceil(windows[n] / bc["batch"]) for n in names]
                pre_h = pre_b * r["sec_per_batch"] * bc["epochs_pretrain"] / 3600
                ft_h = [b * r["sec_per_batch"] * bc["epochs_finetune"] / 3600 for b in ft_b]
                row = {"dt_ms": dt, "mode": mode, "primate": primate, "C": C_of[primate], "L": L, **r,
                       "batches_pretrain": pre_b, "pretrain_h": pre_h, "finetune_h": ft_h,
                       "one_seed_h": pre_h + sum(ft_h)}
                rows.append(row)
                logger.log(len(rows), **row)
                print(json.dumps(row), flush=True)

    lines = ["| Δt | mod | primat | s/batch | GiB | ön-eğitim (s/saat) | ince ayar, 3 oturum (saat) | 1 seed toplam (saat) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dt_ms']:.0f} ms | {r['mode']} | {r['primate']} | {r['sec_per_batch']:.2f} | {r['peak_gib']:.1f} | "
                     f"{r['pretrain_h']:.1f} | {sum(r['finetune_h']):.1f} | {r['one_seed_h']:.1f} |")
    lines += ["", f"batch {bc['batch']}, N = {bc['N']}, ön-eğitim {bc['epochs_pretrain']} / ince ayar {bc['epochs_finetune']} epoch, "
              f"cihaz {device}. Pencere sayıları: {windows}"]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
