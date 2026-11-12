"""Etiketsiz girdi sapması ölçüsü (KR-046, ARCHITECTURE.md §12.14): kanal EMA ortalaması / varyansının Day-0 train'e
Gauss KL'si. `online_v5.py`'den ayrıldı (KR-049, hakem raporu: tek blok dosya).
"""

from dataclasses import dataclass

import torch


@dataclass
class Drift:
    """Etiketsiz girdi sapması (KR-046): kanal EMA ortalaması / varyansının Day-0 train'e Gauss KL'si.

        m ← m + λ_d·(x − m);   v ← v + λ_d·((x − m_önceki)² − v)          m, v: m_0, v_0'dan başlar
        δ(t) = mean_c ½·[ρ − 1 − ln ρ + (m − m_0)² / v_0'],   ρ = max(v, v_f) / v_0',  v_0' = max(v_0, v_f)
    δ_0: aynı ölçünün Day-0 train'deki ortalaması; sürprize δ/δ_0 girer (Day-0 tabanı 1).
    """
    lam: float
    var_floor: float
    delta0: float = 1.0


class DriftMonitor:
    def __init__(self, m0: torch.Tensor, v0: torch.Tensor, spec: Drift) -> None:
        """m0, v0: [C] Day-0 train kanal ortalaması / varyansı."""
        self.m0, self.lam, self.floor = m0, spec.lam, spec.var_floor
        self.v0f = v0.clamp(min=spec.var_floor)                              # [C]
        self.m = m0.clone()                                                  # [C]
        self.v = v0.clone()

    def step(self, xt: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
        """xt: [C] ham x_norm, mt: [] maske → δ(t): [] (yerinde güncellenir)."""
        dl = self.lam * mt
        d = xt - self.m
        self.m.add_(dl * d)
        self.v.add_(dl * (d * d - self.v))
        rho = self.v.clamp(min=self.floor) / self.v0f
        return 0.5 * (rho - 1.0 - torch.log(rho) + (self.m - self.m0).pow(2) / self.v0f).mean()


@torch.no_grad()
def drift_series(x: torch.Tensor, start: int, stop: int, m0: torch.Tensor, v0: torch.Tensor, spec: Drift,
                 block: int = 1024) -> torch.Tensor:
    """`DriftMonitor`'un [start, stop) boyunca (maske 1) verdiği δ(t) dizisi, vektörleştirilmiş (float64).

    Blok içinde kapalı form EMA: e_{a+k} = q^{k+1}·e_{a−1} + λ·q^k·Σ_{j≤k} q^{−j}·z_{a+j},  q = 1 − λ. → [T]"""
    q, lam = 1.0 - spec.lam, spec.lam
    m_prev, v_prev = m0.double().clone(), v0.double().clone()                # [C]
    m0d, v0f = m0.double(), v0.double().clamp(min=spec.var_floor)
    out = torch.empty(stop - start, dtype=torch.float64, device=x.device)
    for a in range(start, stop, block):
        z = x[a: min(a + block, stop)].double()                              # [n, C]
        qk = (q ** torch.arange(len(z), device=x.device, dtype=torch.float64)).unsqueeze(1)   # [n, 1]
        m = q * qk * m_prev + lam * qk * torch.cumsum(z / qk, 0)
        mp = torch.cat([m_prev.unsqueeze(0), m[:-1]])                        # m_{t−1}
        v = q * qk * v_prev + lam * qk * torch.cumsum((z - mp) ** 2 / qk, 0)
        rho = v.clamp(min=spec.var_floor) / v0f
        out[a - start: a - start + len(z)] = (0.5 * (rho - 1.0 - torch.log(rho) + (m - m0d) ** 2 / v0f)).mean(1)
        m_prev, v_prev = m[-1], v[-1]
    return out


def drift_baseline(x: torch.Tensor, start: int, stop: int, m0: torch.Tensor, v0: torch.Tensor, spec: Drift,
                   stat: str = "mean") -> float:
    """Day-0 train'in [start, stop) aralığında δ (akıştaki gibi m_0, v_0'dan başlar) → δ_0.

    stat = "mean": ortalama; "max": tolerans zarfı (Day-0 boyunca görülen en büyük sapma, KR-046). x: [n, C]"""
    s = drift_series(x, start, stop, m0, v0, spec)
    if stat not in ("mean", "max"):
        raise ValueError(f"bilinmeyen taban istatistiği: {stat}")
    return float(s.mean() if stat == "mean" else s.max())
