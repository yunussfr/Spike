"""v2 metrikleri: NeuroBench R² ve sütun başına çapraz kontrol (eski `experiments/common.py`'den, KR-049)."""

import torch


def r2_neurobench(pred: torch.Tensor, y: torch.Tensor) -> float:
    """NeuroBench `R2` metriği (x ve y bileşeninin R²'sinin ortalaması). pred, y: [n, 2]."""
    from neurobench.metrics.workload import R2
    m = R2()
    m.reset()
    m(None, pred.float().cpu(), (None, y.float().cpu()))
    return float(m.compute())


def r2_columns(pred: torch.Tensor, y: torch.Tensor) -> list[float]:
    """Kendi R² hesabımız, sütun başına (çapraz kontrol). [n, D] → D değer."""
    pred, y = pred.double(), y.double()
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    return (1.0 - ss_res / ss_tot).tolist()
