"""İkinci veri seti (Dryad xd2547dkt, TrialData biçimi; Gallego-Carracedo vd. 2022) için yardımcılar (V8, §12.21).

Bu modülde şimdilik yalnız hız türevi var. Dosya → önbellek yükleyicisi (birimlerin elektrot başına toplanması,
denemelerin zaman sırasıyla eklenmesi, 80/10/10 bölmesi deneme sınırında) `scripts/audit_sessions.py` denetimi
alan adlarını ve kutu genişliğini doğruladıktan sonra yazılır (docs/PlanV8.md Adım 0 kapısı).

Hız: konumun **parça içi** merkezî farkı. Parçalar (denemeler) arasında zaman sürekli değildir; fark parça sınırını
aşmaz, parça uçlarında tek yönlü fark kullanılır. NeuroBench de hızı konumun gradyanından alır (np.gradient).
İsteğe bağlı yumuşatma (Gauss, `smooth_ms`) yalnız etiket üretimindedir; parça içinde, kenarlarda yeniden normalize.
"""

import math

import torch


def _gauss_kernel(sigma_steps: float) -> torch.Tensor:
    r = max(1, int(math.ceil(3.0 * sigma_steps)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float64) / sigma_steps) ** 2)
    return k / k.sum()                                                       # [2r + 1]


def _smooth(v: torch.Tensor, sigma_steps: float) -> torch.Tensor:
    """v: [m, D] → [m, D]; kenarlarda yalnız parça içi örneklerle (ağırlıklar yeniden normalize)."""
    k = _gauss_kernel(sigma_steps).to(v)
    r = (len(k) - 1) // 2
    x = v.T.unsqueeze(1)                                                     # [D, 1, m]
    num = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (r, r)), k.view(1, 1, -1))
    ones = torch.nn.functional.pad(torch.ones_like(x[:1]), (r, r))
    den = torch.nn.functional.conv1d(ones, k.view(1, 1, -1))
    return (num / den).squeeze(1).T                                          # [m, D]


def velocity_from_position(pos: torch.Tensor, dt_s: float, seg_starts: list[int] | torch.Tensor,
                           smooth_ms: float = 0.0) -> torch.Tensor:
    """pos: [n, D] konum, dt_s: örnek aralığı (s), seg_starts: parça başlangıç indeksleri (0 dahil, artan).

    → vel: [n, D] (konum birimi / s). Tek örnekli parçada hız 0. smooth_ms > 0: Gauss σ = smooth_ms (parça içi)."""
    if dt_s <= 0:
        raise ValueError("dt_s pozitif olmalı")
    n = pos.shape[0]
    st = [int(s) for s in (seg_starts.tolist() if isinstance(seg_starts, torch.Tensor) else seg_starts)]
    if not st or st[0] != 0 or any(b <= a for a, b in zip(st, st[1:])) or st[-1] >= n:
        raise ValueError(f"seg_starts 0'la başlamalı, kesin artan ve < n olmalı: {st[:5]}…")
    bounds = st + [n]
    pos = pos.to(torch.float64)
    vel = torch.zeros_like(pos)
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = pos[a:b]                                                         # [m, D]
        m = b - a
        if m < 2:
            continue
        v = torch.empty_like(p)
        v[1:-1] = (p[2:] - p[:-2]) / (2.0 * dt_s)
        v[0] = (p[1] - p[0]) / dt_s
        v[-1] = (p[-1] - p[-2]) / dt_s
        if smooth_ms > 0:
            v = _smooth(v, smooth_ms / (1000.0 * dt_s))
        vel[a:b] = v
    return vel.float()
