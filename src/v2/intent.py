"""Sönümlü ReFIT niyet etiketi (ARCHITECTURE.md §12.15, KR-049).

Gerçek bir BCI'da el hızı etiketi yoktur; ReFIT (Gilja vd. 2012) kullanıcının hedefe gitmek istediğini varsayar ve
dekoderin hızını hedefe döndürür. Burada hız büyüklüğü dekoderin kendi çıktısından gelir, yönü görev durumundan:

    Δ = g − p,  ρ = ‖Δ‖,  û = Δ / (ρ + ε),  σ = 1 − exp(−ρ / r_d)          (hedefe yaklaştıkça sönüm)
    dd(t) = 0 (ρ ≤ r_on, hedefte)  |  σ·û (aksi halde)                     [n, 2]  — v̂'den bağımsız, önceden hesaplanır
    y_int,fiz(t) = ‖v̂_fiz(t)‖ · dd(t);   y_int = (y_int,fiz − μ_y) / σ_y   akışta, v̂ skorlandıktan sonra
    m_int(t) = 0  hedef değişiminden sonraki ilk t_rt boyunca (tepki süresi)

Hız etiketi (y) hiç kullanılmaz (sızıntı yok). Öğrenmez; `update` yok.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Intent:
    """Akışa verilen niyet girdisi (cihazda)."""
    dd: torch.Tensor                      # [n, D] σ·û (hedefte 0)
    m: torch.Tensor                       # [n] tepki süresi maskesi
    mu: torch.Tensor                      # [D] hedef ortalaması (train)
    sd: torch.Tensor                      # [D] hedef std'si (train)


def refit_direction(p: torch.Tensor, g: torch.Tensor, r_d: float, r_on: float, eps: float) -> torch.Tensor:
    """p, g: [n, D] imleç ve hedef konumu (mm) → dd: [n, D] sönümlü birim yön (hedefte 0)."""
    delta = g - p                                                            # [n, D]
    rho = delta.norm(dim=1, keepdim=True)                                    # [n, 1]
    u = delta / (rho + eps)
    sigma = 1.0 - torch.exp(-rho / r_d)
    return torch.where(rho > r_on, sigma * u, torch.zeros_like(u))


def reaction_mask(g: torch.Tensor, rt_steps: int) -> torch.Tensor:
    """g: [n, D] hedef konumu → m: [n]; hedef değiştikten sonraki ilk rt_steps adımda (değişim adımı dahil) 0.

    İlk adım değişim sayılmaz (akış öncesi bilinmiyor)."""
    n = g.shape[0]
    change = torch.zeros(n, dtype=torch.bool, device=g.device)
    change[1:] = (g[1:] != g[:-1]).any(dim=1)
    idx = torch.arange(n, device=g.device)
    last = torch.where(change, idx, torch.full_like(idx, -10 ** 9))
    last = torch.cummax(last, dim=0).values                                  # son değişimin indeksi
    return ((idx - last) >= rt_steps).float()


def refit_label(v: torch.Tensor, dd: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """v: [..., D] standart ölçekte dekoder çıktısı, dd: [..., D] → y_int: [..., D] standart ölçek."""
    speed = (v * sd + mu).norm(dim=-1, keepdim=True)                         # fiziksel hız büyüklüğü
    return (speed * dd - mu) / sd


def make_intent(cfg: dict[str, Any], pos: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor,
                device: torch.device) -> Intent:
    """Config `refit.*` ile oturumun niyet girdisi. pos, target: [n, D] (mm)."""
    r = cfg["refit"]
    rt = int(round(r["reaction_ms"] / cfg["time"]["dt_ms"]))
    dd = refit_direction(pos.float(), target.float(), r["damp_radius_mm"], r["on_target_mm"], r["eps"])
    return Intent(dd.to(device), reaction_mask(target.float(), rt).to(device), mu.to(device), sd.to(device))


def label_quality(y: torch.Tensor, dd: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Tanı (§12.15): gerçek hız büyüklüğüyle kurulan niyet etiketi ‖y‖·dd; y: [n, D] fiziksel → [n, D]."""
    return y.norm(dim=1, keepdim=True) * dd * m.unsqueeze(1)
