"""v2 istatistikleri (ARCHITECTURE.md §12.9): kesin Wilcoxon işaretli sıra testi, Holm düzeltmesi ve V7'de
(KR-049, hakem raporu) oturum birimli analiz: seed ortalaması, oturum → seed küme bootstrap'ı, Kaplan–Meier medyanı.

scipy kurulu değil (kurulum kullanıcıda); n ≤ ~60 için kesin dağılım dinamik programlamayla hesaplanır.
Sıfır farklar atılır (Wilcoxon "wilcox" kuralı); eşit |fark|'lara ortalama sıra verilir ve kesin dağılım bu sıra
yapısına koşullu hesaplanır (sıralar ikiyle çarpılıp tamsayıya çevrilir).
"""

from collections import defaultdict
from typing import Hashable

import numpy as np


def average_ranks(a: np.ndarray) -> np.ndarray:
    """a: [n] → 1'den başlayan sıralar, eşitlerde ortalama sıra. [n]"""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(d: np.ndarray | list[float]) -> tuple[float, float, int]:
    """Eşli farklar d: [n] → (W⁺, iki yönlü kesin p, sıfır olmayan fark sayısı n')."""
    x = np.asarray(d, dtype=np.float64)
    x = x[x != 0.0]
    n = len(x)
    if n == 0:
        return 0.0, 1.0, 0
    r = average_ranks(np.abs(x))
    w_plus = float(r[x > 0].sum())
    r2 = np.rint(2.0 * r).astype(np.int64)                                   # ortalama sıralar yarım tamsayı olabilir
    total = int(r2.sum())
    dist = np.zeros(total + 1, dtype=np.float64)                             # 2·W⁺ değerlerinin işaret sayısı
    dist[0] = 1.0
    for k in r2:
        nxt = dist.copy()
        nxt[k:] += dist[: total + 1 - k]
        dist = nxt
    prob = dist / dist.sum()
    w2, mid = 2.0 * w_plus, total / 2.0
    dev = np.abs(np.arange(total + 1) - mid)
    p = float(prob[dev >= abs(w2 - mid) - 1e-9].sum())
    return w_plus, min(1.0, p), n


def holm(p: list[float]) -> list[float]:
    """Holm–Bonferroni düzeltilmiş p değerleri (girdi sırasıyla)."""
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m, dtype=np.float64)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj.tolist()


def min_exact_p(n: int) -> float:
    """n eşli farkta (bağ yok) kesin iki yönlü Wilcoxon'un ulaşabileceği en küçük p: 2 / 2ⁿ (n = 4 → 0,125)."""
    return min(1.0, 2.0 / 2.0 ** n) if n > 0 else 1.0


def session_means(units: list[tuple[Hashable, float]]) -> dict[Hashable, float]:
    """(oturum, değer) birimleri (her seed bir birim) → oturum başına seed ortalaması."""
    acc: dict[Hashable, list[float]] = defaultdict(list)
    for s, v in units:
        acc[s].append(float(v))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def cluster_bootstrap_ci(units: list[tuple[Hashable, float]], n_boot: int, alpha: float,
                         seed: int) -> tuple[float, float, float]:
    """Hiyerarşik bootstrap: önce oturumlar, sonra seçilen her oturumun seed'leri yerine koyarak çekilir.

    units: (oturum, fark) birimleri → (oturum-ortalaması-ortalama, alt sınır, üst sınır); yüzdelik %(1 − alpha) aralık."""
    by: dict[Hashable, np.ndarray] = {}
    acc: dict[Hashable, list[float]] = defaultdict(list)
    for s, v in units:
        acc[s].append(float(v))
    for s, v in acc.items():
        by[s] = np.asarray(v, dtype=np.float64)
    keys = list(by)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    est = float(np.mean([by[k].mean() for k in keys]))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        stats[b] = np.mean([rng.choice(by[keys[i]], len(by[keys[i]]), replace=True).mean() for i in pick])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return est, float(lo), float(hi)


def kaplan_meier_median(times: list[float], censored: list[bool]) -> float:
    """Sağdan sansürlü süreler → Kaplan–Meier medyanı (S(t) ≤ 0,5 olduğu ilk olay zamanı); ulaşılmazsa inf.

    Sansürlü birim: olay (toparlanma) gözlem penceresinde görülmedi; süresi yalnızca alt sınırdır."""
    t = np.asarray(times, dtype=np.float64)
    c = np.asarray(censored, dtype=bool)
    if len(t) == 0:
        return float("nan")
    s = 1.0
    for u in np.unique(t[~c]):                                               # olay zamanları, artan
        at_risk = int((t >= u).sum())
        events = int(((t == u) & ~c).sum())
        s *= 1.0 - events / at_risk
        if s <= 0.5:
            return float(u)
    return float("inf")
