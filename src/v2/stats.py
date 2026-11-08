"""V6 istatistikleri: kesin Wilcoxon işaretli sıra testi ve Holm düzeltmesi (ARCHITECTURE.md §12.9).

scipy kurulu değil (kurulum kullanıcıda); n ≤ ~60 için kesin dağılım dinamik programlamayla hesaplanır.
Sıfır farklar atılır (Wilcoxon "wilcox" kuralı); eşit |fark|'lara ortalama sıra verilir ve kesin dağılım bu sıra
yapısına koşullu hesaplanır (sıralar ikiyle çarpılıp tamsayıya çevrilir).
"""

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
