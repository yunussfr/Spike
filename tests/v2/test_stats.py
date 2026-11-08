"""V6 istatistikleri: kesin Wilcoxon işaretli sıra testi ve Holm düzeltmesi (src/v2/stats.py)."""

import itertools

import numpy as np
import pytest

from src.v2.stats import average_ranks, holm, wilcoxon_signed_rank


def brute_p(d: np.ndarray) -> float:
    """Tüm işaret atamalarını sayarak iki yönlü kesin p (sıfırlar atılmış, ortalama sıralar)."""
    x = d[d != 0]
    r = average_ranks(np.abs(x))
    obs = r[x > 0].sum()
    mid = r.sum() / 2.0
    hits = [abs(sum(rk for rk, s in zip(r, signs) if s) - mid) >= abs(obs - mid) - 1e-9
            for signs in itertools.product([0, 1], repeat=len(x))]
    return float(np.mean(hits))


def test_known_small_values() -> None:
    assert wilcoxon_signed_rank([1, 2, 3, 4, 5]) == (15.0, pytest.approx(0.0625), 5)
    w, p, n = wilcoxon_signed_rank([-1, 2, 3, 4, 5])
    assert w == 14.0 and p == pytest.approx(0.125) and n == 5
    assert wilcoxon_signed_rank([0.0, 0.0]) == (0.0, 1.0, 0)                 # sıfır farklar atılır


def test_ranks_with_ties() -> None:
    assert average_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_with_ties_and_zeros(seed: int) -> None:
    g = np.random.default_rng(seed)
    d = g.integers(-3, 4, size=10).astype(float)                             # eşitler ve sıfırlar bol
    _, p, _ = wilcoxon_signed_rank(d)
    assert p == pytest.approx(brute_p(d), abs=1e-12)


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5] and holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])
