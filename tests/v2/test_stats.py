"""V6 istatistikleri: kesin Wilcoxon işaretli sıra testi ve Holm düzeltmesi (src/v2/stats.py)."""

import itertools

import numpy as np
import pytest

from src.v2.stats import (average_ranks, cluster_bootstrap_ci, holm, kaplan_meier_median, min_exact_p, session_means,
                          wilcoxon_signed_rank)


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


def test_min_exact_p_matches_wilcoxon() -> None:
    """4 oturumda hepsi aynı yönde olsa bile kesin p 0,125'in altına inemez (hakem raporu, madde 1)."""
    assert min_exact_p(4) == pytest.approx(0.125)
    _, p, _ = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4])
    assert p == pytest.approx(min_exact_p(4))


def test_session_means_average_seeds() -> None:
    m = session_means([("a", 1.0), ("a", 3.0), ("b", 5.0)])
    assert m == {"a": 2.0, "b": 5.0}


def test_cluster_bootstrap_session_level() -> None:
    """Oturum içi seed'ler aynıyken aralık yalnızca oturumlar arası değişkenlikten gelir; tek oturumda sıfır genişlik."""
    est, lo, hi = cluster_bootstrap_ci([("a", 0.2)] * 5, 200, 0.05, 0)
    assert est == lo == hi == pytest.approx(0.2)
    units = [(s, v) for s, v in (("a", 0.0), ("b", 1.0)) for _ in range(5)]
    est, lo, hi = cluster_bootstrap_ci(units, 2000, 0.05, 0)
    assert est == pytest.approx(0.5)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_kaplan_meier_median() -> None:
    assert kaplan_meier_median([1.0, 2.0, 3.0], [False, False, False]) == 2.0
    # sansürlü birim olay sayılmaz ama risk kümesinde kalır: S(2) = 1·(1 − 1/3) = 2/3, S(4) = 2/3·(1 − 1/1) = 0
    assert kaplan_meier_median([2.0, 3.0, 4.0], [False, True, False]) == 4.0
    assert kaplan_meier_median([5.0, 5.0], [True, True]) == float("inf")
