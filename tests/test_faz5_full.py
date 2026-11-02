"""Faz 5 çıkış koşusu betiği (experiments/faz5_full.py): akışlar, K4/K5 sızıntı kontrolleri, özet."""

import copy

import numpy as np
import torch

from experiments.faz5_full import TauBins, run_streams, summarize, table
from src.affect import PARAMS
from src.data.primate import Session
from src.reservoir import LIFReservoir
from tests.conftest import CFG

D_IN, D_OUT, N_TR, N_VA, N_TE = 4, 2, 60, 20, 20
CONFIGS = ["A", "B", "B+Z", "C:perturb+EMR"]


def setup() -> tuple[dict, Session, LIFReservoir]:
    cfg = copy.deepcopy(CFG)
    cfg["reservoir"]["N"] = 100
    cfg["nhp"] = {"washout": 5}
    cfg["faz5"] = {"configs": CONFIGS, "epochs": 2, "max_steps": None, "trace_every": 3}
    cfg["regulators"]["perturbation"].update(M=5, noise_std=1.0)
    g = torch.Generator().manual_seed(0)
    n = N_TR + N_VA + N_TE
    s = Session("sentetik", torch.rand(n, D_IN, generator=g), torch.randn(n, D_OUT, generator=g),
                N_TR, N_TR + N_VA)
    res = LIFReservoir.from_config(cfg, D_in=D_IN, seed=0, calibrate=False)
    return cfg, s, res


def test_run_streams_outputs() -> None:
    cfg, s, res = setup()
    W_in, W_rec = res.W_in.clone(), res.W_rec.clone()
    out = run_streams(cfg, s, res, seed=0, configs=CONFIGS)
    assert torch.equal(res.W_in, W_in) and torch.equal(res.W_rec, W_rec)  # K1
    for c in CONFIGS:
        o = out["configs"][c]
        assert o["val"].shape == (N_VA, D_OUT) and o["test"].shape == (N_TE, D_OUT)
        assert torch.isfinite(o["test"]).all() and len(o["curve"]) == 2
    assert torch.equal(out["configs"]["B"]["test"], out["configs"]["B+Z"]["test"])   # duygu uzayı sızmıyor
    c = out["configs"]["C:perturb+EMR"]
    assert c["n_trials"] > 0 and 0 <= c["n_accepted"] <= c["n_trials"]
    tr = c["trace"]["train"]
    assert tr.shape[1] == len(PARAMS) and len(c["trace"]["eval"]) > 0 and len(c["trace"]["E_train"]) > 0
    for i, p in enumerate(PARAMS):                                            # K4
        lo, hi = cfg["regulators"]["bounds"][p]
        v = np.concatenate([tr[:, i], c["trace"]["eval"][:, i]])
        assert (v >= lo * (1 - 1e-5)).all() and (v <= hi * (1 + 1e-5)).all()
    assert sum(b["n"] for b in c["tau_bins"].values()) == (N_TR - 5) + N_VA + N_TE


def test_fixed_stream_matches_standalone_a() -> None:
    """Sabit akıştaki A, aynı r'yi okuyan bağımsız bir A ile birebir aynı (C ve diğerleri karışmıyor)."""
    cfg, s, res = setup()
    full = run_streams(cfg, s, res, seed=0, configs=CONFIGS)
    cfg2, s2, res2 = setup()
    alone = run_streams(cfg2, s2, res2, seed=0, configs=["A"])
    assert torch.equal(full["configs"]["A"]["test"], alone["configs"]["A"]["test"])


def test_tau_bins() -> None:
    b = TauBins([20.0, 100.0, 500.0])
    for tau, rate in [(25.0, 0.1), (90.0, 0.2), (120.0, 0.4), (2000.0, 0.3)]:
        b.add(tau, rate)
    r = b.result()
    assert r["20"]["n"] == 1 and r["100"]["n"] == 2 and r["500"]["n"] == 1
    assert abs(r["100"]["rate"] - 0.3) < 1e-9


def test_summarize_paired_diff() -> None:
    rows = [{"session": "s", "seed": i, "k7_ok": True, "leak_free": True,
             "r2_A": 0.3, "r2_A_val": 0.3, "r2_C": 0.3 + d, "r2_C_val": 0.3} for i, d in enumerate([0.1, -0.1, 0.3])]
    sm = summarize(rows, ["A", "C"])["s"]
    assert abs(sm["C_minus_A"]["mean"] - 0.1) < 1e-9 and sm["C_minus_A"]["n_worse"] == 1
    assert "C − A" in table({"s": sm}, ["A", "C"])
