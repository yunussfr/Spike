"""Faz V8 (KR-051, §12.21): oturum rolleri ve hız türevi."""

import copy
import math
from pathlib import Path

import pytest
import torch

from src.data.trialdata import velocity_from_position
from src.utils.config import load_config
from src.v2.data import check_roles, crossday_sessions, session_animal, session_date, session_roles

ROOT = Path(__file__).resolve().parents[2]
V8 = load_config(ROOT / "configs" / "v2_v8.yaml")


def test_configs_pass_role_check() -> None:
    """V8 ve önceki config'ler rol kurallarını sağlar (V7 davranışı bozulmaz)."""
    for name in ("v2_v8.yaml", "v2_pretrain_v8.yaml", "v2_v7.yaml", "v2_base.yaml"):
        check_roles(load_config(ROOT / "configs" / name))


def test_v8_roles_are_causal_and_disjoint() -> None:
    for p in V8["primates"]:
        r = session_roles(V8, p)
        d0 = session_date(r.day0)
        assert r.pre and all(session_date(n) < d0 for n in r.pre)
        assert all(session_date(n) > d0 for n in r.tune + r.test + r.ext)
        assert len(set(r.all())) == len(r.all())
        assert all(session_animal(n) == p for n in r.all())
    assert crossday_sessions(V8, "loco")[0] == "loco_20170214_02"
    assert crossday_sessions(V8, "indy", protocol="ext") == (session_roles(V8, "indy").day0, session_roles(V8, "indy").ext)


def test_dryad_name_parsing() -> None:
    assert str(session_date("Chewie_CO_20150313")) == "2015-03-13"
    assert session_animal("Chewie_CO_20150313") == "chewie"
    with pytest.raises(ValueError):
        session_date("indy_kayit")


def _bad(mut) -> dict:
    c = copy.deepcopy(V8)
    mut(c)
    return c


@pytest.mark.parametrize("mut", [
    lambda c: c["pre_primates"]["indy"].append("indy_20160627_01"),                       # ÖN Day-0'dan sonra (ve EK'te)
    lambda c: c["ext_primates"]["indy"].append(c["primates"]["indy"][1]),                  # EK ∩ TEST
    lambda c: c["pre_primates"]["indy"].insert(0, "loco_20170210_03"),                     # başka hayvan
    lambda c: c["pre_primates"]["indy"].reverse(),                                         # tarih sırası
    lambda c: c["data"]["pre_files"].remove("indy_20160419_01"),                           # dosya listesinde yok
    lambda c: c["tune_primates"]["loco"].insert(0, "loco_20170213_02"),                    # AYAR Day-0'dan önce, ÖN'de
    lambda c: c["ext_primates"].update(chewie=["Chewie_CO_20161014"]),                     # primates'te olmayan hayvan
])
def test_role_violations_raise(mut) -> None:
    with pytest.raises(ValueError):
        check_roles(_bad(mut))


def test_velocity_sine_matches_cosine() -> None:
    dt, f = 0.004, 0.5
    t = torch.arange(2000, dtype=torch.float64) * dt
    pos = torch.stack([torch.sin(2 * math.pi * f * t), 3.0 * t], 1)            # [n, 2]
    v = velocity_from_position(pos, dt, [0])
    true = 2 * math.pi * f * torch.cos(2 * math.pi * f * t)
    assert torch.allclose(v[1:-1, 0].double(), true[1:-1], atol=1e-3)
    assert torch.allclose(v[:, 1], torch.full((2000,), 3.0), atol=1e-5)


def test_velocity_constant_zero_and_segments_do_not_leak() -> None:
    dt = 0.01
    pos = torch.cat([torch.zeros(50, 2), torch.full((50, 2), 100.0), torch.full((1, 2), -7.0)])   # parça arası sıçrama
    v = velocity_from_position(pos, dt, [0, 50, 100])
    assert torch.count_nonzero(v) == 0                                       # sıçrama hıza sızmaz; tek örnek → 0
    vs = velocity_from_position(pos, dt, [0, 50, 100], smooth_ms=30.0)
    assert torch.count_nonzero(vs) == 0


def test_velocity_smoothing_keeps_linear_ramp() -> None:
    dt = 0.01
    pos = (torch.arange(300, dtype=torch.float64) * 0.5).unsqueeze(1)         # 50 birim/s
    v = velocity_from_position(pos, dt, [0, 120], smooth_ms=50.0)
    assert torch.allclose(v, torch.full_like(v, 50.0), atol=1e-4)


def test_velocity_rejects_bad_segments() -> None:
    pos = torch.zeros(10, 2)
    for st in ([1], [0, 5, 5], [0, 10]):
        with pytest.raises(ValueError):
            velocity_from_position(pos, 0.01, st)
