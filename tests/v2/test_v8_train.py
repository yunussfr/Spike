"""Faz V8 (KR-051, §12.8 Aşama 1b): çoklu-gün eğitimi — ÖN Day-0 istatistikleriyle kodlanır, `readout` ince ayarı
omurgaya dokunmaz, ÖN + Day-0 birlikte eğitilebilir."""

import torch

from src.data.primate import Session
from src.v2.model import V2Model
from src.v2.train import TrainStep, prepare_session, set_finetune_mode, train_model
from tests.v2.test_train import fake_session, metric, small_cfg


def shifted(s: Session, name: str, gain: float) -> Session:
    return Session(name, s.x * gain, s.y, s.val_start, s.test_start)


def test_pre_session_encoded_with_day0_stats() -> None:
    """ÖN oturumu Day-0 istatistikleriyle kodlanınca kayma görünür kalır; kendi istatistikleriyle silinirdi."""
    cfg = small_cfg()
    d0 = fake_session()
    pre = shifted(d0, "fake_20191201_01", 3.0)                               # aynı yapı, 3× ateşleme
    own = prepare_session(cfg, pre, torch.device("cpu"))
    via0 = prepare_session(cfg, pre, torch.device("cpu"), stats_from=d0)
    ref0 = prepare_session(cfg, d0, torch.device("cpu"))
    tr = d0.split("train")
    assert abs(float(own.x[tr].mean() - ref0.x[tr].mean())) < 1e-4        # kendi istatistiği: kayma silinir
    assert float(via0.x[tr].mean() - ref0.x[tr].mean()) > 0.1               # Day-0 istatistiği: kayma kalır
    assert torch.equal(via0.y_mu, ref0.y_mu) and torch.equal(via0.y_sd, ref0.y_sd)
    assert torch.equal(via0.z0, ref0.z0)                                     # augmentation'ın sıfır kodu Day-0'ınki


def test_readout_finetune_keeps_backbone_bitwise() -> None:
    cfg = small_cfg()
    torch.manual_seed(0)
    m = V2Model(cfg, 5, 2, "plif")
    pr = prepare_session(cfg, fake_session(), torch.device("cpu"))
    bb0 = {k: v.clone() for k, v in m.backbone.state_dict().items()}
    ro0 = {k: v.clone() for k, v in m.readout.state_dict().items()}
    set_finetune_mode(m, "readout")
    train_model(m, pr, cfg, 0, metric)
    set_finetune_mode(m, "full")
    assert all(torch.equal(bb0[k], v) for k, v in m.backbone.state_dict().items())
    assert any(not torch.equal(ro0[k], v) for k, v in m.readout.state_dict().items())
    assert all(p.requires_grad for p in m.parameters())


def test_optimizer_only_trainable_params() -> None:
    cfg = small_cfg()
    m = V2Model(cfg, 5, 2, "plif")
    set_finetune_mode(m, "readout")
    step = TrainStep(m, cfg, 2, 4, 20, 5, 2, torch.device("cpu"))
    ids = {id(p) for g in step.opt.param_groups for p in g["params"]}
    assert ids == {id(p) for p in m.readout.parameters() if p.requires_grad}
    try:
        set_finetune_mode(m, "hepsi")
    except ValueError:
        pass
    else:
        raise AssertionError("geçersiz ince ayar biçimi kabul edildi")


def test_multiday_pool_trains() -> None:
    """ÖN + Day-0 birlikte (Day-0 istatistikleriyle) eğitilebilir; val seçimi tüm oturumların ortalaması."""
    cfg = small_cfg()
    d0 = fake_session()
    pres = [prepare_session(cfg, shifted(d0, f"fake_2019120{i}_01", g), torch.device("cpu"), stats_from=d0)
            for i, g in ((1, 0.7), (2, 1.4))]
    torch.manual_seed(0)
    m = V2Model(cfg, 5, 2, "plif")
    res = train_model(m, [*pres, prepare_session(cfg, d0, torch.device("cpu"))], cfg, 0, metric)
    assert res.epochs_run == cfg["train"]["epochs"]
    assert any("val_r2_fake_20191201_01" in h for h in res.history)
