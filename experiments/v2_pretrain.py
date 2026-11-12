"""Faz V3: çoklu oturum ön-eğitimi + oturum başına ince ayar (Day-0 çapası) + günler arası Day-0 modeli.

Primat başına, her seed için (`stages` sırasıyla):
  pretrain — primatın tüm oturumlarının train pencereleriyle (her oturum kendi train istatistikleriyle kodlanır)
  finetune — hedef oturumda ön-eğitim ağırlıklarından ince ayar; val'de en iyi epoch; ardından W_0 çapası (§12.8)
  day0     — yalnızca Day-0 oturumunda sıfırdan eğitim; Day-k oturumlarını hiç görmez. Day-k val/test dilimleri
             Day-0 istatistikleriyle kodlanıp donuk modelle skorlanır (v2-A günler arası). V4'ün başlangıç modeli.
  scratch  — hedef oturumda sıfırdan eğitim (ön-eğitimin katkısını ölçmek için)
  multiday — (V8, KR-051, §12.21) ÖN oturumları (`pre_primates`, hepsi Day-0'dan önce) + Day-0 birlikte, hepsi Day-0
             istatistikleriyle kodlanır (`multiday.epochs`); sonra Day-0 ince ayarı (`multiday.finetune`: full | readout)
             ve W_0 çapası. Day-k'yi hiç görmez. Checkpoint `multiday_<p>_seed<s>.pt` (ara: `multiday_pre_…`).
Model seçimi yalnızca val ile. Test R² yalnızca `--report-test` ile hesaplanır.
Checkpoint'ler koşu klasörüne, SHA-256 özetleri `checkpoints.json`'a yazılır (KV8).

Kullanım (primatlar ayrı süreçte paralel koşabilir):
    uv run python experiments/v2_pretrain.py --config configs/v2_pretrain.yaml --primates indy [--targets all] [--report-test]
    uv run python experiments/v2_pretrain.py --config configs/v2_pretrain_v8.yaml --primates indy --seeds 100   # V8: multiday + day0
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v2.metrics import r2_neurobench  # noqa: E402
from src.utils.config import apply_overrides, load_config, resolve_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.v2.data import check_roles, crossday_sessions, load_v2_session, session_roles  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.train import Prepared, evaluate, prepare_session, set_finetune_mode, train_model  # noqa: E402


def stage_cfg(cfg: dict[str, Any], epochs: int, lr: float | None = None) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["train"]["epochs"] = epochs
    if lr is not None:
        c["train"]["lr"] = lr
    return c


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--stages", nargs="*", default=None)
    p.add_argument("--targets", choices=["tune", "all"], default="tune", help="ince ayar / scratch oturumları")
    p.add_argument("--report-test", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="duman koşusu: tüm aşamalarda epoch sayısı")
    p.add_argument("--resume", nargs="*", default=[], help="çökmüş koşu klasör(ler)i, ilk bulunan kullanılır: oradaki ön-eğitim / ince ayar / day0 checkpoint'leri yeniden kullanılır (eğitilmez, yeniden skorlanır)")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = apply_overrides(load_config(a.config), a.set)
    primates = a.primates or list(cfg["primates"])
    seeds = a.seeds if a.seeds is not None else cfg["experiment"]["tune_seeds"]
    stages = a.stages or cfg["stages"]
    device = resolve_device(cfg["experiment"]["device"])
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{'-'.join(primates)}"   # v2_pretrain_<p>, V7: v2_pretrain_v7_<p>
    logger = RunLogger(cfg, seeds, a.config)
    ep_pre = a.epochs or cfg["pretrain"]["epochs"]
    ep_ft = a.epochs or cfg["finetune"]["epochs"]
    ckpts: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    step = 0

    def save(state: dict[str, torch.Tensor], name: str) -> str:
        path = logger.path(name)
        torch.save(state, path)
        ckpts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.save_json("checkpoints.json", ckpts)
        return str(path)

    def record(**row: Any) -> None:
        nonlocal step
        results.append(row)
        logger.log(step, kind="result", **row)
        step += 1
        logger.save_json("results.json", results)
        print(json.dumps(row), flush=True)

    def scores(model: V2Model, pr: Prepared) -> dict[str, float]:
        v = evaluate(model, pr, "val", cfg, r2_neurobench)
        out = {"val_r2": v["r2"], "val_rate_fast": v["rate_fast"], "val_rate_slow": v["rate_slow"]}
        if a.report_test:
            out["test_r2"] = evaluate(model, pr, "test", cfg, r2_neurobench)["r2"]
        return out

    def resumed(name: str, m: V2Model) -> str | None:
        """--resume klasöründe bu checkpoint varsa modele yükler, yeni klasöre kopyalar (hash'iyle) ve yolunu döner."""
        old = next((Path(d) / name for d in a.resume if (Path(d) / name).exists()), None)
        if old is None:
            return None
        state = torch.load(old, map_location=device)
        m.load_state_dict(state)
        save(state, name)
        return str(old)

    for primate in primates:
        names = list(cfg["primates"][primate])
        day0, dayk = crossday_sessions(cfg, primate)
        sessions = {n: load_v2_session(cfg, n) for n in names}
        preps = {n: prepare_session(cfg, sessions[n], device) for n in names}
        targets = names if a.targets == "all" else [n for n in names if n in cfg["tune_files"]]
        C, D = sessions[names[0]].D_in, sessions[names[0]].y.shape[1]
        pre_names: list[str] = []
        pre_preps: list[Prepared] = []
        if "multiday" in stages:                                             # ÖN: Day-0 istatistikleriyle (§12.21)
            check_roles(cfg)
            pre_names = session_roles(cfg, primate).pre
            if not pre_names:
                raise ValueError(f"multiday: pre_primates.{primate} boş")
            for n in pre_names:
                s = load_v2_session(cfg, n)
                if s.D_in != C:
                    raise ValueError(f"multiday: {n} {s.D_in} kanal, Day-0 {C} (scripts/audit_sessions.py)")
                pre_preps.append(prepare_session(cfg, s, device, stats_from=sessions[day0]))
                del s
        for seed in seeds:
            def log_epoch(row: dict, stage: str, target: str = "") -> None:
                nonlocal step
                logger.log(step, kind="epoch", primate=primate, seed=seed, stage=stage, target=target, **row)
                step += 1
                if "val_r2" in row:
                    print(json.dumps({"primate": primate, "seed": seed, "stage": stage, "target": target,
                                      "epoch": row["epoch"], "val_r2": round(row["val_r2"], 4)}), flush=True)

            pre_state = None
            m = V2Model(cfg, C, D).to(device)
            old = resumed(f"pretrain_{primate}_seed{seed}.pt", m) if "pretrain" in stages else None
            if old is not None:                                             # çökmüş koşudan devam: ön-eğitim yeniden yapılmaz
                pre_state = copy.deepcopy(m.state_dict())
                for n in names:
                    record(primate=primate, seed=seed, stage="pretrain", session=n, best_epoch=-1, resumed_from=old,
                           **scores(m, preps[n]))
            elif "pretrain" in stages:
                set_seed(seed)
                m = V2Model(cfg, C, D).to(device)
                res = train_model(m, [preps[n] for n in names], stage_cfg(cfg, ep_pre), seed, r2_neurobench, None,
                                  lambda r: log_epoch(r, "pretrain"))
                pre_state = res.state
                save(res.state, f"pretrain_{primate}_seed{seed}.pt")
                for n in names:
                    record(primate=primate, seed=seed, stage="pretrain", session=n, best_epoch=res.best_epoch,
                           **scores(m, preps[n]))
            if "finetune" in stages and pre_state is not None:
                for n in targets:
                    set_seed(seed)
                    m = V2Model(cfg, C, D).to(device)
                    prev = resumed(f"finetune_{n}_seed{seed}.pt", m)
                    if prev is not None:
                        record(primate=primate, seed=seed, stage="finetune", session=n, best_epoch=-1,
                               resumed_from=prev, **scores(m, preps[n]))
                        continue
                    m.load_state_dict(pre_state)
                    res = train_model(m, preps[n], stage_cfg(cfg, ep_ft, cfg["finetune"]["lr"]), seed, r2_neurobench,
                                      None, lambda r, n=n: log_epoch(r, "finetune", n))
                    m.readout.save_anchor()
                    save(m.state_dict(), f"finetune_{n}_seed{seed}.pt")
                    record(primate=primate, seed=seed, stage="finetune", session=n, best_epoch=res.best_epoch,
                           **scores(m, preps[n]))
            if "scratch" in stages:
                for n in targets:
                    set_seed(seed)
                    m = V2Model(cfg, C, D).to(device)
                    res = train_model(m, preps[n], stage_cfg(cfg, ep_ft), seed, r2_neurobench, None,
                                      lambda r, n=n: log_epoch(r, "scratch", n))
                    record(primate=primate, seed=seed, stage="scratch", session=n, best_epoch=res.best_epoch,
                           **scores(m, preps[n]))
            if "multiday" in stages:
                md = cfg["multiday"]
                set_seed(seed)
                m = V2Model(cfg, C, D).to(device)
                prev = resumed(f"multiday_{primate}_seed{seed}.pt", m)
                best = -1
                if prev is None:
                    mid = resumed(f"multiday_pre_{primate}_seed{seed}.pt", m)
                    if mid is None:                                          # ÖN + Day-0 ortak eğitim
                        res = train_model(m, [*pre_preps, preps[day0]], stage_cfg(cfg, a.epochs or md["epochs"]), seed,
                                          r2_neurobench, None, lambda r: log_epoch(r, "multiday_pre"))
                        save(res.state, f"multiday_pre_{primate}_seed{seed}.pt")
                        for n, pk in zip(pre_names, pre_preps):
                            record(primate=primate, seed=seed, stage="multiday_pre", session=n, eval_on=n,
                                   best_epoch=res.best_epoch, **scores(m, pk))
                    set_finetune_mode(m, md["finetune"])                    # Day-0 ince ayarı (full | readout)
                    res = train_model(m, preps[day0], stage_cfg(cfg, a.epochs or md["finetune_epochs"],
                                                                md["finetune_lr"]), seed, r2_neurobench, None,
                                      lambda r: log_epoch(r, "multiday", day0))
                    set_finetune_mode(m, "full")
                    m.readout.save_anchor()
                    save(m.state_dict(), f"multiday_{primate}_seed{seed}.pt")
                    best = res.best_epoch
                record(primate=primate, seed=seed, stage="multiday", session=day0, eval_on=day0, best_epoch=best,
                       finetune=md["finetune"], **scores(m, preps[day0]))
                for n in dayk:                                               # donuk M omurgası, Day-0 istatistikleri
                    pk = prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                    record(primate=primate, seed=seed, stage="multiday", session=n, eval_on=n, best_epoch=best,
                           finetune=md["finetune"], **scores(m, pk))
                    del pk
            if "day0" in stages:
                set_seed(seed)
                m = V2Model(cfg, C, D).to(device)
                prev = resumed(f"day0_{primate}_seed{seed}.pt", m)
                if prev is None:
                    res = train_model(m, preps[day0], stage_cfg(cfg, ep_ft), seed, r2_neurobench, None,
                                      lambda r: log_epoch(r, "day0", day0))
                    m.readout.save_anchor()
                    save(m.state_dict(), f"day0_{primate}_seed{seed}.pt")
                best = -1 if prev is not None else res.best_epoch
                record(primate=primate, seed=seed, stage="day0", session=day0, eval_on=day0, best_epoch=best,
                       **scores(m, preps[day0]))
                for n in dayk:                                               # donuk Day-0 modeli, Day-0 istatistikleri
                    pk = prepare_session(cfg, sessions[n], device, stats_from=sessions[day0])
                    record(primate=primate, seed=seed, stage="day0", session=n, eval_on=n, best_epoch=best,
                           **scores(m, pk))
                    del pk
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # tablo: oturum × aşama, val (ve varsa test) ortalama ± std
    key = "test_r2" if a.report_test else "val_r2"
    h = cfg["backbone"]["homeostasis"]
    lines = [f"| Oturum | Aşama | {key} | val R² | oran hızlı / yavaş (val) |", "|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["session"], r["stage"]), []).append(r)
    for (n, stg), rows in groups.items():
        k = np.array([r[key] for r in rows])
        v = np.array([r["val_r2"] for r in rows])
        rf = np.mean([r["val_rate_fast"] for r in rows])
        rs = np.mean([r["val_rate_slow"] for r in rows])
        bad = "" if h["f_min"] <= rf <= h["f_max"] and h["f_min"] <= rs <= h["f_max"] else " ✗"
        lines.append(f"| {n} | {stg} | {k.mean():.3f} ± {k.std():.3f} | {v.mean():.3f} | {rf:.3f} / {rs:.3f}{bad} |")
    lines += ["", f"Seed {seeds}. day0 satırları: Day-0'da eğitilen donuk model; Day-k satırları günler arası (v2-A). "
              f"multiday: ÖN + Day-0 omurgası, Day-0 ince ayarlı (V8, kol M); multiday_pre: ÖN oturumlarında ara model. "
              f"✗: alt-popülasyon oranı [{h['f_min']}, {h['f_max']}] dışında (KV7)."]
    logger.path("table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"koşu klasörü: {logger.dir}")


if __name__ == "__main__":
    main()
