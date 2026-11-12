"""Teşhis (V7): fazik kolun (plastik beyincik sürprizi) normalizasyonu.

Sorun: s_cer = ‖e_cer‖²/C / ε²_0 mutlak büyüklük. Indy'de Day-k ateşlemesi düşük → Day-0 ölçeğiyle kodlanan girdinin
varyansı ve onunla beyincik hatası küçülür → yeni günde sürpriz *azalır* (KR-047: iç arousal Day-k a⁺ 0,00, Day-0 0,18).

Burada plastik beyincik (CerebellumV2, NLMS η_cer) akışta koşturulur; hatanın kare ortalaması dört biçimde
normalize edilir, iç arousal dönüşümüyle (E_a EMA τ_a, a = tanh(E_a − 1); affect.update_arousal, w_u = 0) okunur:
  N0  ‖e‖²                                    / taban        (bugünkü)
  N1  EMA ‖e‖² / EMA ‖x̄(t) − x̄(t−1)‖²         / taban        (kopya tahminine göre: ölçekten bağımsız öngörülebilirlik)
  N2  EMA ‖e‖² / EMA ‖x̄ − EMA_yavaş(x̄)‖²      / taban        (girdinin yerel enerjisine göre)
  N3  EMA ort_c e_c² / v_c                    / taban        (kanal başına yerel varyans, v_c: EMA_yavaş)
Taban: Day-0 train'in ayrılmış dilimi (beyincik ridge'inden sonra), donuk beyincikle. Etiket kullanılmaz (KV3, KV6).

Ölçütler: Day-0 val'de a⁺ (yanlış sürpriz), Day-k (ayar oturumları, val) başında ve sonunda a⁺ (fazik: başta yüksek,
uyumla sönmeli), sönme süresi; Day-0 val'de 20 kanal kaybı öncesi / sonrası a⁺.

Kullanım:
    uv run python scripts/diagnose_phasic.py --config configs/v2_v7.yaml --runs <day0 checkpoint koşuları>
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cer_bocpd import collect  # noqa: E402
from experiments.v2_online import find_ckpt  # noqa: E402
from experiments.v2_v6 import drop_channels  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.cerebellum import CerebellumV2  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.train import prepare_session  # noqa: E402

NORMS = ("N0", "N1", "N2", "N3")


@torch.no_grad()
def run_stream(bar: torch.Tensor, vp: torch.Tensor, r: torch.Tensor, w0: torch.Tensor, etas: list[float],
               nlms_eps: float, lam_a: float, lam_s: float, base: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    """Plastik beyincik (koşul başına η) + normalizasyonlar. bar [L, C] x̄(t), vp [L, D] v̂(t−1), r [L, N] r̄(t).

    → ham oranlar {Nk: [E, L]} (base None) ya da arousal a {Nk: [E, L]} (base verilirse; oran / taban)."""
    L, C = bar.shape
    E = len(etas)
    eta = torch.tensor(etas).view(E, 1, 1)
    W = w0.unsqueeze(0).repeat(E, 1, 1)                                       # [E, C, P]
    phi_prev = torch.cat([r[0], vp[0]])                                       # φ(0)
    x_hat = torch.einsum("ecp,p->ec", W, phi_prev)                            # x̂(1)
    num = torch.zeros(E)                                                      # EMA ‖e‖²/C
    num3 = torch.zeros(E)                                                     # EMA ort_c e²/v_c
    den1 = torch.zeros(())                                                    # EMA ‖Δx̄‖²/C
    m_s = bar[0].clone()                                                      # EMA_yavaş x̄  [C]
    v_s = torch.full((C,), float(bar.var(0).mean()))                          # EMA_yavaş (x̄ − m_s)²  [C]
    den2 = torch.zeros(())
    e_a = {k: torch.zeros(E) for k in NORMS}
    out = {k: torch.empty(E, L - 1) for k in NORMS}
    for t in range(1, L):
        xb = bar[t]
        e = xb - x_hat                                                        # [E, C]
        e2 = e.pow(2).mean(1)                                                 # [E]
        d1 = (xb - bar[t - 1]).pow(2).mean()
        dev = xb - m_s
        m_s += lam_s * dev
        v_s += lam_s * (dev * dev - v_s)
        d2 = (dev * dev).mean()
        e3 = (e.pow(2) / v_s.clamp(min=1e-6)).mean(1)
        a = lam_a
        num += a * (e2 - num)
        num3 += a * (e3 - num3)
        den1 += a * (d1 - den1)
        den2 += a * (d2 - den2)
        raw = {"N0": num, "N1": num / den1.clamp(min=1e-12), "N2": num / den2.clamp(min=1e-12), "N3": num3}
        for k in NORMS:
            out[k][:, t - 1] = raw[k] if base is None else torch.tanh(raw[k] / base[k] - 1.0)
        # NLMS (φ(t−1) ile; KV3: yalnız e_cer'den), sonra x̂(t+1) = W φ(t)
        st = phi_prev / (phi_prev.pow(2).sum() + nlms_eps)
        W += eta * torch.einsum("ec,p->ecp", e, st)
        phi_prev = torch.cat([r[t], vp[t]])
        x_hat = torch.einsum("ecp,p->ec", W, phi_prev)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[100])
    p.add_argument("--etas", nargs="*", type=float, default=[0.0, 1e-3, 1e-2], help="beyincik NLMS adımları")
    p.add_argument("--tau_slow_s", type=float, default=30.0, help="N2, N3 yerel istatistik zaman sabiti")
    p.add_argument("--early_s", type=float, default=30.0)
    p.add_argument("--drop_win_s", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = "diag_phasic"
    dev = torch.device("cpu")
    dt = cfg["time"]["dt_ms"]
    burnin = ms_to_steps(cfg["online"]["burnin_ms"], dt)
    alpha_sm = math.exp(-dt / cfg["encoder"]["smooth_tau_ms"])
    lam = lambda s: 1.0 - math.exp(-dt / (1000.0 * s))                         # noqa: E731
    lam_a, lam_s = lam(cfg["v6"]["dyn"]["tau_arousal_s"]), lam(a.tau_slow_s)
    cc, el = cfg["cerebellum"], cfg["v6"]["electrode"]
    early, dwin = int(a.early_s * 1000 / dt), int(a.drop_win_s * 1000 / dt)
    logger = RunLogger(cfg, a.seeds, a.config)
    logger.save_json("inputs.json", {"runs": a.runs, **{k: v for k, v in vars(a).items() if k != "runs"}})
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}

    for primate in a.primates or list(cfg["tune_primates"]):
        day0, dayk = crossday_sessions(cfg, primate, tune=True)
        s0 = load_v2_session(cfg, day0)
        pr0 = prepare_session(cfg, s0, dev)
        tr, va = s0.split("train"), s0.split("val")
        for seed in a.seeds:
            t0 = time.time()
            ck = find_ckpt([Path(r) for r in a.runs], f"day0_{primate}_seed{seed}.pt")
            if ck is None:
                print(f"[yok] day0_{primate}_seed{seed}.pt")
                continue
            model = V2Model(cfg, s0.D_in, s0.y.shape[1]).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev))
            # ridge + taban: Ctx.make_ref ile aynı dilim (train sonu: fit_steps + holdout_steps)
            n, h = cc["fit_steps"], cc["holdout_steps"]
            bar, vp, _, r = collect(model, pr0.x, max(tr.start, tr.stop - n - h), tr.stop, 1, burnin, alpha_sm, keep_r=True)
            phi = torch.cat([r, vp], 1)
            w0, _ = CerebellumV2.fit(phi[:-1][:-h], bar[1:][:-h], cc["ridge"])
            hb = slice(len(bar) - h - 1, len(bar))
            raw0 = run_stream(bar[hb], vp[hb], r[hb], w0, [0.0], cc["nlms_eps"], lam_a, lam_s, None)
            skip = int(5 * 1000 / dt)                                         # EMA ısınması
            base = {k: raw0[k][0, skip:].mean() for k in NORMS}
            del bar, vp, r, phi

            def part(pr: Any, lo: int, hi: int) -> dict[str, torch.Tensor]:
                b, v, _, rr = collect(model, pr.x, lo, hi, 1, burnin, alpha_sm, keep_r=True)
                return run_stream(b, v, rr, w0, a.etas, cc["nlms_eps"], lam_a, lam_s, base)

            res = {"val0": part(pr0, va.start, va.stop)}
            at = va.start + int(el["at_frac"] * (va.stop - va.start))
            s_e, _ = drop_channels(s0, at, el["n_channels"], 1000 * seed)
            res["drop"] = part(prepare_session(cfg, s_e, dev), va.start, va.stop)
            drop_i = at - va.start - 1
            for nk in dayk:
                sk = load_v2_session(cfg, nk)
                vk = sk.split("val")
                res[nk] = part(prepare_session(cfg, sk, dev, stats_from=s0), vk.start, vk.stop)
            print(json.dumps({"primate": primate, "seed": seed, "sec": round(time.time() - t0)}), flush=True)
            for k in NORMS:
                for e, eta in enumerate(a.etas):
                    pos = lambda x: x.clamp(min=0.0)                          # noqa: E731
                    v0 = res["val0"][k][e]
                    dr = res["drop"][k][e]
                    row = {"primate": primate, "seed": seed, "norm": k, "eta_cer": eta,
                           "a_pos_val0": float(pos(v0).mean()), "a_mean_val0": float(v0.mean()),
                           "drop_pre": float(pos(dr[drop_i - dwin: drop_i]).mean()),
                           "drop_post": float(pos(dr[drop_i: drop_i + dwin]).mean()),
                           "drop_late": float(pos(dr[drop_i + 3 * dwin: drop_i + 6 * dwin]).mean())}
                    for nk in dayk:
                        x = res[nk][k][e]
                        half = len(x) // 2
                        pk = int(x[:early].argmax())
                        below = (x[pk:] < 0.5 * float(x[pk])).nonzero()
                        row[f"early_{nk}"] = float(pos(x[:early]).mean())
                        row[f"late_{nk}"] = float(pos(x[half:]).mean())
                        row[f"decay_s_{nk}"] = float(below[0, 0]) * dt / 1000.0 if len(below) and float(x[pk]) > 0 else -1.0
                    rows.append(row)
                    logger.log(len(rows), **row)
                    traces[f"{primate}_{seed}_{k}_{eta}"] = {pt: res[pt][k][e, ::250].clone() for pt in res}
            logger.save_json("rows.json", rows)
            torch.save(traces, logger.path("traces.pt"))
            del model, res
    write_table(rows, logger)


def write_table(rows: list[dict[str, Any]], logger: RunLogger) -> None:
    def mean(rs: list[dict[str, Any]], k: str) -> float:
        v = [r[k] for r in rs if k in r]
        return sum(v) / len(v) if v else float("nan")

    out = ["# Teşhis: fazik kol normalizasyonu", "",
           "a = tanh(oran/taban − 1), oran τ_a EMA'sı; a⁺ = max(a, 0). İyi bir fazik sinyal: Day-0 val'de a⁺ ≈ 0; "
           "Day-k başında yüksek, sonunda (plastik beyincik uyum sağlayınca) düşük; kanal kaybında sıçrar.", ""]
    for primate in sorted({r["primate"] for r in rows}):
        rp = [r for r in rows if r["primate"] == primate]
        days = sorted(k[6:] for k in rp[0] if k.startswith("early_"))
        out += [f"## {primate}", "",
                "| norm | η_cer | a⁺ Day-0 val | " + " | ".join(f"baş {d[-11:]}" for d in days) + " | "
                + " | ".join(f"son {d[-11:]}" for d in days) + " | " + " | ".join(f"sönme (s) {d[-11:]}" for d in days)
                + " | kayıp öncesi | kayıp 0–10 s | kayıp 30–60 s |",
                "|---|---|---|" + "---|" * (3 * len(days)) + "---|---|---|"]
        for k in NORMS:
            for eta in sorted({r["eta_cer"] for r in rp}):
                g = [r for r in rp if r["norm"] == k and r["eta_cer"] == eta]
                out.append(f"| {k} | {eta:g} | {mean(g, 'a_pos_val0'):.3f} | "
                           + " | ".join(f"{mean(g, 'early_' + d):.3f}" for d in days) + " | "
                           + " | ".join(f"{mean(g, 'late_' + d):.3f}" for d in days) + " | "
                           + " | ".join(f"{mean(g, 'decay_s_' + d):.1f}" for d in days) + " | "
                           + f"{mean(g, 'drop_pre'):.3f} | {mean(g, 'drop_post'):.3f} | {mean(g, 'drop_late'):.3f} |")
        out.append("")
    logger.path("table.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main()
