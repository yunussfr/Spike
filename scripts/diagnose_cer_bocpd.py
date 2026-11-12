"""Teşhis (V7 Aşama 2): beyincik ne öğreniyor (M1–M4) ve BOCPD hangi sinyali izlemeli (S0–S4).

Yalnız okur: eğitim, ayar ya da test verisi yok. Veri Day-0 train (fit, istatistik, kalibrasyon), Day-0 val
(yanlış alarm, elektrot kaybı) ve ayar oturumlarının (`tune_primates`) val dilimi (KV6, kural 9).

Day-0 train dilimi:  [başlangıç + burn-in, + fit)  beyincik ridge'i (M1–M3)
                     [.., %80 sonu)                kaynak istatistikleri m, s² (en çok `--stat_steps` adım)
                     [%80, train sonu)             BOCPD tavlama kalibrasyonu (yanlış alarm ≤ bocpd.calib_false_alarm)
S0 ise duman koşusundakiyle birebir aynıdır (make_ref7: m, s² train'in ilk %80'i, kalibrasyon son %20).

2a — beyincik modelleri (φ(t) → hedef t + h; donuk, Day-0 ridge):
  M1 [r̄, v̂(t−1)] (bugünkü beyincik), M2 [r̄], M3 [v̂(t−1−lag), 1] (kodlama modeli), M4 x̄(t) kopyası.
  Hedef seviye x̄(t+h) ya da değişim x̄(t+h) − x̄(t) ("d" eki).
2b — BOCPD kaynakları: ham x blok ortalamaları (S0, S0s), beyincik artıkları (S1*: M1, S2*: M3, M4 = fark),
  blok düzeyinde kanal başına AR(1) yeniliği (S3), Day-0 FA gizil değişkenleri (S4). Elektrot kaybında
  ölü kanal dedektörü kapalıdır (yalnız BOCPD'nin kendi tespiti ölçülür).

Kullanım:
    uv run python scripts/diagnose_cer_bocpd.py --config configs/v2_v7.yaml \\
        --runs runs/20261107-165844_v2_pretrain_indy_seeds100-101-102 runs/20261107-161250_v2_pretrain_loco_seeds100-101-102
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.v2_online import find_ckpt  # noqa: E402
from experiments.v2_v6 import drop_channels  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.v2.align import bin_means, fa_fit, latent_map  # noqa: E402
from src.v2.cerebellum import CerebellumV2  # noqa: E402
from src.v2.changepoint import BocpdSpec, block_means, bocpd_series, calibrate_temper, day0_block_stats  # noqa: E402
from src.v2.data import crossday_sessions, load_v2_session, ms_to_steps  # noqa: E402
from src.v2.model import V2Model  # noqa: E402
from src.v2.train import prepare_session  # noqa: E402


@dataclass
class Spec:
    """Donuk beyincik modeli: φ(t) → x̄(t+h) (level) ya da x̄(t+h) − x̄(t) (delta)."""
    name: str
    h: int
    kind: str                             # level | delta
    feats: str                            # rv | r | vlag | copy
    W: torch.Tensor | None = None         # [C, P]
    col: slice | None = None              # r̄ projeksiyonundaki sütunlar


# --- akış toplayıcı --------------------------------------------------------------------------------------------------
@torch.no_grad()
def collect(model: V2Model, x: torch.Tensor, a: int, b: int, n_streams: int, burnin: int, alpha_sm: float,
            proj: torch.Tensor | None = None, keep_r: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Donuk model, [a, b) aralığı `n_streams` ardışık akışla (her biri burn-in'li). Kapılanmamış izler (g = 1).

    → x̄ [L, C] (x(t) dahil), v̂(t−1) [L, D], r̄·projᵀ [L, K] (proj: [K, N]), istenirse r̄ [L, N]."""
    model.eval()
    L = b - a
    seg = -(-L // n_streams)
    starts = [a + i * seg for i in range(n_streams) if a + i * seg < b]
    lens = [min(seg, b - s) for s in starts]
    begin = [max(0, s - burnin) for s in starts]
    pre = [s - g for s, g in zip(starts, begin)]
    B, T = len(starts), max(p + n for p, n in zip(pre, lens))
    C, N, dev = x.shape[1], model.backbone.N, x.device
    D = model.readout.weight.shape[1]
    xb = torch.zeros(B, T, C, device=dev)
    for i in range(B):
        xb[i, : pre[i] + lens[i]] = x[begin[i]: starts[i] + lens[i]]
    model.reset_state(B)
    xbar = torch.zeros(B, C, device=dev)
    vprev = torch.zeros(B, D, device=dev)
    o_bar = torch.empty(B, T, C, device=dev)
    o_vp = torch.empty(B, T, D, device=dev)
    o_p = torch.empty(B, T, proj.shape[0], device=dev) if proj is not None else None
    o_r = torch.empty(B, T, N, device=dev) if keep_r else None
    for t in range(T):
        xt = xb[:, t]
        xbar = alpha_sm * xbar + (1.0 - alpha_sm) * xt
        s = model.backbone.step(xt)                                           # [B, N]
        v, rb = model.readout.step(s)                                         # [B, D], [B, N]
        o_bar[:, t] = xbar
        o_vp[:, t] = vprev
        if o_p is not None:
            o_p[:, t] = rb @ proj.T
        if o_r is not None:
            o_r[:, t] = rb
        vprev = v
    model.reset_state(1)

    def cut(o: torch.Tensor | None) -> torch.Tensor | None:
        return None if o is None else torch.cat([o[i, pre[i]: pre[i] + lens[i]] for i in range(B)])

    return cut(o_bar), cut(o_vp), cut(o_p), cut(o_r)                         # type: ignore[return-value]


def vlag_feats(vp: torch.Tensor, lags: list[int]) -> torch.Tensor:
    """vp: [L, D] (v̂(t−1)) → [L, D·len(lags) + 1]: v̂(t−1−lag) (dizi başında 0) ve sabit."""
    m = max(lags)
    vpp = torch.cat([torch.zeros(m, vp.shape[1], device=vp.device), vp])
    L = vp.shape[0]
    return torch.cat([vpp[m - g: m - g + L] for g in lags] + [torch.ones(L, 1, device=vp.device)], 1)


def predict(sp: Spec, bar: torch.Tensor, vp: torch.Tensor, P: torch.Tensor | None, lags: list[int]) -> torch.Tensor:
    """φ(t)'den tahmin, her t için: [L, C]."""
    if sp.feats == "copy":
        return bar
    if sp.feats == "vlag":
        return vlag_feats(vp, lags) @ sp.W.T
    N = sp.W.shape[1] - (vp.shape[1] if sp.feats == "rv" else 0)
    p = P[:, sp.col]
    return p + vp @ sp.W[:, N:].T if sp.feats == "rv" else p


def residual(sp: Spec, bar: torch.Tensor, pred: torch.Tensor, lead: int) -> torch.Tensor:
    """İndeks j ∈ [lead, L) için artık: hedef(j) − tahmin(j − h). → [L − lead, C]"""
    h, L = sp.h, bar.shape[0]
    tgt = bar[lead:] if sp.kind == "level" else bar[lead:] - bar[lead - h: L - h]
    return tgt - pred[lead - h: L - h]


# --- istatistik --------------------------------------------------------------------------------------------------------
def auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """P(pos > neg) + ½ P(eşit) (Mann–Whitney)."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    d = pos.view(-1, 1) - neg.view(1, -1)
    return float((d > 0).float().mean() + 0.5 * (d == 0).float().mean())


def first_block(a: torch.Tensor, at: int = 0) -> int:
    idx = (a[at:] > 0.5).nonzero()
    return int(idx[0, 0]) if len(idx) else -1


def calibrate(hold: torch.Tensor, m: torch.Tensor, s2: torch.Tensor, spec1: BocpdSpec, grid: list[float],
              target: float) -> tuple[float, list, bool]:
    """Blok dizisi hold [nb, C] üzerinde calibrate_temper ile aynı kural. → (τ_T, [(τ, oran)], başarılı mı)"""
    rows, best = [], None
    for t in sorted(grid):
        fa = float((bocpd_series(hold, m, s2, replace(spec1, temper=t)) > 0.5).float().mean())
        rows.append((t, fa))
        if fa <= target:
            best = t
    return (best if best is not None else min(grid)), rows, best is not None


def ar1(stat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kanal başına AR(1), blok düzeyinde: (o_k − m) = ρ (o_{k−1} − m) + e. → m [C], ρ [C]"""
    m = stat.mean(0)
    z = stat - m
    rho = (z[1:] * z[:-1]).sum(0) / z[:-1].pow(2).sum(0).clamp(min=1e-12)
    return m, rho.clamp(-0.999, 0.999)


def ar1_innov(o: torch.Tensor, m: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    z = o - m
    return z[1:] - rho * z[:-1]


# --- ana akış ----------------------------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="day0_<primat>_seed<s>.pt içeren koşular")
    p.add_argument("--primates", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[100])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--piece_steps", type=int, default=60000, help="akış parçası (blok katı); bellek için")
    p.add_argument("--stat_steps", type=int, default=150000, help="model kaynaklarının m, s² dilimi (en çok)")
    p.add_argument("--h_ms", type=float, default=160.0, help="uzun tahmin ufku")
    p.add_argument("--lags_ms", nargs="*", type=float, default=[4.0, 40.0, 80.0, 160.0, 320.0],
                   help="M3 v̂ gecikmeleri (v̂(t−1)'e ek; 4 ms = 1 adım → 0 gecikme sayılır)")
    p.add_argument("--save_blocks", action="store_true", help="kaynak blok dizilerini blocks_<primat>_seed<s>.pt'ye yaz")
    p.add_argument("--skip_bocpd", action="store_true", help="2b BOCPD değerlendirmesini atla (yalnız 2a + bloklar)")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["experiment"]["name"] = "diag_cer_bocpd"
    dev = torch.device(a.device)
    dt = cfg["time"]["dt_ms"]
    burnin = ms_to_steps(cfg["online"]["burnin_ms"], dt)
    alpha_sm = math.exp(-dt / cfg["encoder"]["smooth_tau_ms"])
    H = ms_to_steps(a.h_ms, dt)
    lags = [ms_to_steps(v, dt) - 1 for v in a.lags_ms]
    bc, cc, el = cfg["bocpd"], cfg["cerebellum"], cfg["v6"]["electrode"]
    spec0 = BocpdSpec.from_config(cfg)
    blk, blk_s = spec0.block, spec0.block * dt / 1000.0
    spec1 = replace(spec0, block=1)                                           # kaynaklar blok düzeyinde verilir
    grid, fa_target = [float(v) for v in bc["temper_grid"]], float(bc["calib_false_alarm"])
    piece = (a.piece_steps // blk) * blk
    lead = H + max(lags)
    runs = [Path(r) for r in a.runs]
    logger = RunLogger(cfg, a.seeds, a.config)
    logger.save_json("inputs.json", {"runs": [str(r) for r in runs], **{k: v for k, v in vars(a).items() if k != "runs"}})
    rows: list[dict[str, Any]] = []

    def log(**row: Any) -> None:
        rows.append(row)
        logger.log(len(rows), **row)
        logger.save_json("rows.json", rows)

    for primate in a.primates or list(cfg["tune_primates"]):
        day0, dayk = crossday_sessions(cfg, primate, tune=True)
        s0 = load_v2_session(cfg, day0)
        sk = {n: load_v2_session(cfg, n) for n in dayk}
        C, D = s0.D_in, s0.y.shape[1]
        pr0 = prepare_session(cfg, s0, dev)
        tr, va = s0.split("train"), s0.split("val")
        n80 = tr.start + int(round((1.0 - bc["calib_frac"]) * (tr.stop - tr.start)))
        fit = (tr.start + burnin, tr.start + burnin + cc["fit_steps"])
        stat = (max(fit[1], n80 - a.stat_steps) // blk * blk, n80)
        at = va.start + int(el["at_frac"] * (va.stop - va.start))
        for seed in a.seeds:
            t0 = time.time()
            ck = find_ckpt(runs, f"day0_{primate}_seed{seed}.pt")
            if ck is None:
                print(f"[yok] day0_{primate}_seed{seed}.pt", flush=True)
                continue
            model = V2Model(cfg, C, D).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev))
            N = model.backbone.N
            # --- ridge (fit dilimi) ---
            bar, vp, _, r = collect(model, pr0.x, fit[0], fit[1], 4, burnin, alpha_sm, keep_r=True)
            specs: list[Spec] = []
            for h in (1, H):
                for nm, ft in (("M1", "rv"), ("M2", "r"), ("M3", "vlag"), ("M4", "copy")):
                    specs.append(Spec(f"{nm}_h{h}", h, "level", ft))
            specs += [Spec(f"M1_h{H}d", H, "delta", "rv"), Spec(f"M3_h{H}d", H, "delta", "vlag")]
            m0 = max(lags)
            for sp in specs:
                if sp.feats == "copy":
                    continue
                idx = slice(m0, bar.shape[0] - sp.h)
                phi = {"rv": lambda: torch.cat([r, vp], 1), "r": lambda: r,
                       "vlag": lambda: vlag_feats(vp, lags)}[sp.feats]()[idx]
                tgt = bar[m0 + sp.h:]
                if sp.kind == "delta":
                    tgt = tgt - bar[idx]
                sp.W, _ = CerebellumV2.fit(phi, tgt, cc["ridge"])
            del bar, vp, r
            rsp = [sp for sp in specs if sp.feats in ("rv", "r")]
            for k, sp in enumerate(rsp):
                sp.col = slice(k * C, (k + 1) * C)
            proj = torch.cat([sp.W[:, :N] for sp in rsp])                     # [K·C, N]

            def part(pr: Any, lo: int, hi: int) -> dict[str, Any]:
                """[lo, hi) → model başına artık blok ortalamaları [nb, C], Σ artık², adım sayısı; ham x blokları."""
                out: dict[str, Any] = {sp.name: [[], 0.0, 0] for sp in specs}
                for p0 in range(lo, hi, piece):
                    p1 = min(hi, p0 + piece)
                    bar, vp, P, _ = collect(model, pr.x, p0 - lead, p1, a.streams, burnin, alpha_sm, proj=proj)
                    for sp in specs:
                        res = residual(sp, bar, predict(sp, bar, vp, P, lags), lead)   # [p1 − p0, C]
                        o = out[sp.name]
                        o[0].append(block_means(res, blk))
                        o[1] += float(res.pow(2).sum())
                        o[2] += res.numel()
                for sp in specs:
                    o = out[sp.name]
                    out[sp.name] = (torch.cat(o[0]), o[1] / max(1, o[2]))
                out["x"] = block_means(pr.x[lo:hi], blk)
                return out

            parts = {"stat": part(pr0, *stat), "hold": part(pr0, n80, tr.stop), "val0": part(pr0, va.start, va.stop)}
            s_e, ch = drop_channels(s0, at, el["n_channels"], 1000 * seed)
            pr_e = prepare_session(cfg, s_e, dev)
            parts["drop"] = part(pr_e, va.start, va.stop)
            del pr_e
            drop_blk = (at - va.start) // blk
            for n, s in sk.items():
                prk = prepare_session(cfg, s, dev, stats_from=s0)
                vk = s.split("val")
                parts[n] = part(prk, vk.start, vk.stop)
                del prk
            print(json.dumps({"primate": primate, "seed": seed, "collect_s": round(time.time() - t0)}), flush=True)

            # --- 2a: beyincik modelleri ---
            e2 = {sp.name: parts["val0"][sp.name][1] for sp in specs}
            for sp in specs:
                st = parts["stat"][sp.name][0]
                m, s2 = st.mean(0), st.var(0).clamp(min=spec0.var_floor)
                d0 = ((parts["val0"][sp.name][0] - m).pow(2) / s2).mean(1)
                row = {"kind": "model", "primate": primate, "seed": seed, "model": sp.name, "h": sp.h,
                       "target": sp.kind, "e2_val0": e2[sp.name], "rel_M4": e2[sp.name] / e2[f"M4_h{sp.h}"]}
                base2 = f"M2_h{sp.h}"
                if sp.name.startswith("M1") and sp.kind == "level":
                    row["efference_pct"] = 100.0 * (e2[base2] - e2[sp.name]) / e2[base2]
                for n in sk:
                    dk = ((parts[n][sp.name][0] - m).pow(2) / s2).mean(1)
                    row[f"ratio_{n}"] = parts[n][sp.name][1] / e2[sp.name]
                    row[f"auc_{n}"] = auc(d0, dk)
                dd = ((parts["drop"][sp.name][0] - m).pow(2) / s2).mean(1)
                row["auc_drop"] = auc(dd[:drop_blk], dd[drop_blk:])
                log(**row)

            # --- 2b: BOCPD kaynakları ---
            fa0 = fa_fit(bin_means(pr0.x[stat[0]: stat[1]], ms_to_steps(cfg["align"]["fa_bin_ms"], dt)),
                         int(cfg["align"]["q"]), int(cfg["align"]["em_iters"]), float(cfg["align"]["psi_floor"]))
            Mlat, mu_fa = latent_map(fa0).to(dev), fa0.mu.float().to(dev)
            m_ar, rho = ar1(parts["stat"]["x"])
            src: dict[str, dict[str, torch.Tensor]] = {}
            keys = [k for k in parts]
            src["S0s_x"] = {k: parts[k]["x"] for k in keys}
            for nm in ("M1_h1", f"M1_h{H}", f"M1_h{H}d", f"M3_h{H}", f"M3_h{H}d", "M4_h1"):
                src[f"S_{nm}"] = {k: parts[k][nm][0] for k in keys}
            src["S3_ar1"] = {k: ar1_innov(parts[k]["x"], m_ar, rho) for k in keys}
            src["S4_fa"] = {k: (parts[k]["x"] - mu_fa) @ Mlat.T for k in keys}
            if a.save_blocks:                                                 # BOCPD model teşhisi için (diagnose_bocpd_models.py)
                torch.save({"primate": primate, "seed": seed, "days": list(sk), "drop_blk": drop_blk, "blk_s": blk_s,
                            "src": {n: {k: v.cpu() for k, v in b.items()} for n, b in src.items()}},
                           logger.path(f"blocks_{primate}_seed{seed}.pt"))
            if a.skip_bocpd:
                del model, parts, src
                continue

            def evaluate(name: str, blocks: dict[str, torch.Tensor], m: torch.Tensor, s2: torch.Tensor,
                         tau: float, calib: list, ok: bool) -> None:
                sp = replace(spec1, temper=tau)
                a0 = bocpd_series(blocks["val0"], m, s2, sp)
                ad = bocpd_series(blocks["drop"], m, s2, sp)
                db = drop_blk - (1 if name == "S3_ar1" else 0)
                fb = first_block(ad, db)
                row = {"kind": "bocpd", "primate": primate, "seed": seed, "source": name, "dim": int(m.shape[0]),
                       "tau": tau, "calib_ok": ok, "fa_hold": dict(calib)[tau], "fa_val0": float((a0 > 0.5).float().mean()),
                       "drop_fa_pre": float((ad[:db] > 0.5).float().mean()),
                       "drop_t_s": fb * blk_s if fb >= 0 else -1.0, "calib": calib}
                for n in sk:
                    ak = bocpd_series(blocks[n], m, s2, sp)
                    row[f"det_{n}"] = float((ak > 0.5).float().mean())
                    f = first_block(ak)
                    row[f"t_{n}"] = f * blk_s if f >= 0 else -1.0
                log(**row)

            # S0: duman koşusundakiyle birebir (calibrate_temper, ham x, train'in ilk %80'i)
            xt = pr0.x[tr]
            n_fit = n80 - tr.start
            m, s2 = day0_block_stats(xt[:n_fit], blk, spec0.var_floor)
            tau, calib = calibrate_temper(xt[n_fit:], m, s2, cfg)
            ok = any(t == tau and fa <= fa_target for t, fa in calib)
            evaluate("S0_x", src["S0s_x"], m, s2, tau, calib, ok)
            for name, blocks in src.items():
                st = blocks["stat"]
                m, s2 = st.mean(0), st.var(0).clamp(min=spec0.var_floor)
                tau, calib, ok = calibrate(blocks["hold"], m, s2, spec1, grid, fa_target)
                evaluate(name, blocks, m, s2, tau, calib, ok)
            print(json.dumps({"primate": primate, "seed": seed, "sec": round(time.time() - t0)}), flush=True)
            del model, parts, src
    write_table(rows, logger, H)


def write_table(rows: list[dict[str, Any]], logger: RunLogger, H: int) -> None:
    """Seed ortalamaları: 2a model tablosu ve 2b BOCPD tablosu, primat başına."""
    def mean(rs: list[dict[str, Any]], k: str) -> float:
        v = [r[k] for r in rs if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        return sum(v) / len(v) if v else float("nan")

    out = ["# Teşhis: beyincik (2a) ve BOCPD kaynağı (2b)", "",
           f"h_uzun = {H} adım. Değerler seed ortalaması. Day-k = ayar oturumları (val). Test verisi kullanılmadı.", ""]
    for primate in sorted({r["primate"] for r in rows}):
        rm = [r for r in rows if r["primate"] == primate and r["kind"] == "model"]
        rb = [r for r in rows if r["primate"] == primate and r["kind"] == "bocpd"]
        days = sorted(k[6:] for k in (rm[0] if rm else {}) if k.startswith("ratio_"))
        out += [f"## {primate}", "", "### 2a. Beyincik modelleri", "",
                "| model | ε² val0 | ε²/ε²(M4) | efference % | " + " | ".join(f"ε² oranı {d[-11:]}" for d in days)
                + " | " + " | ".join(f"AUC {d[-11:]}" for d in days) + " | AUC kayıp |",
                "|---|---|---|---|" + "---|" * (2 * len(days)) + "---|"]
        for name in dict.fromkeys(r["model"] for r in rm):
            g = [r for r in rm if r["model"] == name]
            out.append(f"| {name} | {mean(g, 'e2_val0'):.4g} | {mean(g, 'rel_M4'):.3f} | "
                       f"{mean(g, 'efference_pct'):.2f} | "
                       + " | ".join(f"{mean(g, 'ratio_' + d):.2f}" for d in days) + " | "
                       + " | ".join(f"{mean(g, 'auc_' + d):.3f}" for d in days) + f" | {mean(g, 'auc_drop'):.3f} |")
        out += ["", "### 2b. BOCPD kaynakları", "",
                "| kaynak | boyut | τ_T | kalibre | YA ayrılmış | YA Day-0 val | YA kayıp öncesi | kayıp tespiti (s) | "
                + " | ".join(f"tespit oranı {d[-11:]}" for d in days) + " | "
                + " | ".join(f"ilk tespit (s) {d[-11:]}" for d in days) + " |",
                "|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(days))]
        for name in dict.fromkeys(r["source"] for r in rb):
            g = [r for r in rb if r["source"] == name]
            out.append(f"| {name} | {g[0]['dim']} | {mean(g, 'tau'):.3g} | {sum(r['calib_ok'] for r in g)}/{len(g)} | "
                       f"{mean(g, 'fa_hold'):.3f} | {mean(g, 'fa_val0'):.3f} | {mean(g, 'drop_fa_pre'):.3f} | "
                       f"{mean(g, 'drop_t_s'):.1f} | "
                       + " | ".join(f"{mean(g, 'det_' + d):.3f}" for d in days) + " | "
                       + " | ".join(f"{mean(g, 't_' + d):.1f}" for d in days) + " |")
        out.append("")
    out += ["YA: yanlış alarm (a > 0,5 blok oranı; Day-0'da hedef ≤ bocpd.calib_false_alarm). Tespit süresi −1: hiç.",
            "AUC: blok başına normalize kare artık, Day-0 val blokları vs Day-k (ya da kayıp öncesi vs sonrası)."]
    logger.path("table.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main()
