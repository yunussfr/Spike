"""BOCPD model teşhisi (V7 Aşama 2, devam; KR-050 öncesi kanal başına biçim — üretimdeki CR için diagnose_bocpd_cr.py): Day-0 rejiminin modeli ve boyut (A ve C seçenekleri).

Girdi: `scripts/diagnose_cer_bocpd.py --save_blocks` çıktısı blocks_<primat>_seed<s>.pt (kaynak başına, parça başına
blok dizileri: stat, hold, val0, drop, ayar Day-k). Yalnız okur; test verisi yok.

Varyantlar (her kaynak için):
  R    — yeniden girilebilir Day-0 (ReBOCPD); A, C ile birleşir (AR, CR, CAR).
  base — bugünkü §12.16: Day-0 rejimi sabit Gauss N(m, s²), kanal başına, çok boyutlu.
  A    — simetrik: Day-0 rejimi de Normal-Gamma, ama unutmalı (pencere W_D blok) ve ortalaması Day-0'a bağlı:
         önsel μ = m, κ_D,c = f · s²_w,c / s²_b,c (s²_w: W_D pencere içi, s²_b: pencereler arası Day-0 varyansı),
         α_0, β_0 = α_0 s²_w. Yeni rejimler bugünkü gibi (unutmasız). Day-0 içindeki yavaş kayma Day-0 rejiminde kalır;
         zarfın (s²_b) çok dışına çıkan kalıcı kayma yeni rejime gider.
  C    — tek boyut: blok başına y_k = log(ort_c (o_kc − m_c)² / s²_c + ε); BOCPD bu skalerde, Day-0 rejimi sabit Gauss.
  CA   — C'nin skaleri üzerinde A.
Kalibrasyon: her (varyant, f, W_D) için τ_T ızgarası artan sırada; ayrılmış Day-0 diliminde yanlış alarm ≤ hedef olan
en büyük τ_T (ilk aşımda durur). Seçim ölçütleri raporlanır, seçim burada yapılmaz.

Kullanım:
    uv run python scripts/diagnose_bocpd_models.py --config configs/v2_v7.yaml --blocks runs/<diag>/blocks_*.pt
"""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import apply_overrides, load_config  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bocpd_channel import ChannelBOCPD as BOCPD  # noqa: E402  (ilk, kanal başına biçim; KR-050 öncesi)
from src.v2.changepoint import BocpdSpec  # noqa: E402


def t_logpdf(o: torch.Tensor, mu: torch.Tensor, sc2: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    return (torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0) - 0.5 * torch.log(nu * math.pi * sc2)
            - (nu + 1.0) / 2.0 * torch.log1p((o - mu) ** 2 / (nu * sc2)))


class SymBOCPD(BOCPD):
    """Day-0 rejimi: unutmalı Normal-Gamma (önsel m, κ_D [C], α_0, β_0 = α_0 s²_w); yeni rejimler BOCPD ile aynı."""

    def __init__(self, m: torch.Tensor, s2: torch.Tensor, spec: BocpdSpec, kappa_d: torch.Tensor, s2w: torch.Tensor,
                 alpha0: float, window: float) -> None:
        super().__init__(m, s2, spec)
        self.kd, self.a0, self.b0 = kappa_d, alpha0, alpha0 * s2w
        self.g = 1.0 - 1.0 / window
        self.n = 0.0
        self.s = torch.zeros_like(m)
        self.q = torch.zeros_like(m)

    def _loglik(self, o: torch.Tensor, live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, lt = super()._loglik(o, live)
        n, kd = self.n, self.kd
        kap = kd + n
        if n > 0:
            xb = self.s / n
            S = (self.q - self.s ** 2 / n).clamp(min=0.0)
            beta = self.b0 + 0.5 * S + kd * n * (xb - self.m) ** 2 / (2.0 * kap)
        else:
            beta = self.b0
        mu = (kd * self.m + self.s) / kap
        al = self.a0 + 0.5 * n
        sc2 = beta * (kap + 1.0) / (al * kap)
        ld = t_logpdf(o, mu, sc2, torch.full_like(o, 2.0 * al))
        return self.spec.temper * (ld * live).sum(), lt

    def update_block(self, o: torch.Tensor, live: torch.Tensor, valid: torch.Tensor) -> None:
        super().update_block(o, live, valid)
        self.n = self.g * self.n + 1.0
        self.s = self.g * self.s + o
        self.q = self.g * self.q + o * o


class ReBOCPD(BOCPD):
    """Yeniden girilebilir Day-0: değişim noktasında yeni segment p0 olasılıkla yine Day-0 rejimidir.

    π'_D = π_D ℓ_D (1 − H) + H p0 Z,  π'_0 = H (1 − p0) Z,  Z = π_D ℓ_D + Σ π_r ℓ_r.
    Böylece a = 1 − π_D "şu anki segment Day-0 değil" olasılığıdır ve geçici bir sapmadan sonra geri inebilir."""
    p0: float = 0.5

    def update_block(self, o: torch.Tensor, live: torch.Tensor, valid: torch.Tensor) -> None:
        ld, lr = self._loglik(o, live)
        a_d = self.log_pd + ld
        a_r = self.log_pr + lr
        tot = torch.logsumexp(torch.cat([a_d.view(1), a_r]), 0)
        new_d = torch.logaddexp(a_d + self.log_1mh, self.log_h + math.log(self.p0) + tot)
        grow = a_r + self.log_1mh
        new_r = torch.empty_like(self.log_pr)
        new_r[1:] = grow[:-1]
        new_r[-1] = torch.logaddexp(grow[-2], grow[-1])
        new_r[0] = self.log_h + math.log1p(-self.p0) + tot
        z = torch.logsumexp(torch.cat([new_d.view(1), new_r]), 0)
        new_d, new_r = new_d - z, new_r - z
        k1 = self.kappa + 1.0
        mu_u = (self.kappa.unsqueeze(1) * self.mu + o) / k1.unsqueeze(1)
        beta_u = self.beta + (self.kappa / (2.0 * k1)).unsqueeze(1) * (o - self.mu) ** 2
        alpha_u = self.alpha + 0.5
        sp = self.spec
        self.mu = torch.cat([self.m.unsqueeze(0), mu_u[:-1]])
        self.kappa = torch.cat([torch.full((1,), sp.kappa_new), k1[:-1]])
        self.alpha = torch.cat([torch.full((1,), sp.alpha_new), alpha_u[:-1]])
        self.beta = torch.cat([(sp.alpha_new * self.s2).unsqueeze(0), beta_u[:-1]])
        self.log_pd, self.log_pr = new_d, new_r
        self.a = 1.0 - torch.exp(self.log_pd)


class SymReBOCPD(SymBOCPD, ReBOCPD):
    """A + R: unutmalı Day-0 rejimi, yeniden girilebilir."""


def series(make: Any, blocks: torch.Tensor) -> torch.Tensor:
    det = make()
    live = torch.ones(blocks.shape[1])
    one = torch.ones(())
    out = torch.empty(blocks.shape[0])
    for k, o in enumerate(blocks):
        det.update_block(o, live, one)
        out[k] = det.a
    return out


def window_split(o: torch.Tensor, W: int, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """o: [n, C] → pencere içi s²_w ve pencere ortalamalarının (ardışık, örtüşmesiz) varyansı s²_b, [C]."""
    nw = o.shape[0] // W
    if nw < 2:
        raise ValueError(f"stat dilimi {o.shape[0]} blok, W_D = {W} için en az 2 pencere gerekir")
    x = o[: nw * W].reshape(nw, W, -1)
    s2w = x.var(1).mean(0)
    s2b = (x.mean(1).var(0) - s2w / W).clamp(min=floor)                      # pencere ortalamasındaki örnekleme payı çıkarılır
    return s2w.clamp(min=floor), s2b


def first_block(a: torch.Tensor, at: int = 0) -> int:
    idx = (a[at:] > 0.5).nonzero()
    return int(idx[0, 0]) if len(idx) else -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--blocks", nargs="+", required=True)
    p.add_argument("--sources", nargs="*", default=["S0s_x", "S_M1_h1", "S_M1_h40", "S_M3_h40", "S4_fa"])
    p.add_argument("--variants", nargs="*", default=["base", "A", "C", "CA"])
    p.add_argument("--temper_grid", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--f_grid", nargs="*", type=float, default=[0.25, 1.0, 4.0], help="κ_D çarpanı (A, CA)")
    p.add_argument("--w_grid_s", nargs="*", type=float, default=[10.0, 60.0], help="Day-0 rejimi unutma penceresi (s)")
    p.add_argument("--alpha0", type=float, default=10.0)
    p.add_argument("--p0", type=float, default=0.5, help="R: değişim noktasından sonra Day-0'a dönüş olasılığı")
    p.add_argument("--tag", default="", help="çıktı dosya adı eki")
    p.add_argument("--y_eps", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None, help="çıktı klasörü (varsayılan: ilk blok dosyasının klasörü)")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    cfg = apply_overrides(load_config(a.config), a.set)
    spec0 = BocpdSpec.from_config(cfg)
    spec1 = replace(spec0, block=1)
    target = float(cfg["bocpd"]["calib_false_alarm"])
    floor = spec0.var_floor
    out_dir = Path(a.out) if a.out else Path(a.blocks[0]).parent
    rows: list[dict[str, Any]] = []

    for path in a.blocks:
        B = torch.load(path)
        primate, seed, days, drop_blk, blk_s = B["primate"], B["seed"], B["days"], B["drop_blk"], B["blk_s"]
        for src in a.sources:
            blocks = {k: v.float() for k, v in B["src"][src].items()}
            st = blocks["stat"]
            m, s2 = st.mean(0), st.var(0).clamp(min=floor)
            yb = {k: torch.log(((v - m) ** 2 / s2).mean(1, keepdim=True) + a.y_eps) for k, v in blocks.items()}
            for var in a.variants:
                one_d, sym, re_ = "C" in var, "A" in var, "R" in var
                data = yb if one_d else blocks
                dst = data["stat"]
                dm, ds2 = dst.mean(0), dst.var(0).clamp(min=floor)
                configs: list[tuple[float, float]] = ([(f, w) for w in a.w_grid_s for f in a.f_grid] if sym
                                                       else [(0.0, 0.0)])
                for f, w_s in configs:
                    if sym:
                        W = max(2, int(round(w_s / blk_s)))
                        s2w, s2b = window_split(dst, W, floor)
                        kd = (f * s2w / s2b).clamp(1e-2, 1e4)

                        def make(t: float, kd=kd, s2w=s2w, W=W, re_=re_) -> BOCPD:
                            det = (SymReBOCPD if re_ else SymBOCPD)(dm, ds2, replace(spec1, temper=t), kd, s2w,
                                                                   a.alpha0, float(W))
                            det.p0 = a.p0
                            return det
                    else:
                        def make(t: float, re_=re_) -> BOCPD:
                            det = (ReBOCPD if re_ else BOCPD)(dm, ds2, replace(spec1, temper=t))
                            det.p0 = a.p0
                            return det
                    calib, tau = [], None
                    for t in sorted(a.temper_grid):
                        fa = float((series(lambda: make(t), data["hold"]) > 0.5).float().mean())
                        calib.append((t, fa))
                        if fa > target:
                            break
                        tau = t
                    ok = tau is not None
                    tau = tau if ok else min(a.temper_grid)
                    a0 = series(lambda: make(tau), data["val0"])
                    ad = series(lambda: make(tau), data["drop"])
                    fb = first_block(ad, drop_blk)
                    row = {"primate": primate, "seed": seed, "source": src, "variant": var, "f": f, "w_s": w_s,
                           "tau": tau, "calib_ok": ok, "fa_hold": dict(calib).get(tau, float("nan")), "calib": calib,
                           "fa_val0": float((a0 > 0.5).float().mean()),
                           "drop_fa_pre": float((ad[:drop_blk] > 0.5).float().mean()),
                           "drop_t_s": fb * blk_s if fb >= 0 else -1.0}
                    for n in days:
                        ak = series(lambda: make(tau), data[n])
                        row[f"det_{n}"] = float((ak > 0.5).float().mean())
                        fk = first_block(ak)
                        row[f"t_{n}"] = fk * blk_s if fk >= 0 else -1.0
                    rows.append(row)
                    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                      if k != "calib"}, ensure_ascii=False), flush=True)
                    (out_dir / f"bocpd_models{a.tag}_rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    write_table(rows, out_dir, a.tag)


def write_table(rows: list[dict[str, Any]], out_dir: Path, tag: str = "") -> None:
    out = ["# BOCPD model teşhisi (A: simetrik Day-0 rejimi, C: tek boyutlu skor)", "",
           "YA: yanlış alarm (a > 0,5 blok oranı). Kalibre: ayrılmış Day-0 diliminde YA ≤ hedef sağlandı mı.",
           "Tespit: ayar Day-k oturumlarının val diliminde a > 0,5 blok oranı; ilk tespit s (−1: hiç). Kayıp: Day-0 val'de "
           "20 kanal, ölü kanal dedektörü kapalı.", ""]
    for primate in sorted({r["primate"] for r in rows}):
        rp = [r for r in rows if r["primate"] == primate]
        days = sorted(k[4:] for k in rp[0] if k.startswith("det_"))
        out += [f"## {primate}", "",
                "| kaynak | varyant | f | W_D (s) | τ_T | kalibre | YA ayrılmış | YA Day-0 val | YA kayıp öncesi | kayıp tespiti (s) | "
                + " | ".join(f"tespit {d[-11:]}" for d in days) + " | " + " | ".join(f"ilk (s) {d[-11:]}" for d in days) + " |",
                "|---|---|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(days))]
        for r in rp:
            out.append(f"| {r['source']} | {r['variant']} | {r['f']:g} | {r['w_s']:g} | {r['tau']:g} | "
                       f"{'evet' if r['calib_ok'] else 'hayır'} | {r['fa_hold']:.3f} | {r['fa_val0']:.3f} | "
                       f"{r['drop_fa_pre']:.3f} | {r['drop_t_s']:.1f} | "
                       + " | ".join(f"{r['det_' + d]:.3f}" for d in days) + " | "
                       + " | ".join(f"{r['t_' + d]:.1f}" for d in days) + " |")
        out.append("")
    (out_dir / f"bocpd_models{tag}.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main()
