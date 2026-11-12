"""Faz V8 Adım 0 (KR-051, §12.21): oturum denetimi — salt okuma, `results/v8/audit.md` yazar.

Zenodo (O'Doherty, 583331): config'teki rol oturumlarından diskte olanlar için kanal adları (Day-0 ile aynı düzen mi?),
süre, örnekleme, birim satırı sayısı, kanal başına ateşleme oranı ve oran profilinin Day-0 ile korelasyonu (kanal
kimliği kaymasının kaba göstergesi), hiç ateşlemeyen kanal sayısı.

Dryad (xd2547dkt, TrialData): `data/dryad/*.mat` dosyalarının yapısı — alan adları, deneme sayısı, ilk denemenin alan
şekilleri, skaler alanlar (bin_size, monkey, date, task), spike / unit_guide / pos alanları. Yükleyici bu rapora göre
yazılır.

Kullanım:
    uv run python scripts/audit_sessions.py --config configs/v2_v8.yaml [--dryad-dir data/dryad] [--out results/v8]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402
from src.v2.data import check_roles, session_roles  # noqa: E402


def h5_str(f: Any, ref: Any) -> str:
    return "".join(chr(int(c)) for c in f[ref][()].ravel())


def zenodo_info(path: Path) -> dict[str, Any]:
    """HDF5 (.mat v7.3) → kanal adları, süre, adım, kanal başına oran [C] (tüm birimler, hash dahil)."""
    import h5py

    with h5py.File(path, "r") as f:
        cn = f["chan_names"]
        names = [h5_str(f, cn[0, i]) for i in range(cn.shape[1])]
        t = f["t"][()].ravel()
        sp = f["spikes"]
        counts = np.zeros(sp.shape[1])
        for c in range(sp.shape[1]):
            for u in range(sp.shape[0]):
                a = f[sp[u, c]][()]
                if a.ndim == 2 and a.shape[0] == 1 and a.shape[1] > 0 and np.issubdtype(a.dtype, np.floating):
                    counts[c] += a.shape[1]
        dur = float(t[-1] - t[0])
        return {"names": names, "dur_s": dur, "dt_ms": 1000.0 * float(np.median(np.diff(t))), "units": sp.shape[0],
                "rate": counts / dur, "finger_dims": f["finger_pos"].shape[0]}


def audit_zenodo(cfg: dict[str, Any]) -> list[str]:
    root = Path(cfg["data"]["dir"])
    L = ["## Zenodo 583331: rol oturumları", "",
         "Oran: kanal başına tüm birimlerin (hash dahil) spike/s'si. `ρ(Day-0)`: kanal oran profilinin Day-0 ile Pearson "
         "korelasyonu (kanal kimliği kaymasının kaba göstergesi). `kanal düzeni`: chan_names Day-0 ile birebir aynı mı.", ""]
    for primate in cfg["primates"]:
        r = session_roles(cfg, primate)
        roles = [("ÖN", n) for n in r.pre] + [("Day-0", r.day0)] + [("AYAR", n) for n in r.tune] + \
                [("TEST", n) for n in r.test] + [("EK", n) for n in r.ext]
        p0 = root / f"{r.day0}.mat"
        ref = zenodo_info(p0) if p0.exists() else None
        L += [f"### {primate}", "", "| Rol | Oturum | C | kanal düzeni | süre dk | Δt ms | birim | ort. oran | sessiz kanal "
              "| ρ(Day-0) |", "|---|---|---|---|---|---|---|---|---|---|"]
        missing = []
        for role, n in roles:
            p = root / f"{n}.mat"
            if not p.exists():
                missing.append(n)
                continue
            z = ref if (ref is not None and n == r.day0) else zenodo_info(p)
            same = "—" if ref is None else ("aynı" if z["names"] == ref["names"] else
                                            f"FARKLI (Day-0 {len(ref['names'])})")
            rho = "—" if ref is None or len(z["rate"]) != len(ref["rate"]) else \
                f"{np.corrcoef(z['rate'], ref['rate'])[0, 1]:.3f}"
            L.append(f"| {role} | {n} | {len(z['names'])} | {same} | {z['dur_s'] / 60:.1f} | {z['dt_ms']:.3f} | "
                     f"{z['units']} | {z['rate'].mean():.2f} | {int((z['rate'] == 0).sum())} | {rho} |")
            print(f"[zenodo] {n}", flush=True)
        if missing:
            L += ["", f"Diskte yok ({len(missing)}): {', '.join(missing)}"]
        L.append("")
    return L


def describe(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray {v.shape} {v.dtype}"
    if isinstance(v, (str, int, float, np.generic)):
        return f"{type(v).__name__} = {v!r}"[:120]
    return type(v).__name__


def audit_dryad(dryad_dir: Path) -> list[str]:
    files = sorted(dryad_dir.glob("*.mat"))
    L = ["## Dryad xd2547dkt: dosya yapısı", ""]
    if not files:
        return L + [f"`{dryad_dir}` altında .mat yok. İndir: https://doi.org/10.5061/dryad.xd2547dkt "
                    "(ör. Chewie_CO_20150630.mat, 167 MB) → bu klasör.", ""]
    for path in files:
        L += [f"### {path.name} ({path.stat().st_size / 1e6:.0f} MB)", ""]
        try:
            import h5py
            with h5py.File(path, "r") as f:
                L.append("Biçim: HDF5 (.mat v7.3)")
                f.visititems(lambda k, o: L.append(f"- `{k}`: {getattr(o, 'shape', '')} {getattr(o, 'dtype', '')}")
                             if k.count("/") < 2 else None)
            L.append("")
            continue
        except OSError:
            pass
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        L.append("Biçim: MATLAB v5 (scipy)")
        for k, v in m.items():
            if k.startswith("__"):
                continue
            arr = np.atleast_1d(v)
            first = arr.flat[0]
            fields = getattr(first, "_fieldnames", None)
            L.append(f"- değişken `{k}`: {describe(v)}; eleman {len(arr)}" + (" (struct dizisi)" if fields else ""))
            if not fields:
                continue
            L += ["", "| alan | ilk deneme | son deneme |", "|---|---|---|"]
            last = arr.flat[-1]
            for fn in fields:
                L.append(f"| `{fn}` | {describe(getattr(first, fn))} | {describe(getattr(last, fn))} |")
            L.append("")
            ln = [getattr(t, "pos", np.zeros((0,))).shape[0] for t in arr.flat if hasattr(t, "pos")]
            if ln:
                L.append(f"Deneme uzunluğu (pos satırı): min {min(ln)}, medyan {int(np.median(ln))}, max {max(ln)}, "
                         f"toplam {sum(ln)}")
            for fn in fields:
                if "unit_guide" in fn:
                    ug = np.asarray(getattr(first, fn))
                    L.append(f"`{fn}`: {ug.shape}; ilk satırlar {ug[:5].tolist()}; benzersiz elektrot "
                             f"{len(np.unique(ug[:, 0])) if ug.ndim == 2 else '?'}")
            L.append("")
        print(f"[dryad] {path.name}", flush=True)
    return L


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v2_v8.yaml")
    p.add_argument("--dryad-dir", default="data/dryad")
    p.add_argument("--out", default="results/v8")
    p.add_argument("--skip-zenodo", action="store_true")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(a.config)
    check_roles(cfg)
    L = ["# Faz V8 Adım 0: oturum denetimi", "", f"Config: `{a.config}`. Salt okuma (`scripts/audit_sessions.py`).", ""]
    if not a.skip_zenodo:
        L += audit_zenodo(cfg)
    L += audit_dryad(Path(a.dryad_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
