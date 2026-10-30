"""Real-data estimate: effect of biological (secondary/advanced/lagoon) treatment
versus primary-only treatment on log10(EFF/INF) of each legacy PFAS and molar
sums, using LOD-DML and three comparator handlings of non-detects.

Output: results/s03_effects.csv, results/s03_imputations.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import censdml as cd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data_clean"
R = ROOT / "results"
R.mkdir(exist_ok=True)

CMP = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA", "PFBS", "PFHxS", "PFOS", "PFOSA"]
PFCA = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA"]
PFSA = ["PFBS", "PFHxS", "PFOS"]
MW = dict(PFBA=214.04, PFPeA=264.05, PFHxA=314.05, PFHpA=364.06, PFOA=414.07, PFNA=464.08,
          PFDA=514.08, PFBS=300.10, PFHxS=400.11, PFOS=500.13, PFOSA=499.14)
LEACH = {"E", "HG", "MH", "N", "Q", "TB", "WL"}


def proc_class(s):
    s = str(s).lower()
    if s.startswith("lagoon"):
        return "lagoon"
    if s.startswith("primary"):
        return "primary"
    if "biological nutrient" in s:
        return "BNR"
    if "membrane" in s:
        return "MBR"
    if "trickling" in s or "biofilter" in s or "aerated filter" in s:
        return "biofilm"
    if "extended aeration" in s or "oxidation ditch" in s:
        return "ext_aer"
    if "activated sludge" in s:
        return "AS"
    return "other"


def load_design():
    pr = pd.read_parquet(D / "pairs_inf_eff.parquet")
    pl = pd.read_csv(D / "plants.csv")
    pl["code2"] = pl.code.str.replace(r" \(.*\)", "", regex=True)
    pl = pl.drop_duplicates("code2").set_index("code2")
    pl["proc"] = pl.liquid.map(proc_class)
    pl["flow"] = pd.to_numeric(pl.flow_avg.str.replace(",", ""), errors="coerce")
    pl["pres"] = pd.to_numeric(pl.pct_res, errors="coerce")
    pr = pr[pr[[f"INF_hi|{c}" for c in CMP] + [f"EFF_hi|{c}" for c in CMP]].notna().all(axis=1)].copy()
    pr["proc"] = pr.WWTP_CODE.map(pl.proc)
    pr["bio"] = (pr.proc != "primary").astype(float)
    pr["year"] = pr.date.dt.year
    pr["filtered"] = (pr.year < 2018).astype(float)
    pr["comp30min"] = (pr.date < "2012-04-01").astype(float)
    pr["leach"] = pr.WWTP_CODE.isin(LEACH).astype(float)
    f = np.log10(pr.WWTP_CODE.map(pl.flow))
    pr["log_flow"] = f.fillna(f.median())
    pr["flow_na"] = f.isna().astype(float)
    pres = pr.WWTP_CODE.map(pl.pres)
    pr["pres"] = pres.fillna(pres.median())
    pr["T"] = pr.T_inf.fillna(pr.T_inf.median())
    pr["T_na"] = pr.T_inf.isna().astype(float)
    rec = pr.WWTP_CODE.map(pl.receiving).str.strip()
    for r in ["Great Lakes", "St. Lawrence River", "Atlantic Ocean", "Pacific Ocean", "Hudson Bay"]:
        pr["rw_" + r.split()[0]] = (rec == r).astype(float)
    return pr.reset_index(drop=True)


ZCOLS = ["year", "filtered", "comp30min", "leach", "log_flow", "flow_na", "pres", "T", "T_na",
         "rw_Great", "rw_St.", "rw_Atlantic", "rw_Pacific", "rw_Hudson"]


def bounds(pr, side):
    lo = np.column_stack([pr[f"{side}_lo|{c}"].values for c in CMP]).astype(float)
    hi = np.column_stack([pr[f"{side}_hi|{c}"].values for c in CMP]).astype(float)
    lo = np.where(lo > 0, np.log(lo), -np.inf)
    hi = np.log(hi)
    return lo, hi


def outcomes(Vi, Ve):
    """log10 ratios per compound and molar sums from natural-log latent values."""
    out = {}
    for k, c in enumerate(CMP):
        out[c] = (Ve[:, k] - Vi[:, k]) / np.log(10)
    for name, grp in [("sumPFCA", PFCA), ("sumPFSA", PFSA)]:
        ks = [CMP.index(c) for c in grp]
        si = np.sum([np.exp(Vi[:, k]) / MW[CMP[k]] for k in ks], axis=0)
        se = np.sum([np.exp(Ve[:, k]) / MW[CMP[k]] for k in ks], axis=0)
        out[name] = np.log10(se / si)
    return out


def run(M=10, n_perm=300):
    pr = load_design()
    g = pr.WWTP_CODE.values
    d = pr.bio.values
    Z = pr[ZCOLS].values.astype(float)
    ilo, ihi = bounds(pr, "INF")
    elo, ehi = bounds(pr, "EFF")
    Vlo, Vhi = np.hstack([ilo, elo]), np.hstack([ihi, ehi])
    p = len(CMP)
    print(f"n pairs={len(pr)} plants={len(np.unique(g))} primary plants={pr[pr.bio == 0].WWTP_CODE.nunique()}")
    print("censoring INF", np.round((~np.isfinite(ilo)).mean(0), 2))
    print("censoring EFF", np.round((~np.isfinite(elo)).mean(0), 2))

    # --- LOD-DML: multiple imputation
    t0 = time.time()
    Zd = np.column_stack([Z, d])
    imps = []
    for m in range(M):
        V = cd.chained_impute(Vlo, Vhi, Zd, g, n_sweeps=3, seed=1000 + m)
        imps.append(V)
        print(f"  imputation {m + 1}/{M} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(R / "s03_imputations.npz", imps=np.array(imps), Vlo=Vlo, Vhi=Vhi)

    def handle(kind, m=None):
        if kind == "LOD-DML":
            V = imps[m]
        elif kind == "DL/2":
            V = np.where(np.isfinite(Vlo), Vlo, Vhi - np.log(2))
        elif kind == "DL/sqrt2":
            V = np.where(np.isfinite(Vlo), Vlo, Vhi - 0.5 * np.log(2))
        return V[:, :p], V[:, p:]

    rows = []
    for kind in ["LOD-DML", "DL/2", "DL/sqrt2", "complete-case"]:
        reps = range(M) if kind == "LOD-DML" else [None]
        res = {}
        for m in reps:
            if kind == "complete-case":
                Vi, Ve = handle("DL/2")
            else:
                Vi, Ve = handle(kind, m)
            ys = outcomes(Vi, Ve)
            for name, y in ys.items():
                W = np.column_stack([Z, Vi / np.log(10)])
                keep = np.ones(len(y), bool)
                if kind == "complete-case":
                    if name in CMP:
                        k = CMP.index(name)
                        keep = np.isfinite(ilo[:, k]) & np.isfinite(elo[:, k])
                    else:
                        grp = PFCA if name == "sumPFCA" else PFSA
                        ks = [CMP.index(c) for c in grp]
                        keep = np.all(np.isfinite(ilo[:, ks]) & np.isfinite(elo[:, ks]), axis=1)
                    if keep.sum() < 30 or len(np.unique(d[keep])) < 2:
                        continue
                mh, eh = cd.crossfit_nuisance(W[keep], y[keep], d[keep], g[keep])
                th, var = cd.plr(y[keep], d[keep], mh, eh, g[keep])
                res.setdefault(name, []).append((th, var, keep.sum(), mh, eh, y, W, keep))
        for name, lst in res.items():
            th, se = cd.rubin([a[0] for a in lst], [a[1] for a in lst])
            raw = np.mean(lst[0][5][lst[0][7]][d[lst[0][7]] == 1]) - np.mean(lst[0][5][lst[0][7]][d[lst[0][7]] == 0])
            rows.append(dict(method=kind, outcome=name, theta=th, se=se, lo95=th - 1.96 * se, hi95=th + 1.96 * se,
                             n=lst[0][2], raw_diff=raw))
            print(f"{kind:14s} {name:8s} theta={th:+.3f} se={se:.3f} n={lst[0][2]}", flush=True)
            if kind == "LOD-DML" and n_perm and name in ("sumPFCA", "sumPFSA", "PFHxA", "PFOA", "PFOS"):
                a = lst[0]
                _, pp, _ = cd.permutation_p(a[5], d, a[6], g, a[3], n_perm=n_perm, seed=7)
                rows[-1]["perm_p"] = pp
        pd.DataFrame(rows).to_csv(R / "s03_effects.csv", index=False)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    NP = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    run(M, NP)
