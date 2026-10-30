"""(a) process-specific contrasts vs primary-only (LOD-DML, s03 imputations)
(b) within-plant stage test INF->PE (primary step) vs PE->EFF (biological step)
(c) biosolids: effect of biological treatment on log10 biosolids PFAS (ng/g dw)
Outputs: results/s05_process.csv, s05_stage.csv, s05_biosolids.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import censdml as cd  # noqa: E402
import s03_upgrade_effect as s3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D, R = ROOT / "data_clean", ROOT / "results"
LN10 = np.log(10)
P = len(s3.CMP)


def process_contrasts(pr, imps):
    rows = []
    g, Z = pr.WWTP_CODE.values, pr[s3.ZCOLS].values.astype(float)
    for proc in ["lagoon", "AS", "biofilm", "BNR", "ext_aer"]:
        keep = pr.proc.isin(["primary", proc]).values
        d = (pr.proc == proc).values.astype(float)
        for name in ["sumPFCA", "PFHxA", "PFOA", "PFDA", "PFOS", "sumPFSA"]:
            ths, vs = [], []
            for V in imps:
                y = s3.outcomes(V[:, :P], V[:, P:])[name]
                W = np.column_stack([Z, V[:, :P] / LN10])
                mh, eh = cd.crossfit_nuisance(W[keep], y[keep], d[keep], g[keep], n_splits=4)
                th, var = cd.plr(y[keep], d[keep], mh, eh, g[keep])
                ths.append(th)
                vs.append(var)
            th, se = cd.rubin(ths, vs)
            rows.append(dict(proc=proc, outcome=name, theta=th, se=se, n=keep.sum(),
                             plants=pr.loc[keep & (d == 1), "WWTP_CODE"].nunique()))
            print(proc, name, round(th, 3), round(se, 3), flush=True)
    return pd.DataFrame(rows)


def stage_test(M=10):
    w = pd.read_parquet(D / "wide_samples.parquet")
    one = w.groupby(["WWTP_CODE", "date", "loc"]).filter(lambda x: len(x) == 1)
    sets = {}
    for loc in ["INF", "PE", "EFF"]:
        sets[loc] = one[one["loc"] == loc].drop(columns=["loc", "AXYS_ID"]).set_index(["WWTP_CODE", "date"])
    tri = sets["INF"].add_prefix("INF_").join(sets["PE"].add_prefix("PE_"), how="inner").join(
        sets["EFF"].add_prefix("EFF_"), how="inner").reset_index()
    cols = [f"{s}_hi|{c}" for s in ["INF", "PE", "EFF"] for c in s3.CMP]
    tri = tri[tri[cols].notna().all(axis=1)].reset_index(drop=True)
    pr = s3.load_design()[["WWTP_CODE", "date"] + s3.ZCOLS + ["proc"]]
    tri = tri.merge(pr, on=["WWTP_CODE", "date"], how="left")
    tri = tri[tri.year.notna()].reset_index(drop=True)
    print("stage triples", len(tri), tri.WWTP_CODE.nunique())
    lo, hi = [], []
    for s in ["INF", "PE", "EFF"]:
        l = np.column_stack([tri[f"{s}_lo|{c}"].values for c in s3.CMP]).astype(float)
        h = np.column_stack([tri[f"{s}_hi|{c}"].values for c in s3.CMP]).astype(float)
        with np.errstate(divide="ignore"):
            lo.append(np.where(l > 0, np.log(l), -np.inf))
        hi.append(np.log(h))
    Vlo, Vhi = np.hstack(lo), np.hstack(hi)
    g = tri.WWTP_CODE.values
    Z = tri[s3.ZCOLS].values.astype(float)
    rows = []
    res = {}
    for m in range(M):
        V = cd.chained_impute(Vlo, Vhi, Z, g, n_sweeps=3, seed=500 + m)
        Vi, Vp, Ve = V[:, :P], V[:, P:2 * P], V[:, 2 * P:]
        a = s3.outcomes(Vi, Vp)  # primary step
        b = s3.outcomes(Vp, Ve)  # biological step
        for name in a:
            diff = b[name] - a[name]
            for lab, y in [("primary_step", a[name]), ("bio_step", b[name]), ("bio_minus_primary", diff)]:
                gm = pd.Series(y).groupby(g).mean()  # plant means -> cluster-level inference
                res.setdefault((name, lab), []).append((gm.mean(), gm.var(ddof=1) / len(gm)))
    for (name, lab), lst in res.items():
        th, se = cd.rubin([x[0] for x in lst], [x[1] for x in lst])
        rows.append(dict(outcome=name, contrast=lab, theta=th, se=se, n=len(tri), plants=len(np.unique(g))))
    return pd.DataFrame(rows)


def biosolids(pr, imps, M=10):
    w = pd.read_parquet(D / "wide_samples.parquet")
    one = w.groupby(["WWTP_CODE", "date", "loc"]).filter(lambda x: len(x) == 1)
    bs = one[one["loc"] == "BIOS"].drop(columns=["loc", "AXYS_ID"])
    m = pr[["WWTP_CODE", "date"]].reset_index().merge(bs, on=["WWTP_CODE", "date"], how="inner")
    m = m[m[[f"hi|{c}" for c in s3.CMP]].notna().all(axis=1)]
    idx = m["index"].values
    sub = pr.loc[idx].reset_index(drop=True)
    pl = pd.read_csv(D / "plants.csv")
    pl["code2"] = pl.code.str.replace(r" \(.*\)", "", regex=True)
    sol = pl.drop_duplicates("code2").set_index("code2").solids.str.lower()
    s = sub.WWTP_CODE.map(sol).fillna("")
    S = np.column_stack([s.str.contains("alkaline|lime").astype(float), s.str.contains("anaerobic").astype(float),
                         s.str.contains("aerobic") & ~s.str.contains("anaerobic"), s.str.contains("thermophilic").astype(float)]).astype(float)
    l = np.column_stack([m[f"lo|{c}"].values for c in s3.CMP]).astype(float)
    h = np.column_stack([m[f"hi|{c}"].values for c in s3.CMP]).astype(float)
    with np.errstate(divide="ignore"):
        Blo = np.where(l > 0, np.log(l), -np.inf)
    Bhi = np.log(h)
    g, d = sub.WWTP_CODE.values, sub.bio.values
    Z = sub[s3.ZCOLS].values.astype(float)
    print("biosolids rows", len(sub), "plants", len(np.unique(g)), "primary plants", sub[sub.bio == 0].WWTP_CODE.nunique())
    rows, res = [], {}
    for mm, V in enumerate(imps[:M]):
        Vi = V[idx, :P]
        B = cd.chained_impute(Blo, Bhi, np.column_stack([Z, Vi / LN10, S, d]), g, n_sweeps=3, seed=900 + mm)
        for k, c in enumerate(s3.CMP):
            if c in ("PFOSA",):
                pass
            y = B[:, k] / LN10
            for adj, W in [("total", np.column_stack([Z, Vi / LN10])), ("solids_adjusted", np.column_stack([Z, Vi / LN10, S]))]:
                mh, eh = cd.crossfit_nuisance(W, y, d, g, n_splits=5)
                th, var = cd.plr(y, d, mh, eh, g)
                res.setdefault((c, adj), []).append((th, var))
        pfos = np.exp(B[:, s3.CMP.index("PFOS")])
        res.setdefault(("PFOS_gt50", "descriptive"), []).append(
            (np.mean(pfos[d == 1] > 50) - np.mean(pfos[d == 0] > 50), 0.0))
    for (c, adj), lst in res.items():
        th, se = cd.rubin([x[0] for x in lst], [x[1] for x in lst])
        rows.append(dict(outcome=c, adjustment=adj, theta=th, se=se, n=len(sub)))
        print("BIOS", c, adj, round(th, 3), round(se, 3), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pr = s3.load_design()
    imps = list(np.load(R / "s03_imputations.npz")["imps"])
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "process"):
        process_contrasts(pr, imps).to_csv(R / "s05_process.csv", index=False)
    if which in ("all", "stage"):
        st = stage_test()
        st.to_csv(R / "s05_stage.csv", index=False)
        print(st.round(3).to_string())
    if which in ("all", "bios"):
        biosolids(pr, imps).to_csv(R / "s05_biosolids.csv", index=False)
