"""Supplementary analyses for Figs. S3, S4, S6-S9.

  balance   propensity overlap and covariate balance; omitted-confounder contour
            inputs (residual SDs, benchmark partial R2)          -> s13_balance*.csv
  imputed   imputation diagnostics from s03 imputations          -> s13_imputed.csv
  flags     lab-qualifier sensitivity (K/B/H/NQ as upper bounds;
            or pairs with qualified values dropped)              -> s13_flags.csv
  lag       hydraulic-lag pairing (INF day d with EFF day d+1)   -> s13_lag.csv
  extended  2018+ extended-panel compounds (LOD-DML, M = 5)       -> s13_extended.csv
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
P = len(s3.CMP)
LN10 = np.log(10)
LOC = {"Influent": "INF", "Effluent": "EFF"}


def wide(variant="main"):
    e = pd.read_parquet(D / "env_long.parquet")
    e = e[~e.is_dup & e.usable & e.LOCATION.isin(LOC)].copy()
    e["loc"] = e.LOCATION.map(LOC)
    cens = e.censored.values.copy()
    hi = np.where(cens, e.dl, e.value)
    lo = np.where(cens, 0.0, e.value)
    flagged = (e.flag_K | e.flag_B | e.flag_H | e.flag_NQ).values & ~cens
    if variant == "flag_upper":       # qualified detections become [0, value]
        lo = np.where(flagged, 0.0, lo)
    elif variant == "flag_drop":      # qualified detections removed
        lo = np.where(flagged, np.nan, lo)
        hi = np.where(flagged, np.nan, hi)
    e["lo"], e["hi"] = lo, hi
    key = ["WWTP_CODE", "date", "loc", "AXYS_ID"]
    e = e.drop_duplicates(key + ["COMPOUND"])
    L = e.set_index(key + ["COMPOUND"])["lo"].unstack("COMPOUND")
    H = e.set_index(key + ["COMPOUND"])["hi"].unstack("COMPOUND")
    L.columns = [f"lo|{c}" for c in L.columns]
    H.columns = [f"hi|{c}" for c in H.columns]
    w = pd.concat([L, H], axis=1).reset_index()
    return w.groupby(["WWTP_CODE", "date", "loc"]).filter(lambda x: len(x) == 1), int(flagged.sum())


def pairs_from(w, lag_days=0):
    inf = w[w["loc"] == "INF"].drop(columns=["loc", "AXYS_ID"]).copy()
    eff = w[w["loc"] == "EFF"].drop(columns=["loc", "AXYS_ID"]).copy()
    eff["date"] = eff["date"] - pd.Timedelta(days=lag_days)  # align EFF(d+lag) with INF(d)
    pr = inf.set_index(["WWTP_CODE", "date"]).add_prefix("INF_").join(
        eff.set_index(["WWTP_CODE", "date"]).add_prefix("EFF_"), how="inner").reset_index()
    des = s3.load_design()
    keep = ["WWTP_CODE", "date", "bio", "proc"] + s3.ZCOLS
    pr = pr.merge(des[keep], on=["WWTP_CODE", "date"], how="inner")
    cols = [f"{s}_hi|{c}" for s in ["INF", "EFF"] for c in s3.CMP]
    return pr[pr[cols].notna().all(axis=1)].reset_index(drop=True)


def lod_dml(pr, M=5, outs=("sumPFCA", "PFHxA", "PFOA", "PFDA", "PFOS", "sumPFSA"), seed=0):
    g, d = pr.WWTP_CODE.values, pr.bio.values
    Z = pr[s3.ZCOLS].values.astype(float)
    ilo, ihi = s3.bounds(pr, "INF")
    elo, ehi = s3.bounds(pr, "EFF")
    Vlo, Vhi = np.hstack([ilo, elo]), np.hstack([ihi, ehi])
    res = {}
    for m in range(M):
        V = cd.chained_impute(Vlo, Vhi, np.column_stack([Z, d]), g, n_sweeps=3, seed=seed + m)
        ys = s3.outcomes(V[:, :P], V[:, P:])
        W = np.column_stack([Z, V[:, :P] / LN10])
        for o in outs:
            mh, eh = cd.crossfit_nuisance(W, ys[o], d, g)
            res.setdefault(o, []).append(cd.plr(ys[o], d, mh, eh, g))
    rows = []
    for o, lst in res.items():
        th, se = cd.rubin([a[0] for a in lst], [a[1] for a in lst])
        rows.append(dict(outcome=o, theta=th, se=se, n=len(pr), plants=len(np.unique(g)),
                         primary_plants=len(np.unique(g[d == 0]))))
    return rows


def run_flags():
    rows = []
    for v in ["flag_upper", "flag_drop"]:
        w, nflag = wide(v)
        pr = pairs_from(w)
        print(v, "qualified detections", nflag, "pairs", len(pr), flush=True)
        for r in lod_dml(pr, seed=300):
            rows.append(dict(variant=v, n_qualified=nflag, **r))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "s13_flags.csv", index=False)


def run_lag():
    w, _ = wide("main")
    rows = []
    for lag in [1, 2]:
        pr = pairs_from(w, lag_days=lag)
        print("lag", lag, "pairs", len(pr), flush=True)
        for r in lod_dml(pr, seed=400 + lag):
            rows.append(dict(lag_days=lag, **r))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "s13_lag.csv", index=False)


EXT = ["6:2 FTS", "8:2 FTS", "EtFOSAA", "MeFOSAA", "N-EtFOSE", "PFPeS", "PFHpS", "PFDS", "PFNS"]


def run_extended(M=5):
    pr = pd.read_parquet(D / "pairs_inf_eff.parquet")
    des = s3.load_design()
    pr = pr.merge(des[["WWTP_CODE", "date", "bio"] + s3.ZCOLS], on=["WWTP_CODE", "date"])
    pr = pr[pr.date.dt.year >= 2018]
    allc = s3.CMP + EXT
    pr = pr[pr[[f"{s}_hi|{c}" for s in ["INF", "EFF"] for c in allc]].notna().all(axis=1)].reset_index(drop=True)
    det = {c: int((pr[f"INF_lo|{c}"] > 0).sum() + (pr[f"EFF_lo|{c}"] > 0).sum()) for c in allc}
    use = [c for c in allc if det[c] >= 5]
    print("2018+ pairs", len(pr), "imputed", use, flush=True)
    lo, hi = [], []
    for s in ["INF", "EFF"]:
        l_ = np.column_stack([pr[f"{s}_lo|{c}"].values for c in use]).astype(float)
        h_ = np.column_stack([pr[f"{s}_hi|{c}"].values for c in use]).astype(float)
        with np.errstate(divide="ignore"):
            lo.append(np.where(l_ > 0, np.log(l_), -np.inf))
        hi.append(np.log(h_))
    Vlo, Vhi = np.hstack(lo), np.hstack(hi)
    g, d = pr.WWTP_CODE.values, pr.bio.values
    Z = pr[s3.ZCOLS].values.astype(float)
    q = len(use)
    targets = [c for c in EXT if c in use]
    res = {}
    for m in range(M):
        V = cd.chained_impute(Vlo, Vhi, np.column_stack([Z, d]), g, n_sweeps=3, seed=800 + m)
        W = np.column_stack([Z, V[:, [use.index(c) for c in s3.CMP]] / LN10])
        for c in targets:
            k = use.index(c)
            y = (V[:, q + k] - V[:, k]) / LN10
            mh, eh = cd.crossfit_nuisance(W, y, d, g)
            res.setdefault(c, []).append(cd.plr(y, d, mh, eh, g))
        print("imp", m, flush=True)
    rows = []
    for c, lst in res.items():
        th, se = cd.rubin([a[0] for a in lst], [a[1] for a in lst])
        rows.append(dict(outcome=c, theta=th, se=se, n=len(pr), detections=det[c],
                         nd_inf=float((pr[f"INF_lo|{c}"] == 0).mean()), nd_eff=float((pr[f"EFF_lo|{c}"] == 0).mean())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "s13_extended.csv", index=False)


def run_balance():
    pr = s3.load_design()
    V = np.load(R / "s03_imputations.npz")["imps"][0]
    g, d = pr.WWTP_CODE.values, pr.bio.values
    Z = pr[s3.ZCOLS].values.astype(float)
    W = np.column_stack([Z, V[:, :P] / LN10])
    names = s3.ZCOLS + [f"log INF {c}" for c in s3.CMP]
    y = s3.outcomes(V[:, :P], V[:, P:])["sumPFCA"]
    mh, eh = cd.crossfit_nuisance(W, y, d, g)
    th, var = cd.plr(y, d, mh, eh, g)
    pd.DataFrame(dict(plant=g, bio=d, e_hat=eh, y=y, m_hat=mh)).to_csv(R / "s13_propensity.csv", index=False)
    # standardized mean differences: raw and overlap-weighted (treated 1-e, control e)
    wts = np.where(d == 1, 1 - eh, eh)
    rows = []
    for j, n in enumerate(names):
        x = W[:, j]
        s = np.sqrt((x[d == 1].var() + x[d == 0].var()) / 2) + 1e-12
        raw = (x[d == 1].mean() - x[d == 0].mean()) / s
        wm1 = np.average(x[d == 1], weights=wts[d == 1]); wm0 = np.average(x[d == 0], weights=wts[d == 0])
        rows.append(dict(covariate=n, smd_raw=raw, smd_overlap=(wm1 - wm0) / s))
    pd.DataFrame(rows).to_csv(R / "s13_balance.csv", index=False)
    # omitted-variable bias inputs (Cinelli-Hazlett, partially linear form)
    ry, rd = y - mh, d - eh
    bench = {"Plant size": ["log_flow", "flow_na"], "Landfill leachate": ["leach"], "Receiving water":
             ["rw_Great", "rw_St.", "rw_Atlantic", "rw_Pacific", "rw_Hudson"], "Sampling year": ["year", "filtered", "comp30min"]}
    brows = []
    for lab, cols in bench.items():
        idx = [names.index(c) for c in cols]
        Xb = W[:, idx]; Xo = np.delete(W, idx, axis=1)
        Xo1 = np.column_stack([np.ones(len(d)), Xo])
        res = Xb - Xo1 @ np.linalg.lstsq(Xo1, Xb, rcond=None)[0]

        def pr2(target):
            Xr = np.column_stack([np.ones(len(d)), res])
            fit = Xr @ np.linalg.lstsq(Xr, target, rcond=None)[0]
            return 1 - np.sum((target - fit) ** 2) / np.sum((target - target.mean()) ** 2)
        brows.append(dict(benchmark=lab, r2_d=pr2(rd), r2_y=pr2(ry)))
    meta = dict(theta=th, se=np.sqrt(var), sd_ry=ry.std(), sd_rd=rd.std(), n=len(d), plants=len(np.unique(g)))
    pd.DataFrame(brows).assign(**meta).to_csv(R / "s13_ovb.csv", index=False)
    print(pd.DataFrame(brows), meta)


def rv_from(ry, rd, theta):
    """Cinelli-Hazlett robustness value from partial Cohen's f^2 of D in the outcome model."""
    r2 = theta ** 2 * rd.var() / ry.var()
    f2 = r2 / (1 - r2)
    return 0.5 * (np.sqrt(f2 ** 2 + 4 * f2) - f2), r2


def run_rv(M=5):
    """Correct robustness values (replace the cluster-df version printed in s10)."""
    pr = s3.load_design()
    imps = list(np.load(R / "s03_imputations.npz")["imps"])[:M]
    g, d, Z = pr.WWTP_CODE.values, pr.bio.values, pr[s3.ZCOLS].values.astype(float)
    srt = pr.sort_values(["WWTP_CODE", "date"])
    camp = (srt.groupby("WWTP_CODE").date.diff().dt.days.fillna(999) > 10).cumsum().sort_index().values
    rows = []
    for o in ["sumPFCA", "PFHxA", "PFOA"]:
        acc = {"full": [], "campaign_level": []}
        for V in imps:
            y = s3.outcomes(V[:, :P], V[:, P:])[o]
            W = np.column_stack([Z, V[:, :P] / LN10])
            mh, eh = cd.crossfit_nuisance(W, y, d, g)
            th, _ = cd.plr(y, d, mh, eh, g)
            acc["full"].append(rv_from(y - mh, d - eh, th) + (th,))
            df = pd.DataFrame(W).assign(y=y, d=d, c=camp)
            agg = df.groupby("c").mean(numeric_only=True)
            gg = pd.Series(g).groupby(camp).first().values
            Wc = agg.drop(columns=["y", "d"]).values
            mh, eh = cd.crossfit_nuisance(Wc, agg.y.values, agg.d.values, gg)
            th, _ = cd.plr(agg.y.values, agg.d.values, mh, eh, gg)
            acc["campaign_level"].append(rv_from(agg.y.values - mh, agg.d.values - eh, th) + (th,))
        for an, lst in acc.items():
            a = np.array(lst)
            rows.append(dict(outcome=o, analysis=an, robustness_value=a[:, 0].mean(), r2_yd=a[:, 1].mean(), theta=a[:, 2].mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "s13_rv.csv", index=False)


def run_imputed():
    z = np.load(R / "s03_imputations.npz")
    imps, Vlo, Vhi = z["imps"], z["Vlo"], z["Vhi"]
    rows = []
    for j in range(Vlo.shape[1]):
        side = "INF" if j < P else "EFF"
        c = s3.CMP[j % P]
        cens = ~np.isfinite(Vlo[:, j])
        obs = (Vlo[~cens, j] - np.log(np.nanmedian(np.exp(Vhi[cens, j])) if cens.any() else 0)) / LN10
        if cens.any():
            rel = (imps[:, cens, j] - Vhi[cens, j][None, :]) / LN10  # log10(imputed/DL) <= 0
            between = imps[:, cens, j].std(axis=0) / LN10
            rows.append(dict(compound=c, side=side, pct_censored=100 * cens.mean(), med_rel=np.median(rel),
                             q10_rel=np.percentile(rel, 10), q90_rel=np.percentile(rel, 90),
                             between_sd=np.median(between), obs_rel_med=np.median(obs) if len(obs) else np.nan))
    pd.DataFrame(rows).to_csv(R / "s13_imputed.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    {"flags": run_flags, "lag": run_lag, "extended": run_extended, "balance": run_balance,
     "imputed": run_imputed, "rv": run_rv}[sys.argv[1]]()
