"""Semi-synthetic benchmark: known treatment effect on log10(EFF/INF) under the
real design (plants, covariates, treatment, sample-specific detection limits).

Latent influent = one completed data set from s03 (+ replicate noise).
Latent effluent = influent + confounding h(Z, INF) + theta*d + noise.
Censoring uses the real DL of each sample (detected samples: median DL of the
same compound/matrix/year), optionally multiplied by k (MDL stress).
Methods: oracle, DL/2, DL/sqrt2, complete-case, LOD-DML (M=5).
Output: results/s04_benchmark.csv
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import censdml as cd  # noqa: E402
import s03_upgrade_effect as s3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BCMP = ["PFBA", "PFHxA", "PFNA", "PFDA", "PFOS"]
THETA_B = dict(PFBA=0.25, PFHxA=0.25, PFNA=0.25, PFDA=0.25, PFOS=0.0)
LN10 = np.log(10)


def dl_matrix(pr, side):
    out = []
    for c in s3.CMP:
        cens = pr[f"{side}_lo|{c}"] == 0
        dl = pr[f"{side}_hi|{c}"].where(cens)
        yr = dl.groupby(pr.year).transform("median")
        dl = dl.fillna(yr).fillna(dl.median())
        out.append(np.log(dl.values))
    return np.column_stack(out)


def setup():
    pr = s3.load_design()
    imp = np.load(R / "s03_imputations.npz")["imps"][0]
    p = len(s3.CMP)
    return dict(pr=pr, Vi0=imp[:, :p], dli=dl_matrix(pr, "INF"), dle=dl_matrix(pr, "EFF"),
                Z=pr[s3.ZCOLS].values.astype(float), d=pr.bio.values, g=pr.WWTP_CODE.values)


S = None


def init():
    global S
    cd.LIGHT = True
    S = setup()


def one_rep(args):
    scen, k, rep = args
    rng = np.random.default_rng(10000 * k + 100 * (scen == "effect") + rep)
    Z, d, g = S["Z"], S["d"], S["g"]
    n = len(d)
    idx = [s3.CMP.index(c) for c in BCMP]
    zf = (Z[:, s3.ZCOLS.index("log_flow")] - 4.5)
    pres = (Z[:, s3.ZCOLS.index("pres")] - 85) / 15
    filt = Z[:, s3.ZCOLS.index("filtered")]
    Vi_lat = S["Vi0"][:, idx] + rng.normal(0, 0.2, (n, len(idx)))
    Ve_lat = np.empty_like(Vi_lat)
    for j, c in enumerate(BCMP):
        th = THETA_B[c] if scen == "effect" else 0.0
        li = Vi_lat[:, j] / LN10
        h = 0.15 * np.tanh(zf) + 0.10 * filt - 0.08 * pres - 0.12 * (li - li.mean()) + 0.08 * zf * filt
        Ve_lat[:, j] = Vi_lat[:, j] + LN10 * (h + th * d) + rng.normal(0, 0.35 * LN10, n)
    ldi = S["dli"][:, idx] + np.log(k)
    lde = S["dle"][:, idx] + np.log(k)
    ci, ce = Vi_lat <= ldi, Ve_lat <= lde
    Vlo = np.hstack([np.where(ci, -np.inf, Vi_lat), np.where(ce, -np.inf, Ve_lat)])
    Vhi = np.hstack([np.where(ci, ldi, Vi_lat), np.where(ce, lde, Ve_lat)])
    others = np.delete(S["Vi0"], idx, axis=1) / LN10
    Zd = np.column_stack([Z, others, d])
    imps = [cd.chained_impute(Vlo, Vhi, Zd, g, n_sweeps=3, seed=rep * 31 + m) for m in range(3)]
    imps_tob = [cd.chained_impute(Vlo, Vhi, Zd, g, n_sweeps=3, seed=rep * 37 + m, engine="tobit") for m in range(3)]
    imps_nod = [cd.chained_impute(Vlo, Vhi, Zd[:, :-1], g, n_sweeps=3, seed=rep * 41 + m) for m in range(3)]
    p = len(BCMP)
    rows = []
    for method in ["oracle", "DL/2", "DL/sqrt2", "complete-case", "LOD-DML", "ablation:Tobit-MI", "ablation:no-D-MI"]:
        if method == "oracle":
            sets = [np.hstack([Vi_lat, Ve_lat])]
        elif method in ("DL/2", "complete-case"):
            sets = [np.where(np.isfinite(Vlo), Vlo, Vhi - np.log(2))]
        elif method == "DL/sqrt2":
            sets = [np.where(np.isfinite(Vlo), Vlo, Vhi - 0.5 * np.log(2))]
        elif method == "LOD-DML":
            sets = imps
        elif method == "ablation:Tobit-MI":
            sets = imps_tob
        else:
            sets = imps_nod
        for j, c in enumerate(BCMP):
            ths, vs = [], []
            keep = np.ones(n, bool)
            if method == "complete-case":
                keep = ~ci[:, j] & ~ce[:, j]
            for V in sets:
                y = (V[:, p + j] - V[:, j]) / LN10
                W = np.column_stack([Z, others, V[:, :p] / LN10])
                mh, eh = cd.crossfit_nuisance(W[keep], y[keep], d[keep], g[keep])
                th, var = cd.plr(y[keep], d[keep], mh, eh, g[keep])
                ths.append(th)
                vs.append(var)
            th, se = cd.rubin(ths, vs)
            truth = THETA_B[c] if scen == "effect" else 0.0
            rows.append(dict(scen=scen, k=k, rep=rep, method=method, cmp=c, theta=th, se=se, truth=truth,
                             cens_inf=ci[:, j].mean(), cens_eff=ce[:, j].mean(), n=keep.sum()))
    return rows


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    R_ = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tasks = [(sc, k, r) for sc in ["null", "effect"] for k in [1, 3] for r in range(R_)]
    with Pool(12, initializer=init) as pool:
        out = []
        for i, rows in enumerate(pool.imap_unordered(one_rep, tasks)):
            out.extend(rows)
            if (i + 1) % 8 == 0:
                pd.DataFrame(out).to_csv(R / "s04_benchmark.csv", index=False)
                print(f"{i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(R / "s04_benchmark.csv", index=False)
    df["err"] = df.theta - df.truth
    df["cover"] = (abs(df.err) <= 1.96 * df.se)
    print(df.groupby(["scen", "k", "method", "cmp"]).agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                                      cover=("cover", "mean"), ci=("cens_inf", "mean"), ce=("cens_eff", "mean")).round(3).to_string())
