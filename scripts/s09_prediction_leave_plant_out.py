"""Leave-plant-out prediction of effluent PFAS (left-censored) from the influent
profile, process type and plant covariates.

Baselines: persistence (EFF = INF, DL/2), process-ratio (INF x training median
ratio of the process class), linear Tobit. ML: XGBoost-AFT (survival boosting),
stack (mean of Tobit and AFT location).
Metrics on held-out plants: R2/RMSE on detected effluent (log10), censoring-aware
concordance (Harrell C for left-censored data), mean censored NLL.
Output: results/s09_prediction.csv, results/s09_oof.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import censdml as cd  # noqa: E402
import s03_upgrade_effect as s3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
LN10 = np.log(10)
PROCS = ["primary", "lagoon", "AS", "biofilm", "BNR", "ext_aer"]


def cindex_left(mu, lo, hi):
    """Concordance for left-censored outcomes: pair (i, j) comparable if both
    observed, or one observed and the other censored below it."""
    obs = np.isfinite(lo)
    val = np.where(obs, lo, hi)
    num = den = 0.0
    n = len(mu)
    for i in range(n):
        if not obs[i]:
            continue
        # j lower than i: observed with smaller value, or censored with DL <= value_i
        lower = (obs & (val < val[i])) | (~obs & (hi <= val[i]))
        lower[i] = False
        den += lower.sum()
        num += (mu[i] > mu[lower]).sum() + 0.5 * (mu[i] == mu[lower]).sum()
    return num / den if den else np.nan


def cens_nll(mu, sig, lo, hi):
    obs = np.isfinite(lo)
    ll = np.where(obs, stats.norm.logpdf(np.where(obs, lo, 0), mu, sig), stats.norm.logcdf((hi - mu) / sig))
    return -np.mean(ll)


def main():
    pr = s3.load_design()
    g = pr.WWTP_CODE.values
    ilo, ihi = s3.bounds(pr, "INF")
    elo, ehi = s3.bounds(pr, "EFF")
    imid = np.where(np.isfinite(ilo), ilo, ihi - np.log(2))
    icen = (~np.isfinite(ilo)).astype(float)
    P1 = np.column_stack([(pr.proc == p).astype(float) for p in PROCS])
    X = np.column_stack([pr[s3.ZCOLS].values.astype(float), P1, imid / LN10, icen])
    rows, oof_rows = [], []
    gkf = GroupKFold(n_splits=10)
    for k, c in enumerate(s3.CMP):
        lo, hi = elo[:, k], ehi[:, k]
        pred = {m: np.zeros(len(pr)) for m in ["persistence", "process-ratio", "Tobit", "XGB-AFT", "Stack",
                                               "Anchored-Tobit", "Anchored-AFT", "Anchored-Stack"]}
        sig = {}
        for tr, te in gkf.split(X, groups=g):
            pred["persistence"][te] = imid[te, k]
            obs_tr = np.isfinite(lo[tr])
            ratio = pd.Series((np.where(obs_tr, lo[tr], hi[tr] - np.log(2)) - imid[tr, k])).groupby(pr.proc.values[tr]).median()
            pred["process-ratio"][te] = imid[te, k] + pr.proc.iloc[te].map(ratio).fillna(ratio.median()).values
            tob = cd.TobitLinear(alpha=1.0).fit(X[tr], lo[tr], hi[tr])
            pred["Tobit"][te] = tob.predict(X[te])
            bst = cd.fit_aft(X[tr], lo[tr], hi[tr], seed=k)
            pred["XGB-AFT"][te] = cd.predict_mu(bst, X[te])
            pred["Stack"][te] = 0.5 * (pred["Tobit"][te] + pred["XGB-AFT"][te])
            off_tr, off_te = imid[tr, k], imid[te, k]
            atob = cd.TobitLinear(alpha=1.0).fit(X[tr], lo[tr] - off_tr, hi[tr] - off_tr)
            pred["Anchored-Tobit"][te] = cd.predict_mu(atob, X[te], off_te)
            abst = cd.fit_aft(X[tr], lo[tr], hi[tr], seed=k, offset=off_tr)
            pred["Anchored-AFT"][te] = cd.predict_mu(abst, X[te], off_te)
            pred["Anchored-Stack"][te] = 0.5 * (pred["Anchored-Tobit"][te] + pred["Anchored-AFT"][te])
        obs = np.isfinite(lo)
        for m, mu in pred.items():
            s_ = cd.tobit_sigma(lo[obs] - mu[obs], hi[~obs] - mu[~obs])
            yt, yp = lo[obs] / LN10, mu[obs] / LN10
            r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
            rows.append(dict(cmp=c, model=m, R2_detected=r2, RMSE_log10=np.sqrt(np.mean((yt - yp) ** 2)),
                             C_index=cindex_left(mu, lo, hi), cens_NLL=cens_nll(mu, s_, lo, hi),
                             n=len(pr), pct_censored=100 * (~obs).mean()))
            oof_rows.append(pd.DataFrame(dict(cmp=c, model=m, plant=g, proc=pr.proc.values, mu_log10=mu / LN10,
                                              lo_log10=lo / LN10, hi_log10=hi / LN10)))
        print(pd.DataFrame(rows[-8:])[["cmp", "model", "R2_detected", "RMSE_log10", "C_index", "cens_NLL"]].round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(R / "s09_prediction.csv", index=False)
    pd.concat(oof_rows).to_parquet(R / "s09_oof.parquet")


if __name__ == "__main__":
    main()
