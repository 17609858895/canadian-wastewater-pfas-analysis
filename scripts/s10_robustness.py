"""Robustness of the pooled sumPFCA / PFHxA / PFOA effects (LOD-DML, s03 imputations)
(a) leave-one-primary-plant-out
(b) campaign-level aggregation (plant x campaign means; removes 3-day pseudo-replication)
(c) omitted-variable robustness value (Cinelli-Hazlett form on the cluster t-statistic)
Output: results/s10_robustness.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import censdml as cd  # noqa: E402
import s03_upgrade_effect as s3  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
P = len(s3.CMP)
LN10 = np.log(10)
OUTS = ["sumPFCA", "PFHxA", "PFOA", "sumPFSA"]


def est(y, d, W, g):
    mh, eh = cd.crossfit_nuisance(W, y, d, g, n_splits=5)
    return cd.plr(y, d, mh, eh, g)


def main(M=5):
    pr = s3.load_design()
    imps = list(np.load(R / "s03_imputations.npz")["imps"])[:M]
    g, d, Z = pr.WWTP_CODE.values, pr.bio.values, pr[s3.ZCOLS].values.astype(float)
    # campaign id: same plant, dates within 10 days
    pr = pr.sort_values(["WWTP_CODE", "date"])
    camp = (pr.groupby("WWTP_CODE").date.diff().dt.days.fillna(999) > 10).cumsum()
    camp = camp.sort_index().values
    rows = []
    for name in OUTS:
        res = {}
        for V in imps:
            y = s3.outcomes(V[:, :P], V[:, P:])[name]
            W = np.column_stack([Z, V[:, :P] / LN10])
            for plant in sorted(set(g[d == 0])):
                k = g != plant
                res.setdefault(f"drop_{plant}", []).append(est(y[k], d[k], W[k], g[k]))
            df = pd.DataFrame(W).assign(y=y, d=d, g=g, c=camp)
            agg = df.groupby("c").mean(numeric_only=True)
            gg = df.groupby("c").g.first().values
            res.setdefault("campaign_level", []).append(est(agg.y.values, agg.d.values, agg.drop(columns=["y", "d"]).values, gg))
            res.setdefault("full", []).append(est(y, d, W, g))
        for lab, lst in res.items():
            th, se = cd.rubin([a[0] for a in lst], [a[1] for a in lst])
            r = dict(outcome=name, analysis=lab, theta=th, se=se)
            if lab in ("full", "campaign_level"):
                G = len(np.unique(g)) - 1
                f = (th / se) / np.sqrt(G)
                r["robustness_value"] = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(R / "s10_robustness.csv", index=False)


if __name__ == "__main__":
    main()
