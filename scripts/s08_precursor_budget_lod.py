"""LOD-aware molar budget for 2018+ pairs (extended panel): how much of the PFAA
gain across biological treatment can the measured precursors explain?

Compounds with < 5 detections in both matrices are not imputed (no support);
they enter only through the substitution bounds reported in s06.
Output: results/s08_budget_lod.csv (per-imputation) and printed summary.
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
# constants copied from s06_precursor_budget.py
MW=dict(PFBA=214.04,PFPeA=264.05,PFHxA=314.05,PFHpA=364.06,PFOA=414.07,PFNA=464.08,PFDA=514.08,PFUnA=564.09,PFDoA=614.1,
 PFBS=300.1,PFPeS=350.1,PFHxS=400.11,PFHpS=450.12,PFOS=500.13,PFNS=550.14,PFDS=600.14,
 **{"4:2 FTS":328.1,"6:2 FTS":428.17,"8:2 FTS":528.18,"EtFOSAA":585.24,"MeFOSAA":571.21,"N-EtFOSE":571.25,"N-MeFOSE":557.23,
    "N-EtFOSA":527.2,"N-MeFOSA":513.17,"PFOSA":499.14,"5:3 FTCA":342.1,"3:3 FTCA":242.1,"7:3 FTCA":442.1})
PFAA=["PFBA","PFPeA","PFHxA","PFHpA","PFOA","PFNA","PFDA","PFUnA","PFDoA","PFBS","PFPeS","PFHxS","PFHpS","PFOS","PFNS","PFDS"]
PREC=["4:2 FTS","6:2 FTS","8:2 FTS","EtFOSAA","MeFOSAA","N-EtFOSE","N-MeFOSE","N-EtFOSA","N-MeFOSA","PFOSA"]
FT = ["4:2 FTS", "6:2 FTS", "8:2 FTS"]
ECF = ["EtFOSAA", "MeFOSAA", "N-EtFOSE", "N-MeFOSE", "N-EtFOSA", "N-MeFOSA", "PFOSA"]
PFCA_ALL = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA", "PFUnA", "PFDoA"]
PFSA_ALL = ["PFBS", "PFPeS", "PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS"]


def main(M=10):
    pr = pd.read_parquet(D / "pairs_inf_eff.parquet")
    des = s3.load_design()
    pr = pr.merge(des[["WWTP_CODE", "date", "bio"] + s3.ZCOLS], on=["WWTP_CODE", "date"])
    pr = pr[pr.date.dt.year >= 2018]
    allc = PREC + PFAA
    pr = pr[pr[[f"{s}_hi|{c}" for s in ["INF", "EFF"] for c in allc]].notna().all(axis=1)].reset_index(drop=True)
    det = {c: int((pr[f"INF_lo|{c}"] > 0).sum() + (pr[f"EFF_lo|{c}"] > 0).sum()) for c in allc}
    use = [c for c in allc if det[c] >= 5]
    print("pairs", len(pr), "plants", pr.WWTP_CODE.nunique(), "imputed compounds", len(use))
    print("dropped (<5 detections):", [c for c in allc if c not in use])
    lo, hi = [], []
    for s in ["INF", "EFF"]:
        l = np.column_stack([pr[f"{s}_lo|{c}"].values for c in use]).astype(float)
        h = np.column_stack([pr[f"{s}_hi|{c}"].values for c in use]).astype(float)
        with np.errstate(divide="ignore"):
            lo.append(np.where(l > 0, np.log(l), -np.inf))
        hi.append(np.log(h))
    Vlo, Vhi = np.hstack(lo), np.hstack(hi)
    g, d = pr.WWTP_CODE.values, pr.bio.values
    Z = np.column_stack([pr[s3.ZCOLS].values.astype(float), d])
    q = len(use)
    rows = []
    for m in range(M):
        V = cd.chained_impute(Vlo, Vhi, Z, g, n_sweeps=3, seed=700 + m)
        ci, ce = np.exp(V[:, :q]), np.exp(V[:, q:])

        def delta(group):
            ks = [use.index(c) for c in group if c in use]
            return np.sum([(ce[:, k] - ci[:, k]) / MW[use[k]] for k in ks], axis=0) if ks else np.zeros(len(pr))

        df = pd.DataFrame(dict(plant=g, bio=d, dPFCA=delta(PFCA_ALL), dPFSA=delta(PFSA_ALL), dFT=delta(FT), dECF=delta(ECF)))
        df["dPFAA"], df["dPrec"] = df.dPFCA + df.dPFSA, df.dFT + df.dECF
        pm = df.groupby(["plant", "bio"]).median().reset_index()
        for grp, gp in pm.groupby("bio"):
            rows.append(dict(imp=m, group="biological" if grp == 1 else "primary", plants=len(gp),
                             dPFAA=gp.dPFAA.median(), dPFCA=gp.dPFCA.median(), dPrec=gp.dPrec.median(),
                             frac_all=(-gp.dPrec.clip(upper=0)).sum() / gp.dPFAA.clip(lower=0).sum(),
                             frac_FT_to_PFCA=(-gp.dFT.clip(upper=0)).sum() / gp.dPFCA.clip(lower=0).sum(),
                             frac_ECF_to_PFSA=(-gp.dECF.clip(upper=0)).sum() / max(gp.dPFSA.clip(lower=0).sum(), 1e-9)))
        print(rows[-2:], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R / "s08_budget_lod.csv", index=False)
    print(out.groupby("group").agg(["median", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)]).round(4).T.to_string())


if __name__ == "__main__":
    main()
