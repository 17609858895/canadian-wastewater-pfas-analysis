"""Complementary precursor assessment with the ECCC TOP assay data.

The TOP release (PFASTOPBiosolids_EN_FR.csv, generated 2025-05-13) contains
biosolids only (2024, 11 biological plants). It therefore cannot validate the
wastewater estimate directly; it tests two consistency predictions:
  (1) in biosolids, the oxidizable precursor pool revealed by TOP
      (post-TOP minus pre-TOP sumPFCA, molar) is much larger than the
      targeted precursors measured before oxidation;
  (2) plants whose biosolids hold a larger hidden pool show more unexplained
      PFCA formation across their 2024 liquid treatment.
Non-detects are bounded (ND = 0 lower, ND = DL upper); conclusions are
reported for both bounds.
Outputs: results/s15_top_samples.csv, s15_top_summary.csv, s15_top_plant_link.csv
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "PFASTOPBiosolids_EN_FR.csv"
D, R = ROOT / "data_clean", ROOT / "results"

MW = dict(PFBA=214.04, PFPeA=264.05, PFHxA=314.05, PFHpA=364.06, PFOA=414.07, PFNA=464.08, PFDA=514.08,
          PFUnA=564.09, PFDoA=614.10, PFTrDA=664.10, PFTeDA=714.11,
          PFBS=300.10, PFPeS=350.11, PFHxS=400.11, PFHpS=450.12, PFOS=500.13, PFNS=550.14, PFDS=600.14, PFDoS=700.16,
          **{"4:2 FTS": 328.13, "6:2 FTS": 428.17, "8:2 FTS": 528.18, "3:3 FTCA": 242.11, "5:3 FTCA": 342.12,
             "7:3 FTCA": 442.14, "EtFOSAA": 585.24, "MeFOSAA": 571.21, "N-EtFOSE": 571.25, "N-MeFOSE": 557.23,
             "N-EtFOSA": 527.20, "N-MeFOSA": 513.17, "PFOSA": 499.14})
PFCA = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA", "PFUnA", "PFDoA", "PFTrDA", "PFTeDA"]
PFSA = ["PFBS", "PFPeS", "PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS", "PFDoS"]
PREC = ["4:2 FTS", "6:2 FTS", "8:2 FTS", "3:3 FTCA", "5:3 FTCA", "7:3 FTCA", "EtFOSAA", "MeFOSAA", "N-EtFOSE",
        "N-MeFOSE", "N-EtFOSA", "N-MeFOSA", "PFOSA"]
TOPNAME = {"NEtFOSA": "N-EtFOSA", "NEtFOSAA": "EtFOSAA", "NEtFOSE": "N-EtFOSE", "NMeFOSA": "N-MeFOSA",
           "NMeFOSAA": "MeFOSAA", "NMeFOSE": "N-MeFOSE"}


def read_top():
    lines = RAW.read_bytes().decode("cp1252").splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("SAMPLE_NAME,WWTP_CODE"))
    body = []
    for l in lines[h + 1:]:
        if not l.strip(",").strip():
            break
        body.append(l)
    t = pd.read_csv(io.StringIO("\n".join([lines[h]] + body)), dtype=str)
    t = t[(t.LOCATION == "Biosolids") & (t.UNIT == "ng/g (dry weight basis)")].copy()
    t = t[~t.SAMPLE_NAME.str.contains("duplicate", case=False, na=False)]
    t["cmp"] = t.COMPOUND.str.replace("_POSTTOP", "", regex=False).replace(TOPNAME)
    flag = t.LAB_FLAG.fillna("")
    cens = t.CONCENTRATION_FOUND.str.contains("Non-Detect", case=False, na=False) | flag.str.contains(r"(?:^|\s)U", regex=True)
    val = pd.to_numeric(t.CONCENTRATION_FOUND, errors="coerce")
    dl = pd.to_numeric(t.DETECTION_LIMIT, errors="coerce")
    t["lo"] = np.where(cens, 0.0, val); t["hi"] = np.where(cens, dl, val)
    t["date"] = pd.to_datetime(t.COLLECTION_DATE, errors="coerce")
    return t


def molar(df, cmps, col):
    s = 0.0
    for c in cmps:
        if c in df.index:
            v = df.loc[c, col]
            s = s + (0 if pd.isna(v) else v / MW[c])
    return s


def main():
    top = read_top()
    pre = pd.read_parquet(D / "env_long.parquet")
    pre = pre[(pre.LOCATION == "Biosolids") & ~pre.is_dup & pre.usable].copy()
    pre["lo"] = np.where(pre.censored, 0.0, pre.value); pre["hi"] = np.where(pre.censored, pre.dl, pre.value)
    rows = []
    for (plant, date), g in top.groupby(["WWTP_CODE", "date"]):
        p = pre[(pre.WWTP_CODE == plant) & (pre.date == date)]
        if p.AXYS_ID.nunique() != 1 or g.AXYS_ID.nunique() != 1:
            continue
        P = p.set_index("COMPOUND"); T = g.set_index("cmp")
        r = dict(plant=plant, date=date.date(), axys_pre=p.AXYS_ID.iloc[0], axys_top=g.AXYS_ID.iloc[0])
        for b in ["lo", "hi"]:
            r[f"pfca_pre_{b}"] = molar(P, PFCA, b); r[f"pfca_post_{b}"] = molar(T, PFCA, b)
            r[f"pfsa_pre_{b}"] = molar(P, PFSA, b); r[f"pfsa_post_{b}"] = molar(T, PFSA, b)
            r[f"prec_pre_{b}"] = molar(P, PREC, b); r[f"prec_post_{b}"] = molar(T, PREC, b)
        rows.append(r)
    S = pd.DataFrame(rows)
    # conservative pairing of bounds for the TOP-revealed pool and its explained share
    S["pool_low"] = S.pfca_post_lo - S.pfca_pre_hi      # smallest possible gain
    S["pool_high"] = S.pfca_post_hi - S.pfca_pre_lo     # largest possible gain
    S["pool_mid"] = (S.pfca_post_lo + S.pfca_post_hi) / 2 - (S.pfca_pre_lo + S.pfca_pre_hi) / 2
    S["prec_mid"] = (S.prec_pre_lo + S.prec_pre_hi) / 2
    S.to_csv(R / "s15_top_samples.csv", index=False)
    print("matched biosolids samples", len(S), "plants", S.plant.nunique())
    pm = S.groupby("plant")[["pool_low", "pool_mid", "pool_high", "prec_pre_lo", "prec_mid", "prec_pre_hi",
                             "pfsa_pre_lo", "pfsa_post_lo", "pfca_pre_lo", "pfca_post_lo"]].median()
    summ = dict(
        samples=len(S), plants=S.plant.nunique(),
        pool_median_nmol_g=pm.pool_mid.median(), pool_low_median=pm.pool_low.median(), pool_high_median=pm.pool_high.median(),
        plants_pool_positive_lowbound=int((pm.pool_low > 0).sum()),
        pfca_post_over_pre_median=float(np.median(pm.pfca_post_lo / pm.pfca_pre_lo.replace(0, np.nan))),
        pfsa_post_over_pre_median=float(np.median(pm.pfsa_post_lo / pm.pfsa_pre_lo.replace(0, np.nan))),
        # share of the TOP pool explained by targeted precursors: most generous and least generous pairing
        explained_upper=float(pm.prec_pre_hi.sum() / pm.pool_low.clip(lower=1e-9).sum()),
        explained_mid=float(pm.prec_mid.sum() / pm.pool_mid.sum()),
        explained_lower=float(pm.prec_pre_lo.sum() / pm.pool_high.sum()),
        targeted_prec_post_over_pre=float(S.prec_post_lo.sum() / max(S.prec_pre_lo.sum(), 1e-9)),
    )
    pd.DataFrame([summ]).to_csv(R / "s15_top_summary.csv", index=False)
    for k, v in summ.items():
        print(f"{k:34s} {v:.4g}" if isinstance(v, float) else f"{k:34s} {v}")
    # (2) plant-level link with 2024 liquid-phase unexplained formation (midpoint substitution, molar)
    pr = pd.read_parquet(D / "pairs_inf_eff.parquet"); pr = pr[pr.date.dt.year == 2024]
    mid = lambda lo, hi: np.where(lo == 0, hi / 2, lo)
    rows = []
    for plant, g in pr.groupby("WWTP_CODE"):
        def tot(side, cs):
            return np.sum([mid(g[f"{side}_lo|{c}"], g[f"{side}_hi|{c}"]) / MW[c] for c in cs
                           if f"{side}_hi|{c}" in g and g[f"{side}_hi|{c}"].notna().all()], axis=0)
        dpfca = tot("EFF", PFCA[:9]) - tot("INF", PFCA[:9])
        dprec = tot("EFF", PREC) - tot("INF", PREC)
        rows.append(dict(plant=plant, d_pfca_nM=np.median(dpfca), unexplained_nM=np.median(dpfca + np.minimum(dprec, 0)),
                         inf_pfca_nM=np.median(tot("INF", PFCA[:9]))))
    L = pd.DataFrame(rows).merge(pm.reset_index()[["plant", "pool_mid", "pool_low", "pool_high"]], on="plant")
    L.to_csv(R / "s15_top_plant_link.csv", index=False)
    if len(L) >= 5:
        for col in ["unexplained_nM", "d_pfca_nM"]:
            rho, p = stats.spearmanr(L.pool_mid, L[col])
            print(f"plant link: pool_mid vs {col}: Spearman rho = {rho:.2f}, P = {p:.3f}, n = {len(L)}")
        rho, p = stats.spearmanr(L.pool_mid, L.inf_pfca_nM)
        print(f"plant link: pool_mid vs influent PFCA: rho = {rho:.2f}, P = {p:.3f}")


if __name__ == "__main__":
    main()
