"""Receiving-water summaries for the map panels (sampled plants only).
Per plant: median over sampling days of latent effluent sumPFCA (ng/L, mean of
10 imputations) x flow (sample-day flow, else the plant's average flow) -> g/yr.
Upgrade increment for primary-only plants from s07 (pooled scenario).
Output: results/s14_region_summary.csv, results/s14_plant_loads.csv"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); import s03_upgrade_effect as s3
ROOT = Path(__file__).resolve().parents[1]; D, R = ROOT / "data_clean", ROOT / "results"
P = len(s3.CMP)
pr = s3.load_design()
imps = np.load(R / "s03_imputations.npz")["imps"]
ks = [s3.CMP.index(c) for c in s3.PFCA]
pr["eff_ngL"] = np.mean([np.exp(V[:, P:][:, ks]).sum(1) for V in imps], axis=0)
pr["inf_ngL"] = np.mean([np.exp(V[:, :P][:, ks]).sum(1) for V in imps], axis=0)
pl = pd.read_csv(D / "plants.csv"); pl["code2"] = pl.code.str.replace(r" \(.*\)", "", regex=True)
pl = pl.drop_duplicates("code2").set_index("code2")
pr["rec"] = pr.WWTP_CODE.map(pl.receiving.str.strip())
avgf = pd.to_numeric(pl.flow_avg.str.replace(",", ""), errors="coerce")
pr["flow_use"] = pr.flow.fillna(pr.WWTP_CODE.map(avgf))
pr["load_g_d"] = pr.eff_ngL * pr.flow_use * 1e-6        # ng/L * m3/d = 1e3 ng/d... -> g/d
pr["formed_g_d"] = (pr.eff_ngL - pr.inf_ngL) * pr.flow_use * 1e-6
pp = pr.groupby("WWTP_CODE").agg(rec=("rec", "first"), proc=("proc", "first"), pairs=("bio", "size"),
                                 load_g_yr=("load_g_d", lambda v: np.nanmedian(v) * 365),
                                 formed_g_yr=("formed_g_d", lambda v: np.nanmedian(v) * 365),
                                 ratio=("eff_ngL", "median")).reset_index()
pp.to_csv(R / "s14_plant_loads.csv", index=False)
J = pd.read_csv(R / "s07_projection.csv"); J = J[(J.level == "plant") & (J.scenario == "biological (pooled)")]
pp = pp.merge(J[["plant", "added_g_yr", "added_lo", "added_hi"]], left_on="WWTP_CODE", right_on="plant", how="left")
rows = []
for rec, g in pp.groupby("rec"):
    r = dict(rec=rec, plants=len(g), pairs=int(g.pairs.sum()), plants_with_load=int(g.load_g_yr.notna().sum()),
             load_kg_yr=g.load_g_yr.sum() / 1e3, formed_kg_yr=g.formed_g_yr.sum() / 1e3,
             added_kg_yr=g.added_g_yr.sum() / 1e3, added_lo=g.added_lo.sum() / 1e3, added_hi=g.added_hi.sum() / 1e3,
             primary_plants=int((g.proc == "primary").sum()))
    r.update({f"n_{p}": int((g.proc == p).sum()) for p in ["primary", "biofilm", "AS", "BNR", "lagoon", "ext_aer"]})
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(R / "s14_region_summary.csv", index=False)
print(out.round(2).to_string())
