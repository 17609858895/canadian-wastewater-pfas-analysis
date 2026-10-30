"""Counterfactual upgrade projection for the sampled primary-only plants and a
national-scale bound.

Inputs: s03 imputations (latent EFF), s03_effects.csv (LOD-DML theta),
s05_process.csv (process-specific theta), s05_biosolids.csv.
National scaling uses the share of Canadian wastewater volume receiving
primary treatment only (f_p, scenario range 0.20-0.30; ~0.24 reported in
federal summaries -- to be cited from the primary source before use).
Output: results/s07_projection.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import s03_upgrade_effect as s3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D, R = ROOT / "data_clean", ROOT / "results"
P = len(s3.CMP)
rng = np.random.default_rng(42)


def main():
    pr = s3.load_design()
    imps = np.load(R / "s03_imputations.npz")["imps"]
    eff = pd.read_csv(R / "s03_effects.csv")
    proc = pd.read_csv(R / "s05_process.csv") if (R / "s05_process.csv").exists() else None
    th = eff[(eff.method == "LOD-DML") & (eff.outcome == "sumPFCA")].iloc[0]
    scen = {"biological (pooled)": (th.theta, th.se)}
    if proc is not None:
        for _, r in proc[proc.outcome == "sumPFCA"].iterrows():
            scen[r.proc] = (r.theta, r.se)
    # latent effluent sum PFCA (ng/L) at primary plants, mean over imputations
    ks = [s3.CMP.index(c) for c in s3.PFCA]
    se_ng = np.mean([np.exp(V[:, P:][:, ks]).sum(1) for V in imps], axis=0)
    pr["effPFCA_ngL"] = se_ng
    prim = pr[pr.bio == 0].copy()
    pl = pd.read_csv(D / "plants.csv")
    pl["code2"] = pl.code.str.replace(r" \(.*\)", "", regex=True)
    fl = pd.to_numeric(pl.drop_duplicates("code2").set_index("code2").flow_avg.str.replace(",", ""), errors="coerce")
    prim["flow_use"] = prim.flow.fillna(prim.WWTP_CODE.map(fl))
    rows = []
    for plant, gp in prim.groupby("WWTP_CODE"):
        if gp.flow_use.isna().all():
            continue
        load = np.nanmedian(gp.effPFCA_ngL * gp.flow_use) * 1e3 * 365 / 1e9  # ng/L*m3/d -> g/yr
        for name, (t, s) in scen.items():
            draws = 10 ** rng.normal(t, s, 4000)
            rows.append(dict(level="plant", plant=plant, scenario=name, current_g_yr=load,
                             added_g_yr=np.median(load * (draws - 1)),
                             added_lo=np.percentile(load * (draws - 1), 2.5),
                             added_hi=np.percentile(load * (draws - 1), 97.5),
                             fold=np.median(draws)))
    for fp in [0.20, 0.24, 0.30]:
        for name, (t, s) in scen.items():
            draws = 10 ** rng.normal(t, s, 4000)
            tb = 10 ** th.theta  # current secondary plants already carry the pooled effect
            inc = fp * (draws - 1) / (fp + (1 - fp) * tb)
            rows.append(dict(level="national", plant=f"f_p={fp}", scenario=name, pct_increase=100 * np.median(inc),
                             pct_lo=100 * np.percentile(inc, 2.5), pct_hi=100 * np.percentile(inc, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(R / "s07_projection.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
