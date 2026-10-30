"""Sample-level wide tables and same-plant-same-day INF/EFF/BIOS pairs."""
import pandas as pd, numpy as np
from pathlib import Path
D=Path(__file__).resolve().parents[1]/"data_clean"
e=pd.read_parquet(D/"env_long.parquet")
e=e[~e.is_dup & e.usable].copy()
loc_map={"Influent":"INF","Effluent":"EFF","Primary Effluent":"PE","Biosolids":"BIOS",
         "Primary Sludge":"PS","Waste Activated Sludge":"WAS","Sludge":"SLG"}
e=e[e.LOCATION.isin(loc_map)].copy(); e["loc"]=e.LOCATION.map(loc_map)
# lower/upper bounds of concentration (left-censored: [0, DL])
e["lo"]=np.where(e.censored,0.0,e.value); e["hi"]=np.where(e.censored,e.dl,e.value)
key=["WWTP_CODE","date","loc","AXYS_ID"]
lo=e.pivot_table(index=key,columns="COMPOUND",values="lo",aggfunc="first")
hi=e.pivot_table(index=key,columns="COMPOUND",values="hi",aggfunc="first")
lo.columns=[f"lo|{c}" for c in lo.columns]; hi.columns=[f"hi|{c}" for c in hi.columns]
w=pd.concat([lo,hi],axis=1).reset_index()
print(w.groupby("loc").size())
# multiple samples same plant/day/loc?
c=w.groupby(["WWTP_CODE","date","loc"]).size(); print("multi per plant-day-loc", (c>1).sum())
w.to_parquet(D/"wide_samples.parquet")
# pairs: plant-day with exactly one INF and one EFF
one=w.groupby(["WWTP_CODE","date","loc"]).filter(lambda g: len(g)==1)
inf=one[one["loc"]=="INF"].drop(columns=["loc","AXYS_ID"]).set_index(["WWTP_CODE","date"])
eff=one[one["loc"]=="EFF"].drop(columns=["loc","AXYS_ID"]).set_index(["WWTP_CODE","date"])
bios=one[one["loc"]=="BIOS"].drop(columns=["loc","AXYS_ID"]).set_index(["WWTP_CODE","date"])
pr=inf.add_prefix("INF_").join(eff.add_prefix("EFF_"),how="inner")
print("INF-EFF pairs",len(pr), pr.index.get_level_values(0).nunique())
pb=pr.join(bios.add_prefix("BIOS_"),how="inner"); print("INF-EFF-BIOS triples",len(pb))
tf=pd.read_csv(D/"temp_flow.csv",parse_dates=["date"]).dropna(subset=["date"])
tf=tf.groupby(["code","date"]).agg(flow=("flow","mean"),T_inf=("T_inf","mean"),T_eff=("T_eff","mean")).reset_index()
pr=pr.reset_index().merge(tf.rename(columns={"code":"WWTP_CODE"}),on=["WWTP_CODE","date"],how="left")
print("pairs with T_inf",pr.T_inf.notna().sum(), "flow", pr.flow.notna().sum())
pr.to_parquet(D/"pairs_inf_eff.parquet")
# within-plant temp range
r=pr.dropna(subset=["T_inf"]).groupby("WWTP_CODE").agg(n=("T_inf","size"),Tmin=("T_inf","min"),Tmax=("T_inf","max"),yrs=("date",lambda s:s.dt.year.nunique()))
r["range"]=r.Tmax-r.Tmin; print(r.sort_values("range",ascending=False).to_string())
print("pairs within plants with T range>=5:", r[r.range>=5].n.sum(), (r.range>=5).sum())
