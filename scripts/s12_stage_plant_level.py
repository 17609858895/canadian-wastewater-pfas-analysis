"""Plant-level summary of the within-plant stage test (sumPFCA, PFHxA, PFOA):
per-plant mean (biological step - primary step), sign test, one-sample t.
Output results/s12_stage_plants.csv"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).parent)); import censdml as cd, s03_upgrade_effect as s3
import s05_process_stage_biosolids as s5
R=Path(__file__).resolve().parents[1]/"results"; D=R.parent/"data_clean"; P=len(s3.CMP)
w=pd.read_parquet(D/"wide_samples.parquet"); one=w.groupby(["WWTP_CODE","date","loc"]).filter(lambda x:len(x)==1)
sets={l:one[one["loc"]==l].drop(columns=["loc","AXYS_ID"]).set_index(["WWTP_CODE","date"]) for l in ["INF","PE","EFF"]}
tri=sets["INF"].add_prefix("INF_").join(sets["PE"].add_prefix("PE_"),how="inner").join(sets["EFF"].add_prefix("EFF_"),how="inner").reset_index()
tri=tri[tri[[f"{s}_hi|{c}" for s in ["INF","PE","EFF"] for c in s3.CMP]].notna().all(axis=1)].reset_index(drop=True)
tri=tri.merge(s3.load_design()[["WWTP_CODE","date"]+s3.ZCOLS+["proc"]],on=["WWTP_CODE","date"]).reset_index(drop=True)
lo,hi=[],[]
for s in ["INF","PE","EFF"]:
    l=np.column_stack([tri[f"{s}_lo|{c}"].values for c in s3.CMP]).astype(float); h=np.column_stack([tri[f"{s}_hi|{c}"].values for c in s3.CMP]).astype(float)
    with np.errstate(divide="ignore"): lo.append(np.where(l>0,np.log(l),-np.inf))
    hi.append(np.log(h))
Vlo,Vhi=np.hstack(lo),np.hstack(hi); g=tri.WWTP_CODE.values; Z=tri[s3.ZCOLS].values.astype(float)
acc={}
for m in range(5):
    V=cd.chained_impute(Vlo,Vhi,Z,g,n_sweeps=3,seed=500+m); a=s3.outcomes(V[:,:P],V[:,P:2*P]); b=s3.outcomes(V[:,P:2*P],V[:,2*P:])
    for k in ["sumPFCA","PFHxA","PFOA"]:
        acc.setdefault(k,[]).append(pd.DataFrame(dict(g=g,prim=a[k],bio=b[k])).groupby("g").mean())
rows=[]
for k,lst in acc.items():
    pm=sum(lst)/len(lst); pm["diff"]=pm.bio-pm.prim; pm["proc"]=pm.index.map(tri.groupby("WWTP_CODE").proc.first())
    t=stats.ttest_1samp(pm["diff"],0); sg=stats.binomtest(int((pm["diff"]>0).sum()),len(pm),0.5)
    print(k, pm.round(3).to_string(), "\n t",round(t.statistic,2),"p",round(t.pvalue,4),"positive",int((pm["diff"]>0).sum()),"/",len(pm),"sign p",round(sg.pvalue,4))
    pm.assign(outcome=k,t_p=t.pvalue,sign_p=sg.pvalue).to_csv(R/f"s12_stage_plants_{k}.csv")
