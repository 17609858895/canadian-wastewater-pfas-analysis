"""Era-stratified LOD-DML: pre-2018 (filtered, dissolved phase) vs 2018+ (whole water).
Output results/s05b_era.csv"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).parent)); import censdml as cd, s03_upgrade_effect as s3
R=Path(__file__).resolve().parents[1]/"results"; P=len(s3.CMP)
pr=s3.load_design(); imps=list(np.load(R/"s03_imputations.npz")["imps"])
g,d,Z=pr.WWTP_CODE.values,pr.bio.values,pr[s3.ZCOLS].values.astype(float)
rows=[]
for era,keep in [("filtered_pre2018",(pr.year<2018).values),("wholewater_2018plus",(pr.year>=2018).values)]:
    for name in ["sumPFCA","PFHxA","PFOA","PFDA","PFOS","sumPFSA"]:
        ths,vs=[],[]
        for V in imps:
            y=s3.outcomes(V[:,:P],V[:,P:])[name]; W=np.column_stack([Z,V[:,:P]/np.log(10)])
            mh,eh=cd.crossfit_nuisance(W[keep],y[keep],d[keep],g[keep],n_splits=5)
            th,var=cd.plr(y[keep],d[keep],mh,eh,g[keep]); ths.append(th); vs.append(var)
        th,se=cd.rubin(ths,vs)
        rows.append(dict(era=era,outcome=name,theta=th,se=se,n=keep.sum(),plants=len(np.unique(g[keep])),primary_plants=len(np.unique(g[keep&(d==0)]))))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(R/"s05b_era.csv",index=False)
