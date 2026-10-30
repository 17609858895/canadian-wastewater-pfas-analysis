"""Molar budget 2018+: measured precursor loss vs PFAA gain across treatment
(bounds: ND=0 lower, ND=DL upper; and DL/2). Output results/s06_precursor_budget.csv"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).parent)); import s03_upgrade_effect as s3
ROOT=Path(__file__).resolve().parents[1]; D,R=ROOT/"data_clean",ROOT/"results"
MW=dict(PFBA=214.04,PFPeA=264.05,PFHxA=314.05,PFHpA=364.06,PFOA=414.07,PFNA=464.08,PFDA=514.08,PFUnA=564.09,PFDoA=614.1,
 PFBS=300.1,PFPeS=350.1,PFHxS=400.11,PFHpS=450.12,PFOS=500.13,PFNS=550.14,PFDS=600.14,
 **{"4:2 FTS":328.1,"6:2 FTS":428.17,"8:2 FTS":528.18,"EtFOSAA":585.24,"MeFOSAA":571.21,"N-EtFOSE":571.25,"N-MeFOSE":557.23,
    "N-EtFOSA":527.2,"N-MeFOSA":513.17,"PFOSA":499.14,"5:3 FTCA":342.1,"3:3 FTCA":242.1,"7:3 FTCA":442.1})
PFAA=["PFBA","PFPeA","PFHxA","PFHpA","PFOA","PFNA","PFDA","PFUnA","PFDoA","PFBS","PFPeS","PFHxS","PFHpS","PFOS","PFNS","PFDS"]
PREC=["4:2 FTS","6:2 FTS","8:2 FTS","EtFOSAA","MeFOSAA","N-EtFOSE","N-MeFOSE","N-EtFOSA","N-MeFOSA","PFOSA"]
pr=pd.read_parquet(D/"pairs_inf_eff.parquet"); des=s3.load_design()[["WWTP_CODE","date","proc","bio"]]
pr=pr.merge(des,on=["WWTP_CODE","date"]); pr=pr[pr.date.dt.year>=2018]
pr=pr[pr[[f"INF_hi|{c}" for c in PREC+PFAA]].notna().all(axis=1)]
print("2018+ pairs",len(pr),pr.WWTP_CODE.nunique())
def tot(side,cs,mode):
    s=0
    for c in cs:
        lo,hi=pr[f"{side}_lo|{c}"],pr[f"{side}_hi|{c}"]
        v={"lo":lo,"hi":hi,"mid":np.where(lo==0,hi/2,lo)}[mode]
        s=s+np.asarray(v,float)/MW[c]
    return s
rows=[]
for mode in ["lo","mid","hi"]:
    # conservative: PFAA gain lower bound uses EFF lo - INF hi etc.
    dA=tot("EFF",PFAA,mode)-tot("INF",PFAA,mode); dP=tot("EFF",PREC,mode)-tot("INF",PREC,mode)
    pr["dA"],pr["dP"]=dA,dP
    for proc,gp in pr.groupby(pr.bio.map({0:"primary",1:"biological"})):
        pm=gp.groupby("WWTP_CODE")[["dA","dP"]].median()
        rows.append(dict(mode=mode,group=proc,plants=len(pm),median_dPFAA_nM=pm.dA.median(),median_dPrec_nM=pm.dP.median(),
                         frac_explained=(-pm.dP.clip(upper=0)).sum()/max(pm.dA.clip(lower=0).sum(),1e-9)))
out=pd.DataFrame(rows); print(out.round(4).to_string()); out.to_csv(R/"s06_precursor_budget.csv",index=False)
# per-compound median EFF-INF for 2018+ bio
b=pr[pr.bio==1]
print(pd.Series({c:np.median(np.where(b[f"EFF_lo|{c}"]==0,b[f"EFF_hi|{c}"]/2,b[f"EFF_lo|{c}"])/MW[c]-np.where(b[f"INF_lo|{c}"]==0,b[f"INF_hi|{c}"]/2,b[f"INF_lo|{c}"])/MW[c]) for c in PREC+PFAA}).round(4).to_string())
