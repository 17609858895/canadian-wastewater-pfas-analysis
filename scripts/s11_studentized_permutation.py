"""Studentized plant-level permutation test (statistic = theta/cluster-SE) and
stage-test plant-level sign/t tests. Output results/s11_permutation.csv"""
import sys; from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).parent)); import censdml as cd, s03_upgrade_effect as s3
R=Path(__file__).resolve().parents[1]/"results"; P=len(s3.CMP)
G=None
def init():
    global G
    pr=s3.load_design(); V=np.load(R/"s03_imputations.npz")["imps"][0]
    g,d,Z=pr.WWTP_CODE.values,pr.bio.values,pr[s3.ZCOLS].values.astype(float)
    W=np.column_stack([Z,V[:,:P]/np.log(10)]); ys=s3.outcomes(V[:,:P],V[:,P:])
    mh={k:cd.crossfit_nuisance(W,ys[k],d,g)[0] for k in ["sumPFCA","PFHxA","PFOA","sumPFSA"]}
    G=dict(g=g,d=d,W=W,ys=ys,mh=mh,plants=pd.Series(d,index=g).groupby(level=0).first())
def stat(seed):
    rng=np.random.default_rng(seed); g=G["g"]
    if seed==0: dp=G["d"]
    else:
        pl=G["plants"]; perm=pd.Series(rng.permutation(pl.values),index=pl.index); dp=perm.loc[g].values.astype(float)
    _,eh=cd.crossfit_nuisance(G["W"],G["ys"]["sumPFCA"],dp,g,d_only=True,m_hat=G["mh"]["sumPFCA"])
    out={}
    for k,m in G["mh"].items():
        th,var=cd.plr(G["ys"][k],dp,m,eh,g); out[k]=(th,th/np.sqrt(var))
    return out
if __name__=="__main__":
    with Pool(12,initializer=init) as p: res=p.map(stat,range(1000))
    rows=[]
    for k in res[0]:
        th0,t0=res[0][k]; ths=np.array([r[k][0] for r in res[1:]]); ts=np.array([r[k][1] for r in res[1:]])
        rows.append(dict(outcome=k,theta=th0,t=t0,p_perm_theta=(1+np.sum(abs(ths)>=abs(th0)))/(1+len(ths)),p_perm_studentized=(1+np.sum(abs(ts)>=abs(t0)))/(1+len(ts)),
                         p_perm_onesided_t=(1+np.sum(ts>=t0))/(1+len(ts))))
    np.savez_compressed(R/"s11_null.npz",**{f"{k}_theta":np.array([r[k][0] for r in res[1:]]) for k in res[0]},
                        **{f"{k}_t":np.array([r[k][1] for r in res[1:]]) for k in res[0]})
    out=pd.DataFrame(rows); print(out.round(4).to_string()); out.to_csv(R/"s11_permutation.csv",index=False)
