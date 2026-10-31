"""Additional influent-composition diagnostics under the already frozen weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import overlap_analysis as a

def main():
    pr=a.load_pre_design();out=Path(__file__).resolve().parent/'results/overlap'
    pw=pd.read_csv(out/'overlap_pair_weights.csv');assert len(pw)==len(pr)
    lo,hi=a.s3.bounds(pr,'INF');v=np.where(np.isfinite(lo),lo,hi-np.log(2))/np.log(10)
    cols=a.DESIGN_COLS.copy()
    for j,c in enumerate(a.s3.CMP):
        if c=='PFOSA':continue # Already present as INF_PFOSA; do not count twice.
        name='INF_log_'+c;pr[name]=v[:,j];cols.append(name)
    pp=pr.groupby('WWTP_CODE')[cols+['bio']].mean()
    result=a.balance(pr,pw,pp,cols);result['core_design_covariate']=result.covariate.isin(a.DESIGN_COLS)
    result.to_csv(out/'overlap_balance_full_profile.csv',index=False)
    # Independent direct reconstruction of weighted family outcomes from concentration arrays.
    with np.load(a.FROZEN/'results/s03_imputations.npz') as z:V=z['imps']
    old=pd.read_csv(out/'overlap_contrasts.csv');checks={}
    for family,compounds in [('sumPFCA',a.s3.PFCA),('sumPFSA',a.s3.PFSA)]:
        ids=[a.s3.CMP.index(c) for c in compounds];mw=np.array([a.s3.MW[c] for c in compounds])
        y=np.log10((np.exp(V[:,:,np.array(ids)+11])/mw).sum(2)/(np.exp(V[:,:,ids])/mw).sum(2))
        for method in ('equal_plant','overlap_C1','region_era_standardized'):
            d=pr.bio.to_numpy();w=pw['weight_'+method].to_numpy()
            q=np.mean(np.sum(y[:,d==1]*w[d==1],axis=1)/w[d==1].sum()-np.sum(y[:,d==0]*w[d==0],axis=1)/w[d==0].sum())
            expected=old[(old.outcome==family)&(old.method==method)].theta.iloc[0]
            np.testing.assert_allclose(q,expected,rtol=0,atol=1e-12)
            checks[family+'_'+method]=True
    summary=result.groupby('method').SMD.agg(max_abs=lambda x:x.abs().max(),above_02=lambda x:int((x.abs()>.2).sum())).reset_index()
    summary.to_csv(out/'overlap_full_profile_summary.csv',index=False)
    original=result[~result.covariate.isin(['INF_PFCA','INF_PFSA'])]
    original.to_csv(out/'overlap_balance_original25.csv',index=False)
    original.groupby('method').SMD.agg(max_abs=lambda x:x.abs().max(),above_02=lambda x:int((x.abs()>.2).sum())).reset_index().to_csv(out/'overlap_original25_summary.csv',index=False)
    (out/'overlap_independent_reconstruction.json').write_text(json.dumps(checks,indent=2),'utf-8')
    print(summary.to_string(index=False));print(json.dumps(checks))

if __name__=='__main__':main()
