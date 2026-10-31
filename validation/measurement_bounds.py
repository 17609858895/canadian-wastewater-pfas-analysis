"""Propagate reported concentration intervals without fitting an imputation model.

Bounds concern measurement-compatible descriptive endpoints, not causal effects
or confidence intervals. Quantified concentrations are fixed; non-detects range
independently from zero to their own detection limit. No unknown compounds are
added to the targeted panels. Frozen source data and results are read only.
"""
from pathlib import Path
import importlib.util,sys,json,argparse
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];SRC=R;OUT=R/'validation/results'
sys.path.insert(0,str(SRC/'scripts'))
import s03_upgrade_effect as s3
spec=importlib.util.spec_from_file_location('matched_panel_source',R/'scripts/s16_matched_panel_precursor.py')
mp=importlib.util.module_from_spec(spec);spec.loader.exec_module(mp)

def sums(pr,side,compounds,bound):
    return sum(pr[f'{side}_{bound}|{c}'].to_numpy(float)/mp.MW[c] for c in compounds)

def ratio_bounds(il,ih,el,eh):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.log10(el/ih),np.log10(eh/il)

def contrast_interval(pm,weightcol):
    w=pm[weightcol].to_numpy(float);d=pm.bio.to_numpy(bool)
    def mean(v,mask):
        use=mask&(w>0)
        return np.average(pm.loc[use,v].to_numpy(float),weights=w[use]) if use.any() else np.nan
    return mean('lower',d)-mean('upper',~d),mean('upper',d)-mean('lower',~d)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pr=s3.load_design();pairrows=[];plantrows=[];summary=[]
    for family,cs in [('sumPFCA',s3.PFCA),('sumPFSA',s3.PFSA)]:
        il,ih=sums(pr,'INF',cs,'lo'),sums(pr,'INF',cs,'hi')
        el,eh=sums(pr,'EFF',cs,'lo'),sums(pr,'EFF',cs,'hi')
        low,high=ratio_bounds(il,ih,el,eh)
        q=pr[['WWTP_CODE','date','bio']].copy();q['family']=family;q['lower']=low;q['upper']=high
        q['guaranteed_increase']=low>0;q['guaranteed_decrease']=high<0
        pairrows.append(q)
        pm=q.groupby(['WWTP_CODE','bio'],as_index=False).agg(lower=('lower','mean'),upper=('upper','mean'),pairs=('lower','size'))
        pm['family']=family;pm['equal_plant']=1.
        if (OUT/'overlap/overlap_plant_weights.csv').exists():
            weights=pd.read_csv(OUT/'overlap/overlap_plant_weights.csv')
            # Exact schema is controlled by the design-analysis module.
            if 'plant' in weights:weights=weights.rename(columns={'plant':'WWTP_CODE'})
            for col in ('weight_overlap','weight_equal'):
                if col in weights:pm=pm.merge(weights[['WWTP_CODE',col]],on='WWTP_CODE',validate='one_to_one')
        for wc in ['equal_plant']+[c for c in ['weight_overlap'] if c in pm]:
            a,b=contrast_interval(pm,wc)
            summary.append(dict(endpoint=family,population=wc,lower=a,upper=b,plants=len(pm),primary_plants=int((pm.bio==0).sum()),biological_plants=int((pm.bio==1).sum())))
        for d in (0,1):
            z=q[q.bio==d];summary.append(dict(endpoint=family,population='primary_pairs' if d==0 else 'biological_pairs',pairs=len(z),guaranteed_increase=int(z.guaranteed_increase.sum()),guaranteed_decrease=int(z.guaranteed_decrease.sum()),unresolved=int((~z.guaranteed_increase&~z.guaranteed_decrease).sum())))
        plantrows.append(pm)
    pd.concat(pairrows).to_csv(OUT/'family_measurement_pair_bounds.csv',index=False)
    pd.concat(plantrows).to_csv(OUT/'family_measurement_plant_bounds.csv',index=False)
    # Unlike applying one fixed substitution to both matrices, independent
    # interval propagation includes opposite concentration endpoints.
    ex=mp.load_population(SRC/'data_clean');detect={c:int(sum((ex[f'{s}_lo|{c}']>0).sum() for s in ('INF','EFF'))) for c in mp.PREC+mp.PFAA}
    idxrows=[];idxplants=[]
    for panel in [mp.PREC+mp.PFAA,[c for c in mp.PREC+mp.PFAA if detect[c]>=5]]:
        acids=[c for c in mp.PFAA if c in panel];prec=[c for c in mp.PREC if c in panel]
        q=ex[['WWTP_CODE','biological']].copy()
        for label,cs in [('acid',acids),('prec',prec)]:
            q[label+'_lo']=sums(ex,'EFF',cs,'lo')-sums(ex,'INF',cs,'hi')
            q[label+'_hi']=sums(ex,'EFF',cs,'hi')-sums(ex,'INF',cs,'lo')
        for flag in [True,False]:
            z=q[q.biological==flag];pm=z.groupby('WWTP_CODE').median(numeric_only=True)
            nlo=float(np.maximum(-pm.prec_hi,0).sum());nhi=float(np.maximum(-pm.prec_lo,0).sum())
            dlo=float(np.maximum(pm.acid_lo,0).sum());dhi=float(np.maximum(pm.acid_hi,0).sum())
            low=nlo/dhi if dhi>0 else np.nan;high=nhi/dlo if dlo>0 else np.inf
            idxrows.append(dict(compounds=len(panel),population='biological' if flag else 'primary',pairs=len(z),plants=len(pm),loss_lower=nlo,loss_upper=nhi,gain_lower=dlo,gain_upper=dhi,index_lower_percent=100*low,index_upper_percent=100*high))
            for plant,r in pm.iterrows():idxplants.append(dict(plant=plant,compounds=len(panel),population='biological' if flag else 'primary',**r.to_dict()))
    pd.DataFrame(summary).to_csv(OUT/'measurement_bounds_summary.csv',index=False)
    pd.DataFrame(idxrows).to_csv(OUT/'precursor_measurement_bounds.csv',index=False)
    pd.DataFrame(idxplants).to_csv(OUT/'precursor_measurement_plant_bounds.csv',index=False)
    old=np.load(SRC/'results/s03_imputations.npz');v=old['imps'];lo=old['Vlo'];hi=old['Vhi'];c=~np.isfinite(lo)
    excess=np.where(c[None,:,:],v-hi[None,:,:],-np.inf)
    oldcheck={'completed_draws':int(v.size),'censored_draws':int(c.sum()*len(v)), 'draws_above_reported_upper_by_more_than_1e_10':int((excess>1e-10).sum()),'max_log_excess':float(np.max(excess)), 'detected_values_unchanged':bool(np.allclose(v[:,~c],lo[~c][None,:],rtol=0,atol=1e-10))}
    (OUT/'original_imputation_bound_check.json').write_text(json.dumps(oldcheck,indent=2),'utf-8')
    info={'definition':'Measurement-compatible bounds, not sampling confidence intervals or adjusted causal bounds. Existing quantified values fixed; each non-detect independently varies in [0,DL].','summary':summary,'precursor_bounds':idxrows,'original_imputation_check':oldcheck}
    (OUT/'measurement_bounds.json').write_text(json.dumps(info,indent=2),'utf-8')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(idxrows).to_string(index=False));print(oldcheck)

if __name__=='__main__':main()
