"""Within-plant stage comparisons propagated from concentration bounds."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];F=R;O=R/'validation/results'
sys.path.insert(0,str(F/'scripts'))
import s03_upgrade_effect as s3

def main():
    w=pd.read_parquet(F/'data_clean/wide_samples.parquet')
    one=w.groupby(['WWTP_CODE','date','loc']).filter(lambda a:len(a)==1)
    sets={side:one[one['loc']==side].drop(columns=['loc','AXYS_ID']).set_index(['WWTP_CODE','date']) for side in ['INF','PE','EFF']}
    tri=sets['INF'].add_prefix('INF_').join(sets['PE'].add_prefix('PE_'),how='inner').join(sets['EFF'].add_prefix('EFF_'),how='inner').reset_index()
    fields=[f'{s}_hi|{c}' for s in ['INF','PE','EFF'] for c in s3.CMP]
    tri=tri[tri[fields].notna().all(axis=1)].merge(s3.load_design()[['WWTP_CODE','date','proc']],on=['WWTP_CODE','date'],validate='one_to_one').reset_index(drop=True)
    assert len(tri)==90 and tri.WWTP_CODE.nunique()==11
    plant_rows=[];summary=[];pair_rows=[]
    for endpoint,compounds in [('sumPFCA',s3.PFCA),('sumPFSA',s3.PFSA),('PFHxA',['PFHxA']),('PFOA',['PFOA'])]:
        t={}
        for s in ['INF','PE','EFF']:
            for b in ['lo','hi']:t[s,b]=sum(tri[f'{s}_{b}|{c}'].to_numpy(float)/s3.MW[c] for c in compounds)
        with np.errstate(divide='ignore',invalid='ignore'):
            # One shared PE concentration contributes with coefficient -2.
            lower=np.log10(t['EFF','lo'])+np.log10(t['INF','lo'])-2*np.log10(t['PE','hi'])
            upper=np.log10(t['EFF','hi'])+np.log10(t['INF','hi'])-2*np.log10(t['PE','lo'])
        assert not np.isnan(lower).any() and not np.isnan(upper).any()
        q=tri[['WWTP_CODE','date','proc']].copy();q['outcome']=endpoint;q['lower']=lower;q['upper']=upper;pair_rows.append(q)
        pm=q.groupby('WWTP_CODE')[['lower','upper']].agg(lambda s:np.mean(s.to_numpy(float)))
        for plant,a in pm.iterrows():plant_rows.append(dict(plant=plant,outcome=endpoint,lower=a.lower,upper=a.upper,guaranteed_biological_larger=bool(a.lower>0)))
        summary.append(dict(outcome=endpoint,pairs=len(q),plants=len(pm),mean_lower=float(pm.lower.mean()),mean_upper=float(pm.upper.mean()),guaranteed_biological_larger=int((pm.lower>0).sum()),guaranteed_primary_larger=int((pm.upper<0).sum()),unresolved=int(((pm.lower<=0)&(pm.upper>=0)).sum())))
    pd.concat(pair_rows).to_csv(O/'stage_measurement_pair_bounds.csv',index=False)
    pd.DataFrame(plant_rows).to_csv(O/'stage_measurement_plant_bounds.csv',index=False)
    pd.DataFrame(summary).to_csv(O/'stage_measurement_bounds.csv',index=False)
    (O/'stage_measurement_bounds.json').write_text(json.dumps({'definition':'Biological-step minus primary-step log10 molar change, EFF + INF - 2PE, consistent shared PE bound; measurement intervals, not confidence intervals.','summary':summary},indent=2),'utf-8')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
