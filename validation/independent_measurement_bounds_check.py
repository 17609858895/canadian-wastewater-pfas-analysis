"""Independent concentration-corner and random-fill verification of measurement bounds.

Reads raw cleaned pairing bounds directly. It does not call the bound-producing
script's sums, aggregation or ratio functions and never modifies that script.
"""
from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT;OUT=ROOT/'validation/results'
COMMON=['PFBA','PFPeA','PFHxA','PFHpA','PFOA','PFNA','PFDA','PFBS','PFHxS','PFOS','PFOSA']
PFCA=COMMON[:7];PFSA=COMMON[7:10]
ACIDS=['PFBA','PFPeA','PFHxA','PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA','PFBS','PFPeS','PFHxS','PFHpS','PFOS','PFNS','PFDS']
PRECURSORS=['4:2 FTS','6:2 FTS','8:2 FTS','EtFOSAA','MeFOSAA','N-EtFOSE','N-MeFOSE','N-EtFOSA','N-MeFOSA','PFOSA']
spec=importlib.util.spec_from_file_location('chem_constants',ROOT/'scripts/s16_matched_panel_precursor.py')
chem=importlib.util.module_from_spec(spec);spec.loader.exec_module(chem)
MW=chem.MW # Chemical constants only; no bound/ratio functions are reused.
CHECKS={};DETAILS={};rng=np.random.default_rng(9137)

def close(a,b,name,rtol=1e-11,atol=1e-11):
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,equal_nan=False,err_msg=name)
    CHECKS[name]=True

def matrices(pr,cs):
    result={}
    for side in ('INF','EFF'):
        for bound in ('lo','hi'):
            result[side+'_'+bound]=np.column_stack([pr[f'{side}_{bound}|{c}'].to_numpy(float) for c in cs])
    return result

def log_ratio(inf,eff,mw):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.log10(np.sum(eff/mw,axis=-1)/np.sum(inf/mw,axis=-1))

def plant_array(values,plant,ug):
    # NumPy mean intentionally preserves a one-sided infinity; no NaN-skipping.
    return np.array([np.mean(values[...,plant==g],axis=-1) for g in ug]).T

def contrast(v,plant,d,ug,weights):
    pm=plant_array(v,plant,ug);gd=np.array([d[plant==g][0] for g in ug])
    mask0=(gd==0)&(weights>0);mask1=(gd==1)&(weights>0)
    return np.average(pm[...,mask1],weights=weights[mask1],axis=-1)-np.average(pm[...,mask0],weights=weights[mask0],axis=-1)

def fill(b,side,kind):
    l,h=b[side+'_lo'],b[side+'_hi'];det=l==h
    if kind=='zero':fraction=np.zeros(l.shape)
    elif kind=='DL':fraction=np.ones(l.shape)
    elif kind=='half':fraction=np.full(l.shape,.5)
    elif kind=='uniform':fraction=rng.uniform(1e-12,1.,size=l.shape)
    elif kind=='log_uniform':fraction=np.exp(rng.uniform(np.log(1e-12),0.,size=l.shape))
    else:raise ValueError(kind)
    return np.where(det,l,l+fraction*(h-l))

def check_families(pr):
    plant=pr.WWTP_CODE.to_numpy();d=pr.bio.to_numpy();ug=np.unique(plant)
    summary=pd.read_csv(OUT/'measurement_bounds_summary.csv');source_plants=pd.read_csv(OUT/'family_measurement_plant_bounds.csv')
    source_pairs=pd.read_csv(OUT/'family_measurement_pair_bounds.csv')
    wsource=pd.read_csv(OUT/'overlap/overlap_plant_weights.csv').set_index('plant')
    weights={'equal_plant':np.ones(len(ug)),'weight_overlap':wsource.reindex(ug).weight_overlap.to_numpy()}
    result={}
    for family,cs in [('sumPFCA',PFCA),('sumPFSA',PFSA)]:
        b=matrices(pr,cs);mw=np.array([MW[c] for c in cs])
        low=log_ratio(b['INF_hi'],b['EFF_lo'],mw);high=log_ratio(b['INF_lo'],b['EFF_hi'],mw)
        assert not np.isnan(low).any() and not np.isnan(high).any()
        pair_source=source_pairs[source_pairs.family==family].reset_index(drop=True)
        close(low,pair_source.lower,'pair_lower_'+family);close(high,pair_source.upper,'pair_upper_'+family)
        lower_pm=plant_array(low,plant,ug);upper_pm=plant_array(high,plant,ug)
        ps=source_plants[source_plants.family==family].set_index('WWTP_CODE').reindex(ug)
        close(lower_pm,ps.lower,'plant_numpy_lower_'+family);close(upper_pm,ps.upper,'plant_numpy_upper_'+family)
        gd=np.array([d[plant==g][0] for g in ug]);fr={}
        for name,w in weights.items():
            m0=(gd==0)&(w>0);m1=(gd==1)&(w>0)
            lb=float(np.average(lower_pm[m1],weights=w[m1])-np.average(upper_pm[m0],weights=w[m0]))
            ub=float(np.average(upper_pm[m1],weights=w[m1])-np.average(lower_pm[m0],weights=w[m0]))
            expected=summary[(summary.endpoint==family)&(summary.population==name)].iloc[0]
            close([lb,ub],[expected.lower,expected.upper],'weighted_interval_'+family+'_'+name)
            # Adversarial extrema use opposite endpoints in the two process groups.
            inf_low=np.where(d[:,None]==1,b['INF_hi'],b['INF_lo']);eff_low=np.where(d[:,None]==1,b['EFF_lo'],b['EFF_hi'])
            inf_high=np.where(d[:,None]==1,b['INF_lo'],b['INF_hi']);eff_high=np.where(d[:,None]==1,b['EFF_hi'],b['EFF_lo'])
            close(contrast(log_ratio(inf_low,eff_low,mw),plant,d,ug,w),lb,'attained_corner_lower_'+family+'_'+name)
            close(contrast(log_ratio(inf_high,eff_high,mw),plant,d,ug,w),ub,'attained_corner_upper_'+family+'_'+name)
            # Random compatible fills, including concentrations arbitrarily near zero.
            min_seen,max_seen=np.inf,-np.inf
            for k in range(256):
                kind='uniform' if k%2 else 'log_uniform'
                y=log_ratio(fill(b,'INF',kind),fill(b,'EFF',kind),mw)
                q=float(contrast(y,plant,d,ug,w));assert lb-1e-10<=q<=ub+1e-10
                min_seen=min(min_seen,q);max_seen=max(max_seen,q)
            CHECKS['512_random_fills_total_per_family_across_two_weight_designs_'+family+'_'+name]=True
            fr[name]={'lower':lb,'upper':ub,'random_fills':256,'random_min':min_seen,'random_max':max_seen}
        for group,label in [(0,'primary_pairs'),(1,'biological_pairs')]:
            expected=summary[(summary.endpoint==family)&(summary.population==label)].iloc[0];mask=d==group
            counts=[int(np.sum(low[mask]>0)),int(np.sum(high[mask]<0)),int(np.sum(~((low[mask]>0)|(high[mask]<0))))]
            close(counts,[expected.guaranteed_increase,expected.guaranteed_decrease,expected.unresolved],'guaranteed_pair_counts_'+family+'_'+label)
        result[family]=fr
    DETAILS['family_bounds']=result

def precursor_index(pr,b,cs,flag,inf=None,eff=None):
    mw=np.array([MW[c] for c in cs]);acid=np.array([c in ACIDS for c in cs]);mask=pr.bio.to_numpy()==flag
    if inf is None:raise ValueError('Explicit concentrations required')
    a=(eff[:,acid]/mw[acid]-inf[:,acid]/mw[acid]).sum(1)
    p=(eff[:,~acid]/mw[~acid]-inf[:,~acid]/mw[~acid]).sum(1)
    plant=pr.WWTP_CODE.to_numpy();ug=np.unique(plant[mask])
    ap=np.array([np.median(a[plant==g]) for g in ug]);pp=np.array([np.median(p[plant==g]) for g in ug])
    loss=np.maximum(-pp,0).sum();gain=np.maximum(ap,0).sum()
    return float(loss),float(gain),float(100*loss/gain) if gain>0 else np.nan

def check_precursors(pr):
    panel=PRECURSORS+ACIDS;assert len(panel)==26 and not(set(PRECURSORS)&set(ACIDS))
    ex=pr[pr.date.dt.year.ge(2018)].copy()
    ex=ex[ex[[f'{s}_hi|{c}' for s in ('INF','EFF') for c in panel]].notna().all(1)].reset_index(drop=True)
    assert len(ex)==153 and ex.WWTP_CODE.nunique()==31 and ex.loc[ex.bio.eq(1),'WWTP_CODE'].nunique()==26
    detections={c:int(sum((ex[f'{s}_lo|{c}']>0).sum() for s in ('INF','EFF'))) for c in panel}
    supported=[c for c in panel if detections[c]>=5];assert len(supported)==21
    assert set(panel)-set(supported)=={'4:2 FTS','8:2 FTS','N-MeFOSE','N-MeFOSA','PFUnA'}
    source=pd.read_csv(OUT/'precursor_measurement_bounds.csv');fixed=pd.read_csv(ROOT/'results/s16_matched_panel.csv')
    details=[]
    for cs in (panel,supported):
        b=matrices(ex,cs)
        for flag,pop in ((1.,'biological'),(0.,'primary')):
            src=source[(source.compounds==len(cs))&(source.population==pop)].iloc[0]
            # All EFF upper / INF lower minimizes precursor loss and maximizes acid gain.
            nlo,dhi,low=precursor_index(ex,b,cs,flag,b['INF_lo'],b['EFF_hi'])
            nhi,dlo,high=precursor_index(ex,b,cs,flag,b['INF_hi'],b['EFF_lo'])
            high=high if dlo>0 else np.inf
            close([nlo,nhi,dlo,dhi,low,high],[src.loss_lower,src.loss_upper,src.gain_lower,src.gain_upper,src.index_lower_percent,src.index_upper_percent],f'precursor_corners_{len(cs)}_{pop}')
            if dlo==0:
                assert nhi>0 and dhi>0
                CHECKS[f'upper_supremum_unbounded_independent_acid_precursor_components_{len(cs)}_{pop}']=True
            for kind,label in [('zero','0'),('half','DL/2'),('DL','DL')]:
                n,den,value=precursor_index(ex,b,cs,flag,fill(b,'INF',kind),fill(b,'EFF',kind))
                expect=fixed[(fixed.Compounds==len(cs))&(fixed.Population==('Biological plants' if flag else 'Primary-only plants'))&(fixed.Substitution==label)].iloc[0]
                # Table S14 was deliberately exported with ten significant digits.
                close(value,expect.Index_percent,f'fixed_substitution_{len(cs)}_{pop}_{label}',rtol=1e-9,atol=1e-9)
                assert low-1e-9<=value<=high+1e-9
            finite=0;min_seen,max_seen=np.inf,-np.inf
            for k in range(256):
                kind='uniform' if k%2 else 'log_uniform'
                n,den,value=precursor_index(ex,b,cs,flag,fill(b,'INF',kind),fill(b,'EFF',kind))
                if den<=0:continue
                assert low-1e-9<=value<=high+1e-9,(len(cs),pop,value,low,high)
                finite+=1;min_seen=min(min_seen,value);max_seen=max(max_seen,value)
            CHECKS[f'random_precursor_indices_{len(cs)}_{pop}']=True
            details.append({'panel':len(cs),'population':pop,'lower_percent':low,'upper_percent':high,'finite_random_indices':finite,'random_min':min_seen,'random_max':max_seen})
    DETAILS['precursor_bounds']=details;DETAILS['supported_panel_excluded']=sorted(set(panel)-set(supported))

def check_stages(pr):
    wide=pd.read_parquet(SRC/'data_clean/wide_samples.parquet')
    fields=[f'{b}|{c}' for b in ('lo','hi') for c in COMMON]
    keys=['WWTP_CODE','date'];one=wide[wide.groupby(keys+['loc']).loc.transform('size').eq(1)]
    tri=None
    for side in ('INF','PE','EFF'):
        q=one[one['loc'].eq(side)][keys+fields].rename(columns={c:side+'_'+c for c in fields})
        tri=q if tri is None else tri.merge(q,on=keys,how='inner',validate='one_to_one')
    required=[f'{s}_hi|{c}' for s in ('INF','PE','EFF') for c in COMMON]
    tri=tri[tri[required].notna().all(1)].merge(pr[keys],on=keys,how='inner',validate='one_to_one').reset_index(drop=True)
    assert len(tri)==90 and tri.WWTP_CODE.nunique()==11
    source_pairs=pd.read_csv(OUT/'stage_measurement_pair_bounds.csv');source_pairs['date']=pd.to_datetime(source_pairs.date)
    source_plants=pd.read_csv(OUT/'stage_measurement_plant_bounds.csv');source_summary=pd.read_csv(OUT/'stage_measurement_bounds.csv')
    plant=tri.WWTP_CODE.to_numpy();ug=np.unique(plant);details=[]
    for name,cs in [('sumPFCA',PFCA),('sumPFSA',PFSA),('PFHxA',['PFHxA']),('PFOA',['PFOA'])]:
        mw=np.array([MW[c] for c in cs]);b={}
        for side in ('INF','PE','EFF'):
            for bound in ('lo','hi'):b[side+'_'+bound]=np.column_stack([tri[f'{side}_{bound}|{c}'] for c in cs])
        # Evaluate two step ratios with a single shared PE value at each corner.
        lower=log_ratio(b['PE_hi'],b['EFF_lo'],mw)-log_ratio(b['INF_lo'],b['PE_hi'],mw)
        upper=log_ratio(b['PE_lo'],b['EFF_hi'],mw)-log_ratio(b['INF_hi'],b['PE_lo'],mw)
        assert not np.isnan(lower).any() and not np.isnan(upper).any()
        expected=tri[keys].merge(source_pairs[source_pairs.outcome.eq(name)],on=keys,validate='one_to_one')
        close(lower,expected.lower,'stage_shared_PE_corner_lower_'+name);close(upper,expected.upper,'stage_shared_PE_corner_upper_'+name)
        lp=plant_array(lower,plant,ug);up=plant_array(upper,plant,ug)
        ps=source_plants[source_plants.outcome.eq(name)].set_index('plant').reindex(ug)
        close(lp,ps.lower,'stage_plant_mean_lower_'+name);close(up,ps.upper,'stage_plant_mean_upper_'+name)
        row=source_summary[source_summary.outcome.eq(name)].iloc[0]
        close([np.mean(lp),np.mean(up)],[row.mean_lower,row.mean_upper],'stage_summary_mean_'+name)
        close([np.sum(lp>0),np.sum(up<0),np.sum((lp<=0)&(up>=0))],[row.guaranteed_biological_larger,row.guaranteed_primary_larger,row.unresolved],'stage_sign_counts_'+name)
        for k in range(256):
            kind='uniform' if k%2 else 'log_uniform';values={s:fill(b,s,kind) for s in ('INF','PE','EFF')}
            delta=log_ratio(values['PE'],values['EFF'],mw)-log_ratio(values['INF'],values['PE'],mw)
            assert np.all(delta>=lower-1e-10) and np.all(delta<=upper+1e-10)
            pm=plant_array(delta,plant,ug)
            assert np.all(pm>=lp-1e-10) and np.all(pm<=up+1e-10)
        CHECKS['stage_256_random_shared_PE_fills_'+name]=True
        if name=='PFOA':
            assert all(np.array_equal(b[s+'_lo'],b[s+'_hi']) for s in ('INF','PE','EFF'))
            CHECKS['PFOA_all90_triples_all_three_matrices_detected']=True
        details.append({'endpoint':name,'mean_lower':float(np.mean(lp)),'mean_upper':float(np.mean(up)),'strictly_biological_larger_plants':int(np.sum(lp>0)),'strictly_primary_larger_plants':int(np.sum(up<0)),'unresolved':int(np.sum((lp<=0)&(up>=0)))})
    DETAILS['stage_bounds']=details

def main():
    pr=pd.read_parquet(SRC/'data_clean/pairs_inf_eff.parquet')
    pr=pr[pr[[f'{s}_hi|{c}' for s in ('INF','EFF') for c in COMMON]].notna().all(1)].reset_index(drop=True)
    plants=pd.read_csv(SRC/'data_clean/plants.csv');plants['plant']=plants.code.str.replace(r' \(.*\)','',regex=True)
    primary=plants.drop_duplicates('plant').set_index('plant').liquid.str.lower().str.startswith('primary')
    pr['bio']=(~pr.WWTP_CODE.map(primary)).astype(float)
    assert len(pr)==373 and pr.WWTP_CODE.nunique()==54
    check_families(pr);check_precursors(pr);check_stages(pr)
    probe=pd.DataFrame({'g':[0,0,1,1],'v':[1.,np.inf,1.,-np.inf]}).groupby('g').v.mean().to_numpy()
    DETAILS['pandas_infinity_probe']=probe.tolist();CHECKS['current_pandas_preserves_one_sided_infinity']=bool(np.array_equal(probe,np.array([np.inf,-np.inf])))
    files=['measurement_bounds_summary.csv','family_measurement_pair_bounds.csv','family_measurement_plant_bounds.csv','precursor_measurement_bounds.csv','stage_measurement_bounds.csv','stage_measurement_pair_bounds.csv','stage_measurement_plant_bounds.csv']
    DETAILS['verified_output_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}
    payload={'checks':CHECKS,'all_pass':bool(all(CHECKS.values())),'details':DETAILS,'interpretation':'Sharp measurement-compatible descriptive bounds under the rectangular concentration bounds, conditional on fixed design weights where applied; not DML/causal/confidence bounds.'}
    (OUT/'independent_measurement_bounds_invariants.json').write_text(json.dumps(payload,indent=2),'utf-8')
    print(json.dumps(payload,indent=2));assert payload['all_pass']

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
