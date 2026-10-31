"""Plant-first, outcome-blind comparability analyses for the PFAS manuscript.

No source data or frozen estimates are modified. Design weights use descriptors
and deterministic DL/2 influent summaries only. Outcome summaries use the ten
existing completed concentration matrices; their inference is conditional on
those imputations and on the fitted design, not a full-pipeline bootstrap.
"""
from __future__ import annotations
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import optimize,stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT
sys.path.insert(0,str(FROZEN/'scripts'))
import s03_upgrade_effect as s3

DESIGN_COLS=s3.ZCOLS+['INF_PFCA','INF_PFSA','INF_PFOSA']
C_VALUES=(0.1,1.0,10.0)
PRIMARY_METHODS=('equal_plant','overlap_C1','region_era_standardized')

def load_pre_design(correct_q=False):
    pr=s3.load_design().copy()
    if correct_q:
        keep=pr.WWTP_CODE.eq('Q') & pr.date.ge('2024-01-01')
        pr.loc[keep,'log_flow']=np.log10(275000.)
    pl=pd.read_csv(FROZEN/'data_clean/plants.csv')
    pl['code2']=pl.code.str.replace(r' \(.*\)','',regex=True)
    region=pl.drop_duplicates('code2').set_index('code2').receiving.str.strip()
    pr['receiving_region']=pr.WWTP_CODE.map(region)
    lo,hi=s3.bounds(pr,'INF')
    v=np.where(np.isfinite(lo),lo,hi-np.log(2))
    for name,grp in [('PFCA',s3.PFCA),('PFSA',s3.PFSA)]:
        pr['INF_'+name]=np.log10(sum(np.exp(v[:,s3.CMP.index(c)])/s3.MW[c] for c in grp))
    pr['INF_PFOSA']=v[:,s3.CMP.index('PFOSA')]/np.log(10)
    pr['row_index']=np.arange(len(pr));pr['era']=np.where(pr.filtered.eq(1),'filtered_pre2018','wholewater_2018plus')
    return pr

def ess(w):
    w=np.asarray(w,float)
    return float(w.sum()**2/np.sum(w*w)) if np.sum(w*w)>0 else 0.

def design(pr):
    pp=pr.groupby('WWTP_CODE',sort=True)[DESIGN_COLS+['bio']].mean()
    pp['receiving_region']=pr.groupby('WWTP_CODE').receiving_region.first()
    d=pp.bio.to_numpy();x=pp[DESIGN_COLS].to_numpy()
    scaler=StandardScaler().fit(x);xs=scaler.transform(x)
    active=np.std(xs,axis=0)>1e-10
    wp=pd.DataFrame({'plant':pp.index,'d':d,'receiving_region':pp.receiving_region.values,'weight_equal':1.})
    supports=[]
    for C in C_VALUES:
        key=f'C{C:g}'
        fit=LogisticRegression(C=C,max_iter=5000,tol=1e-10,solver='lbfgs').fit(xs[:,active],d)
        e=fit.predict_proba(xs[:,active])[:,1]
        wp['ps_'+key]=e;wp['weight_overlap_'+key]=np.where(d==1,1-e,e)
        lower=max(e[d==0].min(),e[d==1].min());upper=min(e[d==0].max(),e[d==1].max())
        inside=(e>=lower)&(e<=upper)
        wp['strict_support_'+key]=inside
        supports.append(dict(C=C,lower=lower,upper=upper,primary_retained=int(np.sum(inside&(d==0))),biological_retained=int(np.sum(inside&(d==1))),
                             estimate_allowed=bool(np.sum(inside&(d==0))>=4 and np.sum(inside&(d==1))>=6),
                             primary_ess=ess(wp.loc[d==0,'weight_overlap_'+key]),biological_ess=ess(wp.loc[d==1,'weight_overlap_'+key])))
    wp['ps']=wp.ps_C1;wp['weight_overlap']=wp.weight_overlap_C1
    return pp,wp,pd.DataFrame(supports),xs,active

def region_era_weights(pr):
    """Rake biological plant-era units to the six equal-primary-plant contexts."""
    unit=pr.groupby(['WWTP_CODE','receiving_region','era'],sort=True).agg(n_pairs=('bio','size'),d=('bio','first')).reset_index().rename(columns={'WWTP_CODE':'plant'})
    unit['base_weight']=unit.n_pairs/unit.groupby('plant').n_pairs.transform('sum')
    sums=unit.groupby(['receiving_region','era','d']).base_weight.sum().unstack('d',fill_value=0.)
    cells=[]
    for (region,era),row in sums.iterrows():
        a,b=float(row.get(0.,0)),float(row.get(1.,0))
        u=unit[(unit.receiving_region==region)&(unit.era==era)]
        cells.append(dict(receiving_region=region,era=era,primary_plants=int(u.loc[u.d==0,'plant'].nunique()),biological_plants=int(u.loc[u.d==1,'plant'].nunique()),
                          primary_base_mass=a,biological_base_mass=b,supported=bool(a>0 and b>0),target_primary_mass=a))
    cells=pd.DataFrame(cells)
    unsupported=cells[(cells.primary_base_mass>0)&~cells.supported]
    if len(unsupported):raise ValueError('Primary target has region-era cells without biological support; no silent exclusion is allowed.')
    ratio={(r.receiving_region,r.era):r.primary_base_mass/r.biological_base_mass if r.biological_base_mass>0 else 0. for r in cells.itertuples()}
    unit['weight_region_era']=np.where(unit.d.eq(0),unit.base_weight,unit.base_weight*np.array([ratio[(r.receiving_region,r.era)] for r in unit.itertuples()]))
    return unit,cells

def pair_weights(pr,wp,unit):
    out=pr[['row_index','WWTP_CODE','date','bio','receiving_region','era']].rename(columns={'WWTP_CODE':'plant','bio':'d'}).copy()
    n=pr.groupby('WWTP_CODE').size()
    mapper=wp.set_index('plant')
    for old,new in [('weight_equal','weight_equal_plant'),('weight_overlap_C0.1','weight_overlap_C0.1'),('weight_overlap_C1','weight_overlap_C1'),('weight_overlap_C10','weight_overlap_C10')]:
        out[new]=out.plant.map(mapper[old])/out.plant.map(n)
    uw=unit.set_index(['plant','era'])
    out['weight_region_era_standardized']=[uw.loc[(r.plant,r.era),'weight_region_era']/uw.loc[(r.plant,r.era),'n_pairs'] for r in out.itertuples()]
    out['weight_overlap']=out.weight_overlap_C1
    return out

def balance(pr,pw,pp,columns=None):
    # The denominator is the fixed pooled unweighted plant-level SD, shared by designs.
    d=pr.bio.to_numpy();dp=pp.bio.to_numpy();rows=[]
    for col in (DESIGN_COLS if columns is None else columns):
        sd=np.sqrt((pp.loc[dp==1,col].var(ddof=1)+pp.loc[dp==0,col].var(ddof=1))/2)
        for method in ['equal_plant','overlap_C0.1','overlap_C1','overlap_C10','region_era_standardized']:
            w=pw['weight_'+method].to_numpy();x=pr[col].to_numpy()
            m0=np.average(x[d==0],weights=w[d==0]);m1=np.average(x[d==1],weights=w[d==1])
            rows.append(dict(method=method,covariate=col,primary_mean=m0,biological_mean=m1,pooled_plant_sd=sd,SMD=(m1-m0)/sd if sd>0 else 0.))
    return pd.DataFrame(rows)

def full_calibration_feasibility(pp,xs,active):
    d=pp.bio.to_numpy();xb=xs[d==1][:,active];target=xs[d==0][:,active].mean(0)
    ae=np.vstack([np.ones(len(xb)),xb.T]);be=np.r_[1.,target]
    fit=optimize.linprog(np.zeros(len(xb)),A_eq=ae,b_eq=be,bounds=(0,None),method='highs')
    # Minimum attainable max standardized discrepancy with nonnegative biological weights.
    ap=np.vstack([np.c_[xb.T,-np.ones(len(target))],np.c_[-xb.T,-np.ones(len(target))]])
    bp=np.r_[target,-target]
    slack=optimize.linprog(np.r_[np.zeros(len(xb)),1.],A_ub=ap,b_ub=bp,A_eq=np.array([np.r_[np.ones(len(xb)),0.]]),b_eq=[1.],bounds=(0,None),method='highs')
    return {'exact_all_covariate_calibration_feasible':bool(fit.success),'LP_status':int(fit.status),'LP_message':fit.message,
            'minimum_max_discrepancy_in_overall_standardized_units':float(slack.fun) if slack.success else None,
            'interpretation':'Nonnegative biological plant weights matching all prespecified primary plant means; no outcome was used. Penalized overlap weighting does not have the unpenalized-logistic exact-balance guarantee.'}

def units_to_plant(pr,pw,values,method):
    """Return weighted numerator/denominator by plant; keep within-plant era weighting."""
    plant=pr.WWTP_CODE.to_numpy();ug=np.unique(plant);w=pw['weight_'+method].to_numpy()
    denominators=np.array([w[plant==g].sum() for g in ug])
    numerator=np.array([np.sum(values[:,plant==g]*w[plant==g],axis=1) for g in ug]).T
    d=pr.groupby('WWTP_CODE').bio.first().reindex(ug).to_numpy()
    return ug,d,denominators,numerator

def pooled_weighted_summary(pr,pw,values,method,bootstrap=1999,seed=4927):
    ug,d,den,num=units_to_plant(pr,pw,values,method)
    alive=den>0;ug,d,den,num=ug[alive],d[alive],den[alive],num[:,alive]
    means={k:num[:,d==k].sum(1)/den[d==k].sum() for k in (0,1)}
    theta=means[1]-means[0];within=np.zeros(len(theta))
    for k in (0,1):
        G=np.sum(d==k)
        score=(num[:,d==k]-means[k][:,None]*den[d==k])/den[d==k].sum()
        within+=G/(G-1)*np.sum(score**2,axis=1)
    M=len(theta);b=np.var(theta,ddof=1) if M>1 else 0.;u=within.mean();total=u+(1+1/M)*b
    # The five-reference-degree interval is a conservative sensitivity convention,
    # not a claim of exact small-sample or full-imputation calibration.
    df_ref=max(1,int(np.sum(d==0)-1));df_mi=(M-1)*(1+u/((1+1/M)*b))**2 if b>0 and M>1 else np.inf
    df=min(df_ref,df_mi);crit=stats.t.ppf(.975,df);se=np.sqrt(total);q=float(theta.mean())
    rng=np.random.default_rng(seed);boots=[]
    for _ in range(bootstrap):
        ix0=rng.choice(np.flatnonzero(d==0),np.sum(d==0),replace=True);ix1=rng.choice(np.flatnonzero(d==1),np.sum(d==1),replace=True)
        boots.append(float(np.mean(num[:,ix1].sum(1)/den[ix1].sum()-num[:,ix0].sum(1)/den[ix0].sum())))
    return dict(method=method,theta=q,se=float(se),t_df=float(df),lo95=q-crit*se,hi95=q+crit*se,ratio_multiplier=10**q,
                ratio_lo=10**(q-crit*se),ratio_hi=10**(q+crit*se),primary_plants=int(np.sum(d==0)),biological_plants=int(np.sum(d==1)),
                primary_ESS=ess(den[d==0]),biological_ESS=ess(den[d==1]),within_variance=float(u),between_imputation_variance=float(b),
                conditional_plant_bootstrap_lo=float(np.quantile(boots,.025)),conditional_plant_bootstrap_hi=float(np.quantile(boots,.975)),bootstrap=bootstrap),theta

def outcome_arrays(pr,path=None):
    source=Path(path) if path else FROZEN/'results/s03_imputations.npz'
    with np.load(source) as z:imps=z['imps']
    assert imps.ndim==3 and imps.shape[1:]==(len(pr),22),(source,imps.shape)
    out={}
    for V in imps:
        ys=s3.outcomes(V[:,:11],V[:,11:])
        for c in ('sumPFCA','sumPFSA','PFOA','PFOS'):out.setdefault(c,[]).append(ys[c])
    return {k:np.array(v) for k,v in out.items()},source

def run(outdir,bootstrap=1999,design_only=False,imputations=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pr=load_pre_design();pp,wp,support,xs,active=design(pr);unit,cells=region_era_weights(pr);pw=pair_weights(pr,wp,unit)
    protocol={'frozen_before_outcomes_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':'plant; plant-era units retained within cluster where standardized',
              'design_covariates':DESIGN_COLS,'propensity':'fixed ridge logistic C=1; C=0.1/10 prespecified design sensitivity; standardization fit54plantmeans',
              'inflow_handling':'deterministic observed value or DL/2 only; no effluent values used in design',
              'effect_endpoints':['sumPFCA','sumPFSA','PFOA','PFOS'],'primary_designs':list(PRIMARY_METHODS),
              'support_gate':'No hard-support contrast unless at least4primary and6biologicalplants remain',
              'uncertainty':'Conditional fixed-design plant aggregate score variance+between-MI variance; t interval capped at primaryplants-1 df; separate fixed-design stratifiedplantbootstrap of MIpooledpoint, no refit imputation/propensity',
              'target':'Equalplant population, penalized overlap population, and primary-reference region-era context are different targets; none is a causal upgrade effect.'}
    protocol_path=outdir/'overlap_protocol.json'
    if protocol_path.exists():
        saved=json.loads(protocol_path.read_text('utf-8'))
        assert {k:v for k,v in saved.items() if k!='frozen_before_outcomes_utc'}=={k:v for k,v in protocol.items() if k!='frozen_before_outcomes_utc'},'Design changed after freezing; create an explicitly separate sensitivity protocol.'
    else:protocol_path.write_text(json.dumps(protocol,indent=2),'utf-8')
    wp['weight_region_era']=wp.plant.map(unit.groupby('plant').weight_region_era.sum())
    wp.to_csv(outdir/'overlap_plant_weights.csv',index=False);pw.to_csv(outdir/'overlap_pair_weights.csv',index=False)
    unit.to_csv(outdir/'overlap_region_era_unit_weights.csv',index=False);cells.to_csv(outdir/'overlap_region_era_cells.csv',index=False)
    support.to_csv(outdir/'overlap_support.csv',index=False)
    bal=balance(pr,pw,pp);bal.to_csv(outdir/'overlap_balance.csv',index=False)
    feas=full_calibration_feasibility(pp,xs,active);(outdir/'overlap_calibration_feasibility.json').write_text(json.dumps(feas,indent=2),'utf-8')
    # Q update changes only a descriptor. Existing completed concentrations remain fixed.
    qpr=load_pre_design(correct_q=True);qpp,qwp,qsup,qxs,qactive=design(qpr);qunit,_=region_era_weights(qpr);qpw=pair_weights(qpr,qwp,qunit)
    affected=pr.WWTP_CODE.eq('Q')&pr.date.ge('2024-01-01')
    qmeta={'rows_affected':int(affected.sum()),'plant':'Q','effective_date':'2024-01-01','old_average_flow_m3_d':320000,'updated_average_flow_m3_d':275000,
           'process_class_unchanged':'activated sludge','max_abs_propensity_change':float(np.max(np.abs(qwp.ps-wp.ps))),
           'interpretation':'design-only sensitivity holding existing imputations fixed; not a complete re-imputation/refit'}
    (outdir/'overlap_Q_metadata.json').write_text(json.dumps(qmeta,indent=2),'utf-8')
    qwp.to_csv(outdir/'overlap_Q_corrected_plant_weights.csv',index=False)
    if design_only:return
    arrays,source=outcome_arrays(pr,imputations)
    rows=[];leave=[];qm=[]
    methods=['equal_plant','overlap_C0.1','overlap_C1','overlap_C10','region_era_standardized']
    for outcome,values in arrays.items():
        for method in methods:
            summary,_=pooled_weighted_summary(pr,pw,values,method,bootstrap=bootstrap)
            rows.append(dict(outcome=outcome,**summary))
        for control in pr.loc[pr.bio.eq(0),'WWTP_CODE'].unique():
            mask=pr.WWTP_CODE.ne(control).to_numpy();p2=pr.loc[mask].reset_index(drop=True);w2=pw.loc[mask].reset_index(drop=True)
            for method in PRIMARY_METHODS:
                # Influence only; no interval needed after dropping a reference.
                _,d,den,num=units_to_plant(p2,w2,values[:,mask],method)
                theta=(num[:,d==1].sum(1)/den[d==1].sum()-num[:,d==0].sum(1)/den[d==0].sum()).mean()
                leave.append(dict(outcome=outcome,method=method,omitted_primary=control,theta=theta,ratio_multiplier=10**theta))
        for method in PRIMARY_METHODS:
            _,d,den,num=units_to_plant(qpr,qpw,values,method)
            theta=(num[:,d==1].sum(1)/den[d==1].sum()-num[:,d==0].sum(1)/den[d==0].sum()).mean()
            qm.append(dict(outcome=outcome,method=method,theta=theta,ratio_multiplier=10**theta))
    pd.DataFrame(rows).to_csv(outdir/'overlap_contrasts.csv',index=False)
    pd.DataFrame(leave).to_csv(outdir/'overlap_leave_one_primary.csv',index=False)
    pd.DataFrame(qm).to_csv(outdir/'overlap_Q_corrected_contrasts.csv',index=False)
    (outdir/'overlap_sources.json').write_text(json.dumps({'imputations':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),'M':len(next(iter(arrays.values()))),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_data_modified':False},indent=2),'utf-8')
    print(pd.DataFrame(rows).query("outcome=='sumPFCA'").to_string(index=False),flush=True)
    print('Design and conditional contrasts written to',outdir,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'validation/results/overlap'));ap.add_argument('--bootstrap',type=int,default=1999);ap.add_argument('--design-only',action='store_true');ap.add_argument('--imputations')
    a=ap.parse_args();run(a.output,a.bootstrap,a.design_only,a.imputations)
