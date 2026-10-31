"""Replicate-level validation of censored aggregate PFCA contrasts.

The generating design is fixed to the monitored plants. Each simulation has a
known constant log10 effect on the molar PFCA sum; a separately generated PFOS
endpoint remains null. This is an empirical calibration experiment, not a
guarantee for the observational data. No frozen inputs are modified.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT
DEFAULT_OUT = HERE / 'results' / 'expanded_benchmark'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FROZEN / 'scripts'))
import nested_imputation as ni
import s03_upgrade_effect as old

PFCA = list(old.PFCA)
COMPOUNDS = PFCA + ['PFOS']
ENDPOINTS = {'sumPFCA': PFCA, 'PFOS': ['PFOS']}
DGPS = ('gaussian_iid', 'clustered_gaussian', 'clustered_t3')
METHODS = ('oracle', 'DL/2', 'global_mi', 'nested_mi', 'known_h')
LN10 = np.log(10.)


def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem+f'.{os.getpid()}.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    tmp.replace(path)


def detection_limits(pr, side, records, compounds=None):
    """Recover reported DLs for detected as well as non-detected measurements.

    The concentration wide table replaces detected bounds with the measured
    value. Actual laboratory DLs are retained in the long table and are joined
    here using the same unambiguous plant/date/location pairing. Missing DLs
    alone use same-year then pooled matrix/compound median fallbacks.
    """
    columns = [];audit=[]
    compounds=COMPOUNDS if compounds is None else list(compounds)
    location={'INF':'Influent','EFF':'Effluent'}[side]
    for cmp in compounds:
        selected=records[(records.LOCATION==location)&(records.COMPOUND==cmp)]
        included=selected.merge(pr[['WWTP_CODE','date']],on=['WWTP_CODE','date'],how='inner')
        assert included.groupby(['WWTP_CODE','date']).AXYS_ID.nunique().max() <= 1,'Ambiguous analytical sample in paired design'
        limits=selected.groupby(['WWTP_CODE','date']).dl.first()
        index=pd.MultiIndex.from_frame(pr[['WWTP_CODE','date']])
        value=pd.Series(limits.reindex(index).to_numpy(float),index=pr.index)
        value=value.where(value>0)
        reported=value.notna()
        year_median = value.groupby(pr.year).transform('median')
        value = value.fillna(year_median).fillna(value.median())
        if value.isna().any():
            raise ValueError(f'No reported censored limits for {side}/{cmp}; cannot define fallback DL.')
        columns.append(np.log(value.to_numpy(float)))
        former=pr[f'{side}_hi|{cmp}'].where(pr[f'{side}_lo|{cmp}']==0)
        former=former.fillna(former.groupby(pr.year).transform('median')).fillna(former.median())
        audit.append(dict(matrix=side,compound=cmp,reported_DL=int(reported.sum()),fallback_DL=int((~reported).sum()),
                          detected_with_reported_DL=int((reported&(pr[f'{side}_lo|{cmp}']>0)).sum()),
                          old_ND_only_DL_missing=int(former.isna().sum()),
                          min_DL=float(value.min()),median_DL=float(value.median()),max_DL=float(value.max())))
    return np.column_stack(columns),audit


def source_design(outer_folds=3):
    pr = old.load_design()
    complete = np.load(FROZEN / 'results/s03_imputations.npz')['imps'][0]
    selected = [old.CMP.index(cmp) for cmp in COMPOUNDS]
    records=pd.read_parquet(FROZEN/'data_clean/env_long.parquet')
    records=records[~records.is_dup & records.usable]
    dli,ai=detection_limits(pr,'INF',records);dle,ae=detection_limits(pr,'EFF',records)
    data = dict(pr=pr, Vi0=complete[:, selected], dli=dli,
                dle=dle, DL_audit=ai+ae, Z=pr[old.ZCOLS].to_numpy(float),
                d=pr.bio.to_numpy(float), groups=pr.WWTP_CODE.to_numpy(str))
    data['folds'] = list(GroupKFold(outer_folds).split(data['Z'], groups=data['groups']))
    data['hash'] = ni.digest(data['Vi0'], data['dli'], data['dle'], data['Z'], data['d'], data['groups'])
    return data


def generate(data, dgp, effect, dl_multiplier, rep, base_seed=20261031):
    """Generate exact aggregate effects with compound-specific composition noise.

    Normalized composition innovations change individual PFCA ratios while
    preserving the prescribed molar sum. Common random numbers are used across
    effect and DL settings within a DGP/replicate. A plant shock is sampled anew
    in each replicate, shared by its repeated wastewater pairs.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), DGPS.index(dgp), int(rep)]))
    Z, d, groups = data['Z'], data['d'], data['groups']
    _, group_index = np.unique(groups, return_inverse=True)
    n, p = len(d), len(COMPOUNDS)
    G = group_index.max() + 1
    Vi = data['Vi0'] + rng.normal(0, .20, (n, p))
    zflow = Z[:, old.ZCOLS.index('log_flow')] - 4.5
    residence = (Z[:, old.ZCOLS.index('pres')] - 85.) / 15.
    filtered = Z[:, old.ZCOLS.index('filtered')]
    inlet = np.log10((np.exp(Vi[:, :len(PFCA)]) / np.array([old.MW[c] for c in PFCA])).sum(1))
    inlet -= inlet.mean()
    h_pfca = .15*np.tanh(zflow) + .10*filtered - .08*residence - .12*inlet + .08*zflow*filtered
    h_pfos = .10*np.tanh(zflow) - .05*filtered - .06*residence - .08*(Vi[:, -1]/LN10 - Vi[:, -1].mean()/LN10)
    if dgp == 'gaussian_iid':
        epfca, epfos = rng.normal(0, .35, (2, n))
    elif dgp == 'clustered_gaussian':
        epfca = .35*(.60*rng.normal(size=G)[group_index] + .80*rng.normal(size=n))
        epfos = .35*(.60*rng.normal(size=G)[group_index] + .80*rng.normal(size=n))
    elif dgp == 'clustered_t3':
        # t3 has variance 3. Scaling retains the unit-variance noise convention.
        sigma = .35*(.65 + .35*np.minimum(np.abs(zflow), 3.))
        epfca = sigma*(.60*rng.standard_t(3, G)[group_index] + .80*rng.standard_t(3, n))/np.sqrt(3.)
        epfos = sigma*(.60*rng.standard_t(3, G)[group_index] + .80*rng.standard_t(3, n))/np.sqrt(3.)
    else:
        raise ValueError(dgp)
    theta = .25 if effect == 'effect' else 0.
    relative = rng.normal(0, .25, (n, len(PFCA)))
    mol_inlet = np.exp(Vi[:, :len(PFCA)]) / np.array([old.MW[c] for c in PFCA])
    normalization = np.log((mol_inlet*np.exp(relative)).sum(1)/mol_inlet.sum(1))
    Ve = np.empty_like(Vi)
    Ve[:, :len(PFCA)] = Vi[:, :len(PFCA)] + LN10*(h_pfca + theta*d + epfca)[:, None] + relative - normalization[:, None]
    Ve[:, -1] = Vi[:, -1] + LN10*(h_pfos + epfos)
    latent = np.column_stack([Vi, Ve])
    y = ni.endpoints_from_completed(latent, COMPOUNDS, old.MW, ENDPOINTS)
    assert np.max(np.abs(y['sumPFCA']-(h_pfca+theta*d+epfca))) < 1e-10
    assert np.max(np.abs(y['PFOS']-(h_pfos+epfos))) < 1e-10
    upper = np.column_stack([data['dli'], data['dle']]) + np.log(float(dl_multiplier))
    censored = latent <= upper
    Vlo = np.where(censored, -np.inf, latent)
    Vhi = np.where(censored, upper, latent)
    return dict(Vlo=Vlo, Vhi=Vhi, latent=latent, truth={'sumPFCA':theta, 'PFOS':0.},
                h={'sumPFCA':h_pfca, 'PFOS':h_pfos}, censored=censored)


def complete_dml(V, data, light=True):
    p = len(COMPOUNDS); Z, d, groups = data['Z'], data['d'], data['groups']
    ys = ni.endpoints_from_completed(V, COMPOUNDS, old.MW, ENDPOINTS)
    mh = {endpoint:np.zeros(len(d)) for endpoint in ENDPOINTS}; eh = np.zeros(len(d))
    for train, test in data['folds']:
        ztrain, ztest, _ = ni.fill_train_medians(Z[train], Z[test])
        Wtrain = np.column_stack([ztrain, V[train, :p]/LN10])
        Wtest = np.column_stack([ztest, V[test, :p]/LN10])
        eh[test] = ni.propensity(Wtrain, Wtest, d[train], light=light)
        for endpoint, y in ys.items():
            mh[endpoint][test] = np.mean([model.fit(Wtrain, y[train]).predict(Wtest) for model in ni.regressors(light=light)], axis=0)
    result = {}
    for endpoint, y in ys.items():
        theta, variance, _, _ = ni.plr_scores(y, d, mh[endpoint], eh, groups)
        result[endpoint] = dict(theta=theta, variance=variance)
    return result


def known_h_contrast(sim, data):
    """Unimplementable reference with the known generating baseline removed.

    Uses equally weighted plant means. It isolates cluster sampling variability
    from nuisance-model and imputation error; it is not a competing field model.
    """
    ys = ni.endpoints_from_completed(sim['latent'], COMPOUNDS, old.MW, ENDPOINTS)
    plant = pd.DataFrame({'group':data['groups'], 'd':data['d']}).groupby('group').first()
    result = {}
    for endpoint, y in ys.items():
        means = pd.Series(y-sim['h'][endpoint], index=data['groups']).groupby(level=0).mean()
        a, b = means[plant.d == 1].to_numpy(), means[plant.d == 0].to_numpy()
        va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
        se = np.sqrt(va+vb)
        df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
        result[endpoint] = dict(theta=float(a.mean()-b.mean()), se=float(se), within_variance=float(se**2),
                                between_variance=0., M=1, imputation_MCSE=0., welch_df=float(df))
    return result


def intervals(theta, se, data, welch_df=None):
    G = len(np.unique(data['groups']))
    G0 = len(np.unique(data['groups'][data['d'] == 0]))
    G1 = len(np.unique(data['groups'][data['d'] == 1]))
    # These alternative critical values are tested sensitivities, not guarantees.
    critical = {'normal':float(stats.norm.ppf(.975)), 'cluster_t':float(stats.t.ppf(.975, G-1)),
                'control_t':float(stats.t.ppf(.975, min(G0, G1)-1))}
    if welch_df is not None:
        critical['welch_t'] = float(stats.t.ppf(.975, welch_df))
    return {kind:(theta-c*se, theta+c*se) for kind,c in critical.items()}


_DATA = None
_CONFIG = None
def initialize_worker(config):
    global _DATA, _CONFIG
    _CONFIG = config; _DATA = source_design(config['outer_folds'])


def task_name(task):
    dgp, effect, k, rep = task
    return f'{dgp}__{effect}__dl{k}__rep{rep:05d}'


def one_task(task, config=None, data=None):
    config = config or _CONFIG; data = data or _DATA
    dgp, effect, k, rep = task; key = task_name(task)
    target = Path(config['output'])/'replicates'/(key+'.json')
    if target.exists():
        return key, 'cached'
    started = time.perf_counter(); sim = generate(data, dgp, effect, k, rep, config['seed'])
    rows=[]; failure=[]; timings={}
    with threadpool_limits(limits=1):
        for method in config['methods']:
            start = time.perf_counter()
            try:
                if method in ('oracle', 'DL/2'):
                    completed = sim['latent'] if method == 'oracle' else np.where(np.isfinite(sim['Vlo']), sim['Vlo'], sim['Vhi']-np.log(2.))
                    estimates = complete_dml(completed, data, config['light'])
                    summary = {name:ni.pool([value]) for name,value in estimates.items()}
                elif method == 'known_h':
                    summary = known_h_contrast(sim, data)
                elif method == 'global_mi':
                    zd = np.column_stack([data['Z'], data['d']]); estimates={name:[] for name in ENDPOINTS}
                    for m in range(config['M']):
                        completed, _, _ = ni.impute_fold(sim['Vlo'],sim['Vhi'],np.empty((0,2*len(COMPOUNDS))),np.empty((0,2*len(COMPOUNDS))),
                                                         zd,np.empty((0,zd.shape[1])),data['groups'],
                                                         n_sweeps=config['sweeps'],rounds=config['rounds'],inner_folds=config['inner_folds'],
                                                         seed=config['seed']+rep*1000003+m*10007)
                        fitted = complete_dml(completed, data, config['light'])
                        for name,value in fitted.items():estimates[name].append(value)
                    summary = {name:ni.pool(values) for name,values in estimates.items()}
                elif method == 'nested_mi':
                    # Detailed per-fold stdout is kept local; one completion line per replicate is emitted below.
                    with redirect_stdout(io.StringIO()):
                        result = ni.nested_dml(sim['Vlo'],sim['Vhi'],data['Z'],data['d'],data['groups'],COMPOUNDS,old.MW,ENDPOINTS,
                                              M=config['M'],n_sweeps=config['sweeps'],rounds=config['rounds'],outer_folds=config['outer_folds'],
                                              inner_folds=config['inner_folds'],seed=config['seed']+rep*1000003,
                                              light=config['light'],folds=data['folds'],return_scores=False)
                    summary = result['summary']
                else:
                    raise ValueError(method)
                for name,value in summary.items():
                    truth = sim['truth'][name]
                    record = dict(dgp=dgp,effect=effect,dl_multiplier=k,rep=rep,method=method,outcome=name,truth=truth,**value)
                    for interval,(lo,hi) in intervals(value['theta'],value['se'],data,value.get('welch_df')).items():
                        record.update({f'lo_{interval}':lo,f'hi_{interval}':hi,f'cover_{interval}':bool(lo<=truth<=hi),
                                       f'reject_{interval}':bool(lo>0 or hi<0),f'width_{interval}':hi-lo})
                    indices=[COMPOUNDS.index(c) for c in ENDPOINTS[name]]
                    record['censoring_inf'] = float(sim['censored'][:,indices].mean())
                    record['censoring_eff'] = float(sim['censored'][:,[len(COMPOUNDS)+j for j in indices]].mean())
                    record['n_pairs']=len(data['d']);record['plants']=len(np.unique(data['groups']))
                    record['primary_plants']=len(np.unique(data['groups'][data['d']==0]))
                    assert np.isfinite(value['theta']) and np.isfinite(value['se']) and value['se']>=0
                    rows.append(record)
            except Exception as exc:
                failure.append(dict(method=method,error=type(exc).__name__,message=str(exc)))
            timings[method]=time.perf_counter()-start
    result=dict(task=key,rows=rows,failures=failure,timings_seconds=timings,elapsed_seconds=time.perf_counter()-started,
                data_hash=data['hash'],configuration_hash=config['configuration_hash'])
    atomic_json(result,target)
    return key, 'failed' if failure else 'completed'


def wilson(successes,n):
    if n==0:return math.nan,math.nan
    z=stats.norm.ppf(.975);p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(max(0,min(p,center-half))),float(min(1,max(p,center+half)))


def add_finite_m_intervals(frame):
    """Add standard finite-M critical values without changing fitted estimates.

    Rubin's old df is (M-1)/lambda^2. Barnard--Rubin combines it with
    df_obs=df_com*(df_com+1)/(df_com+3)*(1-lambda). The complete-data
    choice G-1 is a cluster approximation; substituting min(G0,G1)-1 is
    explicitly a reference-limited sensitivity rule, not an exact theorem.
    """
    frame=frame.copy()
    for index,row in frame.iterrows():
        needed=('within_variance','between_variance','M','plants','primary_plants','theta','se','truth')
        if any(key not in row or pd.isna(row[key]) for key in needed):
            frame.loc[index,'mi_interval_status']='unavailable_components';continue
        G=int(row.plants);G0=int(row.primary_plants);G1=G-G0
        frame.loc[index,'G']=G;frame.loc[index,'G0']=G0;frame.loc[index,'G1']=G1
        M=int(row.M)
        if M<2:
            for kind,df in [('br_cluster_t',G-1),('br_reference_t',min(G0,G1)-1)]:
                frame.loc[index,'df_'+kind]=df
                c=float(stats.t.ppf(.975,df));lo=float(row.theta)-c*float(row.se);hi=float(row.theta)+c*float(row.se)
                frame.loc[index,'lo_'+kind]=lo;frame.loc[index,'hi_'+kind]=hi
                frame.loc[index,'cover_'+kind]=bool(lo<=row.truth<=hi)
                frame.loc[index,'reject_'+kind]=bool(lo>0 or hi<0)
                frame.loc[index,'width_'+kind]=hi-lo
            frame.loc[index,'mi_interval_status']='non_MI_complete_data_t';continue
        U=float(row.within_variance);B=float(row.between_variance)
        T=U+(1+1/M)*B
        assert U>=0 and B>=0 and np.isclose(T,float(row.se)**2,rtol=1e-8,atol=1e-12)
        lam=(1+1/M)*B/T if T>0 else 0.
        nu_old=(M-1)/lam**2 if lam>0 else np.inf
        frame.loc[index,'fraction_between_MI']=lam
        frame.loc[index,'df_rubin']=nu_old
        values={'rubin_t':nu_old}
        for kind,df_com in [('br_cluster_t',G-1),('br_reference_t',min(G0,G1)-1)]:
            tmp=(1-lam)*(1+df_com)*df_com
            df=(M-1)*tmp/((df_com+3)*(M-1)+lam**2*tmp)
            frame.loc[index,'df_'+kind]=df;values[kind]=df
        for kind,df in values.items():
            if pd.isna(df) or df<=0:continue
            c=float(stats.t.ppf(.975,df));lo=float(row.theta)-c*float(row.se);hi=float(row.theta)+c*float(row.se)
            frame.loc[index,'lo_'+kind]=lo;frame.loc[index,'hi_'+kind]=hi
            frame.loc[index,'cover_'+kind]=bool(lo<=row.truth<=hi)
            frame.loc[index,'reject_'+kind]=bool(lo>0 or hi<0)
            frame.loc[index,'width_'+kind]=hi-lo
        frame.loc[index,'mi_interval_status']='finite_M_intervals_added'
    return frame


def summarize(output):
    output=Path(output);records=[];failures=[];timing=[]
    config=json.loads((output/'configuration.json').read_text('utf-8'))
    replicate_files=[p for p in (output/'replicates').glob('*.json') if not p.name.endswith('.tmp.json')]
    for path in sorted(replicate_files):
        value=json.loads(path.read_text('utf-8'))
        if value['configuration_hash']!=config['configuration_hash']:raise ValueError('Replicate/config mismatch')
        records.extend(value['rows']);failures.extend([dict(task=value['task'],**f) for f in value['failures']])
        timing.extend([dict(task=value['task'],method=m,seconds=s) for m,s in value['timings_seconds'].items()])
    if not records:raise ValueError('No completed estimates to summarize')
    frame=add_finite_m_intervals(pd.DataFrame(records));frame.to_csv(output/'replicate_estimates.csv',index=False)
    pd.DataFrame(failures,columns=['task','method','error','message']).to_csv(output/'failed_methods.csv',index=False)
    pd.DataFrame(timing).to_csv(output/'timings.csv',index=False)
    rows=[]
    keys=['dgp','effect','dl_multiplier','method','outcome']
    for key,part in frame.groupby(keys,sort=False):
        err=part.theta-part.truth;n=len(part)
        row=dict(zip(keys,key));rmse=float(np.sqrt(np.mean(err**2)))
        row.update(n_replicates=n,truth=float(part.truth.iloc[0]),bias=float(err.mean()),
                   bias_MCSE=float(err.std(ddof=1)/np.sqrt(n)) if n>1 else math.nan,RMSE=rmse,
                   RMSE_MCSE=float((err**2).std(ddof=1)/(2*rmse*np.sqrt(n))) if n>1 and rmse>0 else math.nan,
                   empirical_SD=float(part.theta.std(ddof=1)) if n>1 else math.nan,mean_SE=float(part.se.mean()),
                   median_SE=float(part.se.median()),censoring_inf=float(part.censoring_inf.mean()),censoring_eff=float(part.censoring_eff.mean()))
        for component in ('within_variance','between_variance','fraction_between_MI','df_rubin','df_br_cluster_t','df_br_reference_t'):
            if component in part and part[component].notna().any():row['mean_'+component]=float(part[component].mean())
        for interval in ('normal','cluster_t','control_t','welch_t','rubin_t','br_cluster_t','br_reference_t'):
            if f'cover_{interval}' not in part or part[f'cover_{interval}'].isna().all():continue
            keep=part[f'cover_{interval}'].notna();ni_=int(keep.sum())
            for criterion in ('cover','reject'):
                successes=int(part.loc[keep,f'{criterion}_{interval}'].sum());rate=successes/ni_
                lower,upper=wilson(successes,ni_)
                row.update({f'{criterion}_{interval}':rate,f'{criterion}_{interval}_MCSE':float(np.sqrt(rate*(1-rate)/ni_)),
                            f'{criterion}_{interval}_Wilson_lo':lower,f'{criterion}_{interval}_Wilson_hi':upper})
            row[f'width_{interval}']=float(part.loc[keep,f'width_{interval}'].mean())
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(output/'performance_summary.csv',index=False)
    pairs=[]
    setting=['dgp','effect','dl_multiplier','outcome','rep']
    for a,b in [('nested_mi','global_mi'),('nested_mi','DL/2'),('DL/2','oracle')]:
        left=frame[frame.method==a];right=frame[frame.method==b]
        paired=left.merge(right,on=setting,suffixes=('_a','_b'))
        for key,part in paired.groupby(setting[:-1],sort=False):
            rec=dict(zip(setting[:-1],key));rec.update(method_a=a,method_b=b,n_paired=len(part))
            for metric in ('cover_normal','reject_normal','width_normal','cover_br_cluster_t','reject_br_cluster_t','width_br_cluster_t','cover_br_reference_t','reject_br_reference_t','width_br_reference_t'):
                if metric+'_a' not in part or metric+'_b' not in part:continue
                use=part[[metric+'_a',metric+'_b']].notna().all(axis=1)
                if not use.any():continue
                diff=part.loc[use,metric+'_a'].astype(float)-part.loc[use,metric+'_b'].astype(float)
                rec[metric+'_difference']=float(diff.mean())
                rec[metric+'_difference_MCSE']=float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else math.nan
                rec[metric+'_n_paired']=len(diff)
            pairs.append(rec)
    pd.DataFrame(pairs).to_csv(output/'paired_method_differences.csv',index=False)
    status=dict(completed_task_files=len(replicate_files),successful_endpoint_estimates=len(frame),
                failed_method_runs=len(failures),requested_replicates_per_setting=config.get('requested_replicates'),
                configuration_hash=config['configuration_hash'],summary_rows=len(summary))
    number_settings=len(config['effects'])*(len(config['dgps']) if config.get('paired_settings') else len(config['dgps'])*len(config['limits']))
    expected_tasks=number_settings*config['requested_replicates']
    expected_rows=number_settings*len(config['methods'])*len(config['endpoints'])
    status.update(expected_task_files=expected_tasks,all_requested_tasks_present=len(replicate_files)==expected_tasks,
                  all_method_endpoint_cells_present=len(summary)==expected_rows,
                  all_cells_have_requested_replicates=bool((summary.n_replicates==config['requested_replicates']).all()))
    atomic_json(status,output/'completion_summary.json')
    print(json.dumps(status,indent=2),flush=True)
    return summary


def verify_generator(data):
    comparisons=[]
    for dgp in DGPS:
        null=generate(data,dgp,'null',1,0);effect=generate(data,dgp,'effect',1,0)
        yn=ni.endpoints_from_completed(null['latent'],COMPOUNDS,old.MW,ENDPOINTS)
        ye=ni.endpoints_from_completed(effect['latent'],COMPOUNDS,old.MW,ENDPOINTS)
        err=float(np.max(np.abs((ye['sumPFCA']-yn['sumPFCA'])-.25*data['d'])))
        assert err<1e-10 and np.array_equal(ye['PFOS'],yn['PFOS'])
        comparisons.append(dict(dgp=dgp,aggregate_effect_error=err,PFOS_null_unchanged=True))
    return dict(checks=comparisons,n_pairs=len(data['d']),plants=len(np.unique(data['groups'])),
                primary_plants=len(np.unique(data['groups'][data['d']==0])),data_hash=data['hash'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT/'pilot')
    parser.add_argument('--dgps',nargs='+',choices=DGPS,default=['gaussian_iid','clustered_t3'])
    parser.add_argument('--effects',nargs='+',choices=['null','effect'],default=['null','effect'])
    parser.add_argument('--limits',nargs='+',type=int,default=[1,3])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--reps',type=int,default=1);parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--M',type=int,default=2);parser.add_argument('--sweeps',type=int,default=1)
    parser.add_argument('--rounds',type=int,default=30);parser.add_argument('--inner-folds',type=int,default=3)
    parser.add_argument('--outer-folds',type=int,default=3)
    parser.add_argument('--paired-settings',action='store_true',help='Pair each DGP with its corresponding DL multiplier instead of a Cartesian product.')
    parser.add_argument('--seed',type=int,default=20261031);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--progress-every',type=int,default=10)
    parser.add_argument('--full-nuisance',action='store_true');parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--generator-check',action='store_true')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize(output);return
    if not (1<=args.workers<=4):raise ValueError('workers must be1–4; shared host budget is set by root')
    data=source_design(args.outer_folds);config=dict(M=args.M,sweeps=args.sweeps,rounds=args.rounds,inner_folds=args.inner_folds,
                                   seed=args.seed,methods=args.methods,dgps=args.dgps,effects=args.effects,limits=args.limits,
                                   outer_folds=args.outer_folds,paired_settings=args.paired_settings,
                                   light=not args.full_nuisance,source_design_hash=data['hash'])
    config['configuration_hash']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config.update(output=str(output),requested_replicates=args.reps+args.start,
                  generated_utc=datetime.now(timezone.utc).isoformat(),compound_names=COMPOUNDS,endpoints=ENDPOINTS)
    marker=output/'configuration.json'
    if marker.exists():
        previous=json.loads(marker.read_text('utf-8'))
        assert previous['configuration_hash']==config['configuration_hash'],'Existing output has a different configuration'
        config['generated_utc']=previous['generated_utc']
        config['requested_replicates']=max(previous['requested_replicates'],config['requested_replicates'])
        config['launch_history']=previous.get('launch_history',[])
    config.setdefault('launch_history',[]).append(dict(utc=datetime.now(timezone.utc).isoformat(),workers=args.workers,
                                                     first_rep=args.start,stop_rep=args.start+args.reps))
    atomic_json(config,marker)
    atomic_json(verify_generator(data),output/'generator_verification.json')
    pd.DataFrame(data['DL_audit']).to_csv(output/'laboratory_detection_limit_audit.csv',index=False)
    if args.generator_check:return
    if args.paired_settings:
        assert len(args.dgps)==len(args.limits),'Paired DGP/limit lists must have equal lengths'
        settings=list(zip(args.dgps,args.limits))
    else:settings=[(dgp,k) for dgp in args.dgps for k in args.limits]
    tasks=[(dgp,effect,k,rep) for rep in range(args.start,args.start+args.reps) for dgp,k in settings for effect in args.effects]
    if args.workers==1:
        for index,task in enumerate(tasks):
            key,status=one_task(task,config,data)
            if (index+1)%args.progress_every==0 or status=='failed' or index+1==len(tasks):
                print(f'{index+1}/{len(tasks)} {key} {status}',flush=True)
    else:
        with get_context('spawn').Pool(args.workers,initializer=initialize_worker,initargs=(config,)) as pool:
            for index,(key,status) in enumerate(pool.imap_unordered(one_task,tasks,chunksize=1)):
                if (index+1)%args.progress_every==0 or status=='failed' or index+1==len(tasks):
                    print(f'{index+1}/{len(tasks)} {key} {status}',flush=True)
    summarize(output)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
