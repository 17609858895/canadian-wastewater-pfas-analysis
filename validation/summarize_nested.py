"""Assemble completed nested folds and quantify MI and small-reference sensitivity."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
import nested_imputation as ni


def ratio_transform(value):
    # Preserve genuinely unbounded sensitivity intervals; never cap critical values.
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        return float(np.float_power(10.,value))


def run(tag='field_nested_M20_s3_r200',require_complete=True):
    path=ni.OUT/'checkpoints'/tag;config=json.loads((path/'configuration.json').read_text())
    data,pr=ni.field_data();g=data['groups'];d=data['d'];n=len(d)
    assert config['data_hash']==ni.digest(data['Vlo'],data['Vhi'],data['Z'],d,g)
    endpoints=config['endpoints'];folds=list(GroupKFold(5).split(data['Z'],groups=g))
    complete=[m for m in range(config['M']) if all((path/f'm{m:03d}_fold{k:02d}.npz').exists() and (path/f'm{m:03d}_fold{k:02d}.json').exists() for k in range(5))]
    if require_complete and len(complete)!=config['M']:raise RuntimeError(f'Only{len(complete)}/{config["M"]}imputationscomplete')
    estimates=[];score_arrays={};fold_rows=[];score_diagnostics=[];completed_matrices=[]
    group_treatment=pd.Series(d,index=g).groupby(level=0).first()
    for m in complete:
        yy={key:np.zeros(n) for key in endpoints};mh={key:np.zeros(n) for key in endpoints};eh=np.zeros(n)
        Vi=np.zeros((n,len(data['compound_names'])))
        for k,(train,test) in enumerate(folds):
            cache=np.load(path/f'm{m:03d}_fold{k:02d}.npz');eh[test]=cache['eh'];Vi[test]=cache['Vi']
            for key in endpoints:yy[key][test]=cache['y_'+key];mh[key][test]=cache['mh_'+key]
            audit=json.loads((path/f'm{m:03d}_fold{k:02d}.json').read_text())
            fold_rows.append({key:value for key,value in audit.items() if key!='conditional_models'})
        if all(name in yy for name in data['compound_names']):
            Ve=np.column_stack([Vi[:,j]+ni.LN10*yy[name] for j,name in enumerate(data['compound_names'])])
            V=np.hstack([Vi,Ve]);censored=~np.isfinite(data['Vlo'])
            assert np.allclose(V[~censored],data['Vlo'][~censored],rtol=0,atol=1e-10)
            assert np.all(V[censored]<=data['Vhi'][censored]+1e-10)
            completed_matrices.append(V)
        for key in endpoints:
            q,u,psi,rd=ni.plr_scores(yy[key],d,mh[key],eh,g)
            estimates.append(dict(imputation=m,outcome=key,theta=q,variance=u))
            cluster=pd.Series(psi,index=g).groupby(level=0).sum()
            square=cluster**2;total=float(square.sum());share=square/total
            largest=share.idxmax()
            score_diagnostics.append(dict(imputation=m,outcome=key,
                primary_squared_score_share=float(share[group_treatment.loc[share.index]==0].sum()),
                largest_cluster_share=float(share.max()),largest_cluster=str(largest),
                largest_cluster_primary=bool(group_treatment.loc[largest]==0),
                score_contribution_count=float(total**2/(square**2).sum()),
                primary_plants=int((group_treatment==0).sum()),biological_plants=int((group_treatment==1).sum())))
            for name,value in dict(y=yy[key],mh=mh[key],eh=eh,psi=psi,rd=rd).items():score_arrays[f'm{m}_{key}_{name}']=value
    frame=pd.DataFrame(estimates);summary=[];convergence=[];t5=stats.t.ppf(.975,5)
    if len(frame):
        for endpoint in endpoints:
            f=frame[frame.outcome==endpoint].sort_values('imputation');pooled=ni.pool(f.to_dict('records'))
            total=pooled['se']**2;B=pooled['between_variance'];M=pooled['M']
            pooled.update(outcome=endpoint,ratio=ratio_transform(pooled['theta']),ratio_lo95=ratio_transform(pooled['lo95']),ratio_hi95=ratio_transform(pooled['hi95']),
                lo95_t5=pooled['theta']-t5*pooled['se'],hi95_t5=pooled['theta']+t5*pooled['se'],
                between_variance_fraction=(1+1/M)*B/total if total>0 else 0,
                MCSE_fraction_SE=pooled['imputation_MCSE']/pooled['se'] if pooled['se']>0 else 0)
            pooled['ratio_lo95_t5']=ratio_transform(pooled['lo95_t5']);pooled['ratio_hi95_t5']=ratio_transform(pooled['hi95_t5'])
            for df_complete,label in ((53,'BR53'),(5,'BR5')):
                br=ni.barnard_rubin_df(M,pooled['within_variance'],B,df_complete)
                critical=stats.t.ppf(.975,br['df_BR']) if br['df_BR']>0 else float('inf')
                pooled.update({label+'_'+key:value for key,value in br.items()})
                pooled[label+'_lo95']=pooled['theta']-critical*pooled['se']
                pooled[label+'_hi95']=pooled['theta']+critical*pooled['se']
                pooled[label+'_ratio_lo95']=ratio_transform(pooled[label+'_lo95']);pooled[label+'_ratio_hi95']=ratio_transform(pooled[label+'_hi95'])
            summary.append(pooled)
            for prefix in (5,10,20):
                if set(range(prefix)).issubset(set(complete)):
                    sub=f[f.imputation<prefix];res=ni.pool(sub.to_dict('records'))
                    res.update(outcome=endpoint,prefix_M=prefix,q_min=float(sub.theta.min()),q_max=float(sub.theta.max()),
                        q10=float(sub.theta.quantile(.1)),q50=float(sub.theta.quantile(.5)),q90=float(sub.theta.quantile(.9)))
                    br=ni.barnard_rubin_df(prefix,res['within_variance'],res['between_variance'],53)
                    res.update({'BR53_'+key:value for key,value in br.items()})
                    convergence.append(res)
    prefix=ni.OUT/'results'/tag
    pd.DataFrame(summary).to_csv(prefix.with_suffix('.csv'),index=False)
    frame.to_csv(prefix.parent/(tag+'_imputations.csv'),index=False)
    pd.DataFrame(convergence).to_csv(prefix.parent/(tag+'_convergence.csv'),index=False)
    diagnostics=pd.DataFrame(score_diagnostics)
    diagnostics.to_csv(prefix.parent/(tag+'_score_diagnostics_by_imputation.csv'),index=False)
    score_summary=[]
    for endpoint,f in diagnostics.groupby('outcome'):
        row=dict(outcome=endpoint,M=len(f))
        for key in ('primary_squared_score_share','largest_cluster_share','score_contribution_count'):
            row[key+'_mean']=float(f[key].mean());row[key+'_q10']=float(f[key].quantile(.1));row[key+'_q90']=float(f[key].quantile(.9))
        score_summary.append(row)
    pd.DataFrame(score_summary).to_csv(prefix.parent/(tag+'_score_diagnostics.csv'),index=False)
    np.savez_compressed(prefix.parent/(tag+'_scores.npz'),**score_arrays)
    if completed_matrices:
        imps=np.asarray(completed_matrices);lo=data['Vlo'];hi=data['Vhi'];P=len(data['compound_names']);imputation_rows=[]
        np.savez_compressed(prefix.parent/(tag+'_completed.npz'),imps=imps,Vlo=lo,Vhi=hi,imputation_indices=np.asarray(complete))
        for j in range(lo.shape[1]):
            cens=~np.isfinite(lo[:,j])
            if not cens.any():continue
            rel=(imps[:,cens,j]-hi[cens,j][None,:])/ni.LN10
            between=imps[:,cens,j].std(axis=0)/ni.LN10
            obs=(lo[~cens,j]-np.log(np.median(np.exp(hi[cens,j]))))/ni.LN10
            imputation_rows.append(dict(compound=data['compound_names'][j%P],side='INF' if j<P else 'EFF',
                pct_censored=float(100*cens.mean()),med_rel=float(np.median(rel)),q10_rel=float(np.percentile(rel,10)),
                q90_rel=float(np.percentile(rel,90)),between_sd=float(np.median(between)),obs_rel_med=float(np.median(obs)) if len(obs) else np.nan))
        pd.DataFrame(imputation_rows).to_csv(prefix.parent/(tag+'_imputed.csv'),index=False)
    pd.DataFrame(fold_rows).to_csv(prefix.parent/(tag+'_fold_audit.csv'),index=False)
    result=dict(status='COMPLETE' if len(complete)==config['M'] else 'PARTIAL_NOT_FINAL',
                M_complete=len(complete),M_requested=config['M'],complete_imputation_indices=complete,
                summary=summary,config=config,
                t5_scope='Critical-value sensitivity using six primary-only reference plants; not a guarantee of valid small-cluster inference.',
                MCSE_scope='sqrt(B/M) measures conditional Monte Carlo variation of the pooled point estimate across these imputation chains.')
    result['score_diagnostic_scope']='Squared cluster-score shares describe concentration of the within-imputation score variance. Their inverse Herfindahl count is a diagnostic, not a formal degrees of freedom or effective sample size; it excludes between-imputation variance.'
    result['score_diagnostics']=score_summary
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='field_nested_M20_s3_r200');parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    run(args.tag,not args.allow_partial)
