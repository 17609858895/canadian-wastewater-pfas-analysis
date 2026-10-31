"""Detection-selected held-out reconstruction using frozen training chains.

Fit mode captures independent training-model chains. Analyze mode withholds
held-out detections below a training-detection quantile and replays those fixed
models. The withheld measured values are evaluation references; this design
does not identify calibration for genuinely unobserved below-limit values.
"""
from __future__ import annotations
import argparse,json,time,os
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import nested_imputation as ni


def fit_traces(*,M=5,n_sweeps=3,rounds=200,seed=42000,tag='masked_M5_s3_r200',indices=None,task_indices=None):
    if task_indices is not None and any(t<0 or t>=5*M for t in task_indices):raise ValueError('Trace task index outside M x 5 folds')
    data,pr=ni.field_data();g=data['groups'];d=data['d'];Z=data['Z'];lo=data['Vlo'];hi=data['Vhi']
    checkpoint=ni.OUT/'checkpoints'/tag;checkpoint.mkdir(parents=True,exist_ok=True)
    config=dict(M=M,n_sweeps=n_sweeps,rounds=rounds,seed=seed,data_hash=ni.digest(lo,hi,Z,d,g))
    marker=checkpoint/'config.json'
    if marker.exists():assert json.loads(marker.read_text())==config
    else:
        temp=marker.with_suffix(f'.{os.getpid()}.tmp.json');temp.write_text(json.dumps(config,indent=2));temp.replace(marker)
    folds=list(GroupKFold(5).split(Z,groups=g));start=time.time()
    with threadpool_limits(limits=1):
        for m in (range(M) if indices is None else indices):
            for fold,(train,test) in enumerate(folds):
                if task_indices is not None and 5*m+fold not in task_indices:continue
                stem=checkpoint/f'm{m:03d}_f{fold:02d}';trace=stem.with_suffix('.zip');meta=stem.with_suffix('.json');reference=stem.with_suffix('.npz')
                if trace.exists() and meta.exists() and reference.exists():continue
                ztr,zte,med=ni.fill_train_medians(Z[train],Z[test]);zdt=np.column_stack([ztr,d[train]]);zde=np.column_stack([zte,d[test]])
                local_seed=seed+m*1000003+fold*10007
                vt,ve,audit=ni.impute_fold(lo[train],hi[train],lo[test],hi[test],zdt,zde,g[train],
                    n_sweeps=n_sweeps,rounds=rounds,seed=local_seed,trace_zip=trace)
                np.savez_compressed(reference,test_indices=test,completed_training=vt,completed_heldout=ve)
                main_cache=ni.OUT/'checkpoints/field_nested_M20_s3_r200'/f'm{m:03d}_fold{fold:02d}.npz'
                matching=None
                if main_cache.exists():
                    main=np.load(main_cache);p=len(data['compound_names'])
                    reconstructed=np.hstack([main['Vi'],np.column_stack([main['Vi'][:,j]+ni.LN10*main['y_'+name] for j,name in enumerate(data['compound_names'])])])
                    matching=bool(np.allclose(ve,reconstructed,rtol=0,atol=1e-10))
                    assert matching,'Trace refit differs from the corresponding main imputation chain'
                meta.write_text(json.dumps(dict(imputation=m,fold=fold,seed=local_seed,training_medians=med.tolist(),
                    training_plants=np.unique(g[train]).tolist(),heldout_plants=np.unique(g[test]).tolist(),
                    matches_main_chain=matching,elapsed_seconds=time.time()-start),indent=2))
                print(f'Trace {m+1}/{M} fold {fold+1}/5 elapsed {time.time()-start:.1f}s',flush=True)


def continue_sweeps(*,tag='masked_M5_s3_r200',final_sweeps=5,indices=None,task_indices=None):
    source=ni.OUT/'checkpoints'/tag;initial=json.loads((source/'config.json').read_text())
    if task_indices is not None and any(t<0 or t>=5*initial['M'] for t in task_indices):raise ValueError('Continuation task index outside M x 5 folds')
    assert final_sweeps>initial['n_sweeps']
    destination=ni.OUT/'checkpoints'/(tag+f'_S{final_sweeps}')
    destination.mkdir(parents=True,exist_ok=True)
    data,pr=ni.field_data();g=data['groups'];d=data['d'];Z=data['Z'];lo=data['Vlo'];hi=data['Vhi'];p=len(data['compound_names'])
    endpoints=dict(data['endpoints']);endpoints.update({name:[name] for name in data['compound_names']})
    folds=list(GroupKFold(5).split(Z,groups=g))
    config=json.loads((ni.OUT/'checkpoints/field_nested_M20_s3_r200/configuration.json').read_text())
    config.update(M=initial['M'],n_sweeps=final_sweeps)
    marker=destination/'configuration.json'
    if marker.exists():assert json.loads(marker.read_text())==config
    else:
        tmp=marker.with_suffix(f'.{os.getpid()}.tmp.json');tmp.write_text(json.dumps(config,indent=2));tmp.replace(marker)
    start=time.time()
    with threadpool_limits(limits=1):
        for m in (range(initial['M']) if indices is None else indices):
            for fold,(train,test) in enumerate(folds):
                if task_indices is not None and 5*m+fold not in task_indices:continue
                stem=source/f'm{m:03d}_f{fold:02d}';saved=np.load(stem.with_suffix('.npz'))
                meta=json.loads(stem.with_suffix('.json').read_text())
                cache=destination/f'm{m:03d}_fold{fold:02d}.npz';auditfile=cache.with_suffix('.json')
                if cache.exists() and auditfile.exists():continue
                with ZipFile(stem.with_suffix('.zip')) as archive:state=json.loads(archive.read('trace.json'))
                ztr,zte,med=ni.fill_train_medians(Z[train],Z[test]);zdt=np.column_stack([ztr,d[train]]);zde=np.column_stack([zte,d[test]])
                vt,ve,records=ni.impute_fold(lo[train],hi[train],lo[test],hi[test],zdt,zde,g[train],
                    n_sweeps=final_sweeps-initial['n_sweeps'],start_sweep=initial['n_sweeps'],rounds=initial['rounds'],seed=meta['seed'],
                    initial_train=saved['completed_training'],initial_test=saved['completed_heldout'],
                    training_rng_state=state['final_training_rng_state'],test_rng_state=state['final_test_rng_state'],
                    trace_zip=destination/f'm{m:03d}_fold{fold:02d}_continuation.zip')
                Wtr=np.column_stack([ztr,vt[:,:p]/ni.LN10]);Wte=np.column_stack([zte,ve[:,:p]/ni.LN10])
                yt=ni.endpoints_from_completed(vt,data['compound_names'],data['mol_weights'],endpoints)
                ye=ni.endpoints_from_completed(ve,data['compound_names'],data['mol_weights'],endpoints)
                output=dict(eh=ni.propensity(Wtr,Wte,d[train],light=False),Vi=ve[:,:p])
                for key in endpoints:
                    output['y_'+key]=ye[key]
                    output['mh_'+key]=np.mean([model.fit(Wtr,yt[key]).predict(Wte) for model in ni.regressors(light=False)],axis=0)
                tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,**output);tmp.replace(cache)
                audit=dict(imputation=m,fold=fold,training_plants=np.unique(g[train]).tolist(),heldout_plants=np.unique(g[test]).tolist(),
                    training_primary_plants=int(len(np.unique(g[train][d[train]==0]))),heldout_primary_plants=int(len(np.unique(g[test][d[test]==0]))),
                    covariate_medians=med.tolist(),conditional_models=state['conditional_models']+records,
                    continued_from_sweeps=initial['n_sweeps'],elapsed_seconds=time.time()-start)
                auditfile.write_text(json.dumps(audit,indent=2))
                print(f'Sweep continuation {m+1}/{initial["M"]} fold {fold+1}/5 elapsed {time.time()-start:.1f}s',flush=True)


def metric_table(frame,bootstrap=2000):
    rows=[];rng=np.random.default_rng(9841)
    groups=[('All','All',frame)]+[(side,cmp,s) for (side,cmp),s in frame.groupby(['side','compound'])]
    for side,cmp,f in groups:
        model=(f.predicted_median_log10-f.reference_log10).to_numpy()
        base=(f.DL2_log10-f.reference_log10).to_numpy()
        cover90=((f.reference_log10>=f.lo90_log10)&(f.reference_log10<=f.hi90_log10)).to_numpy()
        cover95=((f.reference_log10>=f.lo95_log10)&(f.reference_log10<=f.hi95_log10)).to_numpy()
        row=dict(side=side,compound=cmp,n=len(f),plants=f.WWTP_CODE.nunique(),
                 RMSE_model=float(np.sqrt(np.mean(model**2))),RMSE_DL2=float(np.sqrt(np.mean(base**2))),
                 MAE_model=float(np.mean(np.abs(model))),MAE_DL2=float(np.mean(np.abs(base))),
                 coverage90=float(cover90.mean()),coverage95=float(cover95.mean()),
                 width90=float(np.mean(f.hi90_log10-f.lo90_log10)),width95=float(np.mean(f.hi95_log10-f.lo95_log10)))
        row['RMSE_improvement']=row['RMSE_DL2']-row['RMSE_model']
        if row['plants']>=8:
            temp=pd.DataFrame(dict(g=f.WWTP_CODE.to_numpy(),m=model**2,b=base**2,c90=cover90.astype(float),c95=cover95.astype(float),n=1.,
                am=np.abs(model),ab=np.abs(base),w95=(f.hi95_log10-f.lo95_log10).to_numpy()))
            totals=temp.groupby('g')[['m','b','c90','c95','n','am','ab','w95']].sum().to_numpy();G=len(totals)
            sample=totals[rng.integers(0,G,(bootstrap,G))].sum(1)
            metrics={'RMSE_improvement':np.sqrt(sample[:,1]/sample[:,4])-np.sqrt(sample[:,0]/sample[:,4]),
                     'coverage90':sample[:,2]/sample[:,4],'coverage95':sample[:,3]/sample[:,4],
                     'MAE_model':sample[:,5]/sample[:,4],'MAE_DL2':sample[:,6]/sample[:,4],'width95':sample[:,7]/sample[:,4]}
            for name,values in metrics.items():row[name+'_lo'],row[name+'_hi']=np.quantile(values,[.025,.975]).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(*,tag='masked_M5_s3_r200',quantile=.2,predictive_draws=100,minimum_training_detections=5,masking='train_q20'):
    checkpoint=ni.OUT/'checkpoints'/tag;config=json.loads((checkpoint/'config.json').read_text())
    data,pr=ni.field_data();g=data['groups'];d=data['d'];Z=data['Z'];lo=data['Vlo'];hi=data['Vhi'];names=data['compound_names']
    assert config['data_hash']==ni.digest(lo,hi,Z,d,g)
    folds=list(GroupKFold(5).split(Z,groups=g));rows=[];threshold_rows=[];start=time.time();replay_checks=[]
    dl_upper=None;dl_audit=[]
    if masking=='reported_DL3':
        from expanded_benchmark import detection_limits
        default_source=ni.FROZEN if ni.FROZEN.exists() else Path(ni.__file__).resolve().parents[1]
        source=Path(os.environ.get('ECCC_PFAS_ANALYSIS_ROOT',str(default_source)))
        records=pd.read_parquet(source/'data_clean/env_long.parquet');records=records[~records.is_dup & records.usable]
        inf,ai=detection_limits(pr,'INF',records,compounds=names);eff,ae=detection_limits(pr,'EFF',records,compounds=names)
        dl_upper=np.hstack([inf,eff])+np.log(3.);dl_audit=ai+ae
    with threadpool_limits(limits=1):
        for fold,(train,test) in enumerate(folds):
            thresholds=np.full(lo.shape[1],np.nan)
            for j in range(lo.shape[1]):
                detected=lo[train,j][np.isfinite(lo[train,j])]
                if masking=='train_q20':
                    if len(detected)>=minimum_training_detections:thresholds[j]=np.quantile(detected,quantile)
                    threshold_rows.append(dict(fold=fold,side='INF' if j<len(names) else 'EFF',compound=names[j%len(names)],
                        training_detections=len(detected),quantile=quantile,threshold_kind=masking,threshold_ln=thresholds[j]))
                else:
                    threshold_rows.append(dict(fold=fold,side='INF' if j<len(names) else 'EFF',compound=names[j%len(names)],
                        training_detections=len(detected),quantile=None,threshold_kind=masking,threshold_ln=float(np.median(dl_upper[test,j])),
                        threshold_ln_min=float(dl_upper[test,j].min()),threshold_ln_max=float(dl_upper[test,j].max())))
            upper=np.broadcast_to(thresholds,hi[test].shape) if dl_upper is None else dl_upper[test]
            mask=np.isfinite(lo[test])&(lo[test]<=upper)
            masked_lo=lo[test].copy();masked_hi=hi[test].copy();masked_lo[mask]=-np.inf
            masked_hi=np.where(mask,upper,masked_hi)
            _,zte,med=ni.fill_train_medians(Z[train],Z[test]);zde=np.column_stack([zte,d[test]])
            chains=[]
            for m in range(config['M']):
                stem=checkpoint/f'm{m:03d}_f{fold:02d}';trace=stem.with_suffix('.zip');meta=json.loads(stem.with_suffix('.json').read_text())
                assert np.array_equal(np.array(meta['training_medians']),med)
                # One unmasked replay must exactly recover the saved held-out chain.
                check=ni.replay_imputation_trace(lo[test],hi[test],zde,trace,predictive_draws=1,seed=meta['seed'])[0]
                expected=np.load(stem.with_suffix('.npz'))['completed_heldout'];assert np.array_equal(check,expected)
                replay_checks.append(dict(imputation=m,fold=fold,exact=True))
                draws=ni.replay_imputation_trace(masked_lo,masked_hi,zde,trace,predictive_draws=predictive_draws,seed=meta['seed']+70000001)
                chains.append(draws)
            samples=np.concatenate(chains,axis=0)
            q=np.quantile(samples,[.025,.05,.5,.95,.975],axis=0)/ni.LN10
            for local,j in zip(*np.where(mask)):
                original=test[local];truth=lo[original,j]/ni.LN10
                rows.append(dict(row=int(original),fold=fold,WWTP_CODE=str(g[original]),treatment=float(d[original]),
                    side='INF' if j<len(names) else 'EFF',compound=names[j%len(names)],reference_log10=float(truth),
                    upper_log10=float(masked_hi[local,j]/ni.LN10),DL2_log10=float((masked_hi[local,j]-np.log(2.))/ni.LN10),
                    predicted_median_log10=float(q[2,local,j]),lo90_log10=float(q[1,local,j]),hi90_log10=float(q[3,local,j]),
                    lo95_log10=float(q[0,local,j]),hi95_log10=float(q[4,local,j]),
                    empirical_CDF=float(np.mean(samples[:,local,j]<=lo[original,j]))))
            print(f'Masked replay fold {fold+1}/5 elapsed {time.time()-start:.1f}s',flush=True)
    frame=pd.DataFrame(rows);output_tag=tag if masking=='train_q20' else tag+'_reportedDL3';prefix=ni.OUT/'results'/output_tag
    prefix.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(prefix.parent/(output_tag+'_reconstruction.csv'),index=False)
    pd.DataFrame(threshold_rows).to_csv(prefix.parent/(output_tag+'_thresholds.csv'),index=False)
    metrics=metric_table(frame);metrics.to_csv(prefix.parent/(output_tag+'_metrics.csv'),index=False)
    report=dict(training_chains=config['M'],predictive_draws_per_chain=predictive_draws,masking=masking,quantile=quantile if masking=='train_q20' else None,
        minimum_training_detections=minimum_training_detections,masked_measurements=len(frame),
        masked_plants=int(frame.WWTP_CODE.nunique()),elapsed_seconds=time.time()-start,
        exact_unmasked_replays=replay_checks,summary=metrics.iloc[0].to_dict(),
        scope='Detected-reference reconstruction in the detection-selected held-out subset; not identified calibration for true non-detects.',
        interval_scope='Cluster-bootstrap intervals condition on the fitted training chains and their predictions; no refitting occurs in that bootstrap.')
    if dl_audit:report['reported_DL_recovery_audit']=dl_audit
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','analyze','continue']);p.add_argument('--tag',default='masked_M5_s3_r200')
    p.add_argument('--M',type=int,default=5);p.add_argument('--sweeps',type=int,default=3);p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--seed',type=int,default=42000,help='Training-chain root seed for fit; later stages read the recorded seed.')
    p.add_argument('--m-indices',nargs='+',type=int);p.add_argument('--draws',type=int,default=100)
    p.add_argument('--task-indices',nargs='+',type=int)
    p.add_argument('--final-sweeps',type=int,default=5)
    p.add_argument('--masking',choices=['train_q20','reported_DL3'],default='train_q20')
    args=p.parse_args()
    if args.m_indices is not None and args.task_indices is not None:p.error('Chooseimputationindicesortaskindices,notboth')
    if args.mode=='fit':fit_traces(M=args.M,n_sweeps=args.sweeps,rounds=args.rounds,seed=args.seed,tag=args.tag,indices=args.m_indices,task_indices=args.task_indices)
    elif args.mode=='analyze':analyze(tag=args.tag,predictive_draws=args.draws,masking=args.masking)
    else:continue_sweeps(tag=args.tag,final_sweeps=args.final_sweeps,indices=args.m_indices,task_indices=args.task_indices)


if __name__=='__main__':main()
