"""Executable information-boundary and truncation checks on a small design."""
from pathlib import Path
import json,time
from zipfile import ZipFile
import numpy as np
from threadpoolctl import threadpool_limits
from nested_imputation import impute_fold, draw_upper, fill_train_medians, replay_imputation_trace, OUT

def run():
    started=time.time();rng=np.random.default_rng(91)
    ntr,nte,p=18,9,4
    latent=rng.normal(size=(ntr+nte,p));censor=latent<-.1
    censor[:,1]=False  # Test-only censoring must not change the training chain.
    lo=np.where(censor,-np.inf,latent);hi=np.where(censor,-.1,latent)
    z=rng.normal(size=(ntr+nte,3));groups=np.repeat(np.arange(6),3)
    z[2,0]=np.nan;z[ntr+1,0]=np.nan
    ztr,zte,median=fill_train_medians(z[:ntr],z[ntr:])
    trace=OUT/'checkpoints/boundary_test_models.zip';trace.parent.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        a,b,qa=impute_fold(lo[:ntr],hi[:ntr],lo[ntr:],hi[ntr:],ztr,zte,groups,n_sweeps=3,rounds=8,seed=87,store_model_hashes=True,trace_zip=trace)
        changed_lo=lo[ntr:].copy();changed_hi=hi[ntr:].copy();changed_z=zte.copy()
        changed_lo[:,0]=-np.inf;changed_hi[:,0]=-4
        changed_lo[:,1]=-np.inf;changed_hi[:,1]=-2
        changed_z[:,0]+=5
        aa,bb,qb=impute_fold(lo[:ntr],hi[:ntr],changed_lo,changed_hi,ztr,changed_z,groups,n_sweeps=3,rounds=8,seed=87,store_model_hashes=True)
    assert np.array_equal(a,aa),'Held-out changes affected completed training values'
    assert [{k:v for k,v in row.items() if k!='trace_member'} for row in qa]==qb,'Held-out changes affected fitted model/scale/bootstrap records'
    replay=replay_imputation_trace(lo[ntr:],hi[ntr:],zte,trace,predictive_draws=1,seed=87)[0]
    assert np.array_equal(b,replay),'Captured model replay did not reproduce held-out imputation'
    with ZipFile(trace) as archive:state=json.loads(archive.read('trace.json'))
    with threadpool_limits(limits=1):
        c,e,qcont=impute_fold(lo[:ntr],hi[:ntr],lo[ntr:],hi[ntr:],ztr,zte,groups,n_sweeps=2,start_sweep=3,rounds=8,seed=87,
            initial_train=a,initial_test=b,training_rng_state=state['final_training_rng_state'],test_rng_state=state['final_test_rng_state'],store_model_hashes=True)
        cc,ee,qfull=impute_fold(lo[:ntr],hi[:ntr],lo[ntr:],hi[ntr:],ztr,zte,groups,n_sweeps=5,rounds=8,seed=87,store_model_hashes=True)
    assert np.array_equal(c,cc) and np.array_equal(e,ee),'Continuation differs from fresh five-sweep run'
    assert [{k:v for k,v in row.items() if k!='trace_member'} for row in qa]+qcont==qfull,'Continuation model hashes differ'
    extreme=draw_upper(np.array([100.,1000.,0.]),.2,np.array([-100.,-1000.,-30.]),rng)
    assert np.isfinite(extreme).all() and np.all(extreme<=np.array([-100.,-1000.,-30.]))
    _,_,changed_median=fill_train_medians(z[:ntr],changed_z)
    assert np.array_equal(median,changed_median)
    result=dict(status='PASS',training_values_invariant=True,all_conditional_model_hashes_invariant=True,
                training_sigma_and_bootstrap_invariant=True,training_medians_invariant=True,
                heldout_values_may_change=True,extreme_truncated_draws_below_limits=True,
                fixed_training_model_replay_exact=True,
                continued_S3_to_S5_values_and_model_hashes_exact=True,
                fitted_model_count_per_run=len(qa),elapsed_seconds=time.time()-started)
    out=OUT/'results/nested_information_boundary_check.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
