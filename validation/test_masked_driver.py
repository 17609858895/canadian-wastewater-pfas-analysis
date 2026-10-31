"""Small synthetic end-to-end check of trace, masking and continuation drivers."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import nested_imputation as ni
import masked_detection_validation as md


def run():
    started=time.time();rng=np.random.default_rng(2701);groups=np.repeat(np.arange(9).astype(str),3)
    d=(np.repeat(np.arange(9),3)>=3).astype(float);Z=rng.normal(size=(27,2));Z[3,0]=np.nan
    latent=rng.normal(.2,.7,(27,4));latent[:,2:]=latent[:,:2]+.2*d[:,None]+rng.normal(0,.25,(27,2))
    lo=np.where(latent<-.2,-np.inf,latent);hi=np.where(latent<-.2,-.2,latent)
    data=dict(Vlo=lo,Vhi=hi,Z=Z,d=d,groups=groups,compound_names=['A','B'],mol_weights={'A':1.,'B':2.},endpoints={'sumPFCA':['A','B']})
    saved_out,saved_loader=ni.OUT,ni.field_data;test_out=saved_out/'checkpoints/masked_driver_smoke'
    test_out.mkdir(parents=True,exist_ok=True)
    ni.OUT=test_out;ni.field_data=lambda:(data,None)
    try:
        md.fit_traces(M=2,n_sweeps=1,rounds=8,tag='toy')
        md.analyze(tag='toy',predictive_draws=100)
        report=json.loads((test_out/'results/toy.json').read_text())
        assert report['masked_measurements']>0 and all(r['exact'] for r in report['exact_unmasked_replays'])
        base=test_out/'checkpoints/field_nested_M20_s3_r200';base.mkdir(parents=True,exist_ok=True)
        (base/'configuration.json').write_text(json.dumps(dict(M=20,n_sweeps=1,rounds=8,seed=42000,data_hash=ni.digest(lo,hi,Z,d,groups),endpoints={'sumPFCA':['A','B'],'A':['A'],'B':['B']})))
        md.continue_sweeps(tag='toy',final_sweeps=3)
        final=test_out/'checkpoints/toy_S3';assert len(list(final.glob('m*_fold*.npz')))==10
        assert all(key in np.load(next(final.glob('m*_fold*.npz'))).files for key in ['eh','Vi','y_A','mh_A','y_B','mh_B'])
        result=dict(status='PASS',toy_training_chains=2,outer_folds=5,predictive_draws_per_chain=100,
                    detected_references_masked=report['masked_measurements'],all_unmasked_replays_exact=True,
                    trace_fit_mask_analyze_and_continuation_drivers_work=True,all_continuation_nuisance_caches_complete=True,
                    elapsed_seconds=time.time()-started)
        saved_out.joinpath('results').mkdir(parents=True,exist_ok=True)
        saved_out.joinpath('results/masked_driver_smoke_test.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:ni.OUT=saved_out;ni.field_data=saved_loader


if __name__=='__main__':run()
