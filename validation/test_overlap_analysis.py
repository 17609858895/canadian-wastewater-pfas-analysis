"""Focused checks of information boundary, weighting identities and aggregation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import overlap_analysis as a

def run():
    pr=a.load_pre_design();pp,wp,_,_,_=a.design(pr);unit,cells=a.region_era_weights(pr);pw=a.pair_weights(pr,wp,unit)
    perturbed=pr.copy()
    for c in perturbed:
        if c.startswith('EFF_') and pd.api.types.is_numeric_dtype(perturbed[c]):perturbed[c]=perturbed[c]*10000
    pp2,wp2,_,_,_=a.design(perturbed);u2,c2=a.region_era_weights(perturbed)
    np.testing.assert_array_equal(wp.select_dtypes('number'),wp2.select_dtypes('number'))
    np.testing.assert_array_equal(unit.weight_region_era,u2.weight_region_era)
    checks={'effluent_perturbation_design_invariance':True}
    for method in ('equal_plant','overlap_C0.1','overlap_C1','overlap_C10','region_era_standardized'):
        assert np.isfinite(pw['weight_'+method]).all() and (pw['weight_'+method]>=0).all()
        values=np.tile(np.where(pr.bio.eq(1),2.,0.),(3,1))
        result,_=a.pooled_weighted_summary(pr,pw,values,method,bootstrap=100)
        np.testing.assert_allclose(result['theta'],2.,atol=1e-12)
        np.testing.assert_allclose(result['se'],0.,atol=1e-12)
    checks['constant_group_contrast_weighting_identity']=True
    den=pw.groupby('plant').weight_equal_plant.sum()
    np.testing.assert_allclose(den,1.,atol=1e-12)
    checks['each_plant_has_unit_equal_weight']=True
    for r in cells[cells.supported].itertuples():
        q=unit[(unit.receiving_region==r.receiving_region)&(unit.era==r.era)]
        np.testing.assert_allclose(q.loc[q.d.eq(0),'weight_region_era'].sum(),q.loc[q.d.eq(1),'weight_region_era'].sum(),atol=1e-12)
    checks['region_era_joint_masses_exact']=True
    np.testing.assert_allclose(a.ess(np.array([1.,2.,3.])),a.ess(np.array([10.,20.,30.])))
    checks['ESS_scale_invariance']=True
    # Removing the only biological unit in an artificial primary target cell must fail,
    # rather than silently redefine the primary target.
    toy=pr.iloc[:4].copy();toy['WWTP_CODE']=['p1','p2','b1','b2'];toy['receiving_region']=['R1','R2','R1','R1'];toy['era']='E';toy['bio']=[0.,0.,1.,1.]
    try:a.region_era_weights(toy)
    except ValueError:checks['unsupported_primary_context_rejected']=True
    else:raise AssertionError('Unsupported primary target silently excluded')
    out=Path(__file__).resolve().parent/'results/overlap/overlap_tests.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks))

if __name__=='__main__':run()
