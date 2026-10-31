"""Propagate the nested primary analysis to descriptive loads and scenarios.

Stitched held-out imputations are used only for descriptive concentrations.
They are never used to refit a treatment-effect estimator.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import t, norm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
OUT = ROOT / 'validation/results/loads'
sys.path.insert(0, str(SOURCE / 'scripts'))
import s03_upgrade_effect as s3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = OUT.parent
    data = np.load(res / 'field_nested_M20_s3_r200_completed.npz')
    print('completed keys', data.files)
    key = 'completed' if 'completed' in data.files else 'imps'
    if key not in data.files:
        key = next(k for k in data.files if data[k].shape == (20, 373, 22))
    values = data[key]
    assert values.shape == (20, 373, 22)
    pr = s3.load_design().copy()
    ks = [s3.CMP.index(c) for c in s3.PFCA]
    pr['eff_ngL'] = np.exp(values[:, :, 11:][:, :, ks]).sum(2).mean(0)
    pr['inf_ngL'] = np.exp(values[:, :, :11][:, :, ks]).sum(2).mean(0)
    plants = pd.read_csv(SOURCE / 'data_clean/plants.csv')
    plants['code2'] = plants.code.str.replace(r' \(.*\)', '', regex=True)
    plants = plants.drop_duplicates('code2').set_index('code2')
    flow = pd.to_numeric(plants.flow_avg.str.replace(',', ''), errors='coerce')
    pr['rec'] = pr.WWTP_CODE.map(plants.receiving.str.strip())
    pr['flow_use'] = pr.flow.fillna(pr.WWTP_CODE.map(flow))
    pr['load_g_d'] = pr.eff_ngL * pr.flow_use * 1e-6
    pr['formed_g_d'] = (pr.eff_ngL - pr.inf_ngL) * pr.flow_use * 1e-6
    plant = pr.groupby('WWTP_CODE').agg(rec=('rec', 'first'), proc=('proc', 'first'),
        pairs=('bio', 'size'), load_g_yr=('load_g_d', lambda x: np.nanmedian(x) * 365),
        formed_g_yr=('formed_g_d', lambda x: np.nanmedian(x) * 365)).reset_index()
    effect = pd.read_csv(res / 'field_nested_M20_s3_r200.csv')
    pfca = effect.set_index('outcome').loc['sumPFCA']
    process = pd.read_csv(SOURCE / 'results/s05_process.csv')
    settings = {'biological (pooled)': (pfca.theta, pfca.se, pfca.BR53_df_BR)}
    settings.update({r.proc: (r.theta, r.se, np.inf)
                     for r in process[process.outcome == 'sumPFCA'].itertuples()})
    # Shared transported contrast within a scenario: regional sums then have
    # exact conditional quantiles, not sums of unrelated Monte Carlo endpoints.
    probs = np.array([.025, .5, .975])
    rows = []
    for _, p in plant[plant.proc == 'primary'].dropna(subset=['load_g_yr']).iterrows():
        for name, (theta, se, df) in settings.items():
            quant = norm.ppf(probs) if np.isinf(df) else t.ppf(probs, df)
            multipliers = 10 ** (theta + se * quant)
            added = p.load_g_yr * (multipliers - 1)
            rows.append(dict(level='plant', plant=p.WWTP_CODE, scenario=name,
                current_g_yr=p.load_g_yr, added_g_yr=added[1], added_lo=added[0], added_hi=added[2],
                fold=multipliers[1], theta=theta, se=se, df=df))
    projection = pd.DataFrame(rows)
    projection.to_csv(OUT / 's07_projection.csv', index=False)
    pooled = projection[projection.scenario == 'biological (pooled)']
    plant = plant.merge(pooled[['plant', 'added_g_yr', 'added_lo', 'added_hi']],
                        left_on='WWTP_CODE', right_on='plant', how='left')
    plant.to_csv(OUT / 's14_plant_loads.csv', index=False)
    rows = []
    for rec, g in plant.groupby('rec'):
        row = dict(rec=rec, plants=len(g), pairs=int(g.pairs.sum()),
                   plants_with_load=int(g.load_g_yr.notna().sum()),
                   load_kg_yr=g.load_g_yr.sum()/1000, formed_kg_yr=g.formed_g_yr.sum()/1000,
                   added_kg_yr=g.added_g_yr.sum()/1000, added_lo=g.added_lo.sum()/1000,
                   added_hi=g.added_hi.sum()/1000, primary_plants=int((g.proc == 'primary').sum()))
        row.update({f'n_{p}': int((g.proc == p).sum()) for p in ['primary','biofilm','AS','BNR','lagoon','ext_aer']})
        rows.append(row)
    regions = pd.DataFrame(rows)
    regions.to_csv(OUT / 's14_region_summary.csv', index=False)
    effect['method'] = 'LOD-DML'
    effect['lo95'] = effect.BR53_lo95
    effect['hi95'] = effect.BR53_hi95
    effect.to_csv(OUT / 's03_effects.csv', index=False)
    # Curves use deterministic distribution quantiles rather than simulation.
    curves = []
    b = 10 ** pfca.theta
    for name, (theta, se, df) in settings.items():
        quant = norm.ppf(probs) if np.isinf(df) else t.ppf(probs, df)
        mult = 10 ** (theta + se * quant)
        for f in np.linspace(.05, .4, 36):
            inc = 100*f*(mult-1)/(f+(1-f)*b)
            curves.append(dict(scenario=name, primary_flow_share=f,
                               lo=inc[0], median=inc[1], hi=inc[2]))
    pd.DataFrame(curves).to_csv(OUT / 'system_curves.csv', index=False)
    summary = dict(load_kg_yr=float(regions.load_kg_yr.sum()),
                   aqueous_excess_kg_yr=float(regions.formed_kg_yr.sum()),
                   conditional_increment_kg_yr=float(pooled.added_g_yr.sum()/1000),
                   conditional_increment_lo_kg_yr=float(pooled.added_lo.sum()/1000),
                   conditional_increment_hi_kg_yr=float(pooled.added_hi.sum()/1000),
                   source='20 nested held-out completed datasets for descriptive sums; primary BR53 contrast',
                   G=54, primary_G=6, plants_with_load=52, pooled_df=float(pfca.BR53_df_BR))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), 'utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
