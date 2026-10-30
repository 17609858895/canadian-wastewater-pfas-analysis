"""Compare fixed non-detect substitutions at matched chemical coverage.

No model is fitted. Inputs are the cleaned ECCC pairing and plant-description
tables. Chemical support is screened once in the complete expanded population,
matching the screening used by the precursor-imputation analysis. Group changes
are summed per pair, summarized as plant medians, then clipped by sign.

Example: python scripts/s16_matched_panel_precursor.py --root /path/to/repository
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMMON = ['PFBA','PFPeA','PFHxA','PFHpA','PFOA','PFNA','PFDA','PFBS','PFHxS','PFOS','PFOSA']
PFAA = ['PFBA','PFPeA','PFHxA','PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA',
        'PFBS','PFPeS','PFHxS','PFHpS','PFOS','PFNS','PFDS']
PREC = ['4:2 FTS','6:2 FTS','8:2 FTS','EtFOSAA','MeFOSAA','N-EtFOSE','N-MeFOSE',
        'N-EtFOSA','N-MeFOSA','PFOSA']
MW = dict(PFBA=214.04,PFPeA=264.05,PFHxA=314.05,PFHpA=364.06,PFOA=414.07,
          PFNA=464.08,PFDA=514.08,PFUnA=564.09,PFDoA=614.1,PFBS=300.1,
          PFPeS=350.1,PFHxS=400.11,PFHpS=450.12,PFOS=500.13,PFNS=550.14,
          PFDS=600.14,**{'4:2 FTS':328.1,'6:2 FTS':428.17,'8:2 FTS':528.18,
          'EtFOSAA':585.24,'MeFOSAA':571.21,'N-EtFOSE':571.25,'N-MeFOSE':557.23,
          'N-EtFOSA':527.2,'N-MeFOSA':513.17,'PFOSA':499.14})


def load_population(clean: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(clean/'pairs_inf_eff.parquet')
    plants = pd.read_csv(clean/'plants.csv')
    plants['code2'] = plants.code.str.replace(r' \(.*\)', '', regex=True)
    plants = plants.drop_duplicates('code2').set_index('code2')
    # Reproduce the common-panel inclusion and treatment mapping in load_design.
    fields = [f'{s}_hi|{c}' for s in ['INF','EFF'] for c in COMMON]
    pairs = pairs[pairs[fields].notna().all(axis=1)].copy()
    primary = plants.liquid.astype(str).str.lower().str.startswith('primary')
    pairs['biological'] = ~pairs.WWTP_CODE.map(primary).fillna(False).astype(bool)
    pairs = pairs[pairs.date.dt.year>=2018].copy()
    complete = [f'{s}_hi|{c}' for s in ['INF','EFF'] for c in PREC+PFAA]
    return pairs[pairs[complete].notna().all(axis=1)].reset_index(drop=True)


def concentration(pairs, side, compound, mode):
    lo = pairs[f'{side}_lo|{compound}'].to_numpy(float)
    hi = pairs[f'{side}_hi|{compound}'].to_numpy(float)
    if mode=='0': return lo
    if mode=='DL': return hi
    return np.where(lo==0,hi/2,lo)


def compute(root: Path):
    source = root
    pairs = load_population(source/'data_clean')
    detection_counts = {c:int(sum((pairs[f'{s}_lo|{c}']>0).sum() for s in ['INF','EFF'])) for c in PREC+PFAA}
    supported = [c for c in PREC+PFAA if detection_counts[c]>=5]
    rows, plant_rows = [], []
    for name, panel in [('Full panel', PREC+PFAA),('Detection-supported panel',supported)]:
        acids = [c for c in PFAA if c in panel]
        precursors = [c for c in PREC if c in panel]
        for mode in ['0','DL/2','DL']:
            delta = {}
            for label, compounds in [('PFAA',acids),('Prec',precursors)]:
                delta[label] = sum((concentration(pairs,'EFF',c,mode)-concentration(pairs,'INF',c,mode))/MW[c] for c in compounds)
            d = pairs[['WWTP_CODE','biological']].copy()
            d['dPFAA'],d['dPrec'] = delta['PFAA'],delta['Prec']
            for flag, population in [(True,'Biological plants'),(False,'Primary-only plants')]:
                pop = d[d.biological==flag]
                pm = pop.groupby('WWTP_CODE')[['dPFAA','dPrec']].median()
                numerator = float((-pm.dPrec.clip(upper=0)).sum())
                denominator = float(pm.dPFAA.clip(lower=0).sum())
                rows.append(dict(Population=population,Panel=name,Compounds=len(panel),
                    Substitution=mode,Pairs=len(pop),Plants=len(pm),
                    Precursor_loss_sum_nmol_L=numerator,PFAA_gain_sum_nmol_L=denominator,
                    Index_percent=100*numerator/denominator if denominator>0 else np.nan))
                for plant,row in pm.iterrows():
                    plant_rows.append(dict(Population=population,Panel=name,Substitution=mode,
                                          Plant=plant,dPFAA_nmol_L=float(row.dPFAA),dPrec_nmol_L=float(row.dPrec)))
    table = pd.DataFrame(rows)
    original_file = source/'results/s06_precursor_budget.csv'
    original_checked = original_file.exists()
    if original_checked:
        original = pd.read_csv(original_file)
        for mode, key in [('0','lo'),('DL/2','mid'),('DL','hi')]:
            for population, group in [('Biological plants','biological'),('Primary-only plants','primary')]:
                actual = table[(table.Population==population)&(table.Panel=='Full panel')&(table.Substitution==mode)].iloc[0]
                expected = original[(original.group==group)&(original['mode']==key)].iloc[0]
                assert actual.Plants==expected.plants
                assert np.isclose(actual.Index_percent,100*expected.frac_explained,rtol=1e-10,atol=1e-10)
    assert len(supported)==21
    assert pairs.WWTP_CODE.nunique()==31
    assert pairs[pairs.biological].WWTP_CODE.nunique()==26
    info = dict(total_pairs=len(pairs),total_plants=pairs.WWTP_CODE.nunique(),
                biological_pairs=int(pairs.biological.sum()),biological_plants=26,
                full_panel=PREC+PFAA,supported_panel=supported,
                excluded_compounds=[c for c in PREC+PFAA if c not in supported],
                detection_counts=detection_counts,
                original_comparison_checked=original_checked,
                full_panel_reproduces_original=True if original_checked else None,
                table=table.to_dict(orient='records'))
    return table,pd.DataFrame(plant_rows),info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],
                        help='Repository directory containing data_clean/ and results/.')
    parser.add_argument('--output',type=Path,
                        help='Output CSV path; default: ROOT/results/s16_matched_panel.csv.')
    args=parser.parse_args();root=args.root.resolve()
    table,plants,info=compute(root)
    output=(args.output.resolve() if args.output else root/'results/s16_matched_panel.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(output,index=False,float_format='%.10g')
    plants.to_csv(output.with_name(output.stem+'_plant_changes.csv'),index=False)
    output.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    biological=table[table.Population=='Biological plants']
    print(biological.to_string(index=False))
    print('Excluded:',info['excluded_compounds'])
    print('Output:',output)

if __name__=='__main__':main()
