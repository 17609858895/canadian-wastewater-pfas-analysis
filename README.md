# Canadian wastewater PFAS analysis

Core data-processing and statistical-analysis code for **Aqueous PFCA enrichment and censoring-sensitive precursor indices across Canadian wastewater plants**.

The analysis compares paired influent and effluent measurements, examines treatment stages and biosolids, and evaluates the sensitivity of targeted precursor indices to analytical censoring. It implements conditional AFT multiple imputation, plant-grouped nuisance fitting, semi-synthetic benchmarks, sensitivity analyses and complementary biosolids oxidation calculations. The observed process groups are not observed same-plant upgrades.

## Inputs and environment

Use Python 3.12 and the versions in `requirements.txt`. Install with `python -m pip install -r requirements.txt`.

Obtain the public monitoring data from Environment and Climate Change Canada:

- [PFAS in Canadian municipal wastewater treatment systems](https://open.canada.ca/data/en/dataset/f4e25ea2-e070-448c-9105-129807f5fc3a).
- The associated ECCC Chemicals Management Plan wastewater monitoring directory supplies the plant-code and temperature/flow tables and the biosolids TOP release.

Place these five files in a local `raw/` directory:

1. `PFASWastewater_EN_FR.csv`
2. `PFASBiosolids_EN_FR.csv`
3. `CMP_Master_WWTP_Code_List_EN_FR.csv`
4. `CMP_Master_Temp_Flows_EN_FR.csv`
5. `PFASTOPBiosolids_EN_FR.csv`

The two concentration releases were generated on 29 April 2025; the TOP release was generated on 13 May 2025. The downloaded plant and flow tables are identified by their hashes in `input_sha256.json`. Raw data retain their original metadata and embedded dictionaries; the parser reads their CP1252 encoding. ECCC data are provided under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada). This repository does not imply government endorsement.

Raw data, processed data, fitted outputs, manuscripts, and artwork are excluded from Git. Create `data_clean/` and `results/` before running analyses.

## Execution

Run commands from this repository root. Scripts are research analysis entry points, not import-safe library modules; several execute on import. Full fitting and permutations can take hours. The benchmark uses 12 worker processes. Keep raw inputs and outputs in separate directories as specified above.

```sh
python scripts/s01_load_clean.py
python scripts/s02_build_wide.py
python scripts/s03_upgrade_effect.py 10 0
python scripts/s04_semisynthetic_benchmark.py 16
python scripts/s05_process_stage_biosolids.py
python scripts/s05b_era_effects.py
python scripts/s06_precursor_budget.py
python scripts/s07_upgrade_projection.py
python scripts/s08_precursor_budget_lod.py
python scripts/s09_prediction_leave_plant_out.py
python scripts/s10_robustness.py
python scripts/s11_studentized_permutation.py
python scripts/s12_stage_plant_level.py
python scripts/s13_si_analyses.py flags
python scripts/s13_si_analyses.py lag
python scripts/s13_si_analyses.py extended
python scripts/s13_si_analyses.py balance
python scripts/s13_si_analyses.py imputed
python scripts/s13_si_analyses.py rv
python scripts/s14_region_loads.py
python scripts/s15_top_validation.py
python scripts/s16_matched_panel_precursor.py
```

The second argument `0` in the main comparison skips its earlier auxiliary permutation calculation; the studentized plant-label comparison is calculated by `s11_studentized_permutation.py`. Main/stage/biosolids analyses use ten imputations. The plant summaries in the stage figure use five imputations, as specified in the Supporting Information. Default values and seeds are explicit in each script.

## Interpretation and reproducibility checks

The principal comparison uses 373 pairs from 54 plants. Expanded-panel precursor indices are computed separately by treatment group; reported aqueous indices concern 26 biological plants within the 31-plant expanded population. The biosolids oxidation index concerns 11 biological plants and has a different denominator. Precursor indices are screening ratios, not conversion yields or closed mass balances. Annualized sampled-day loads are conditional summaries, not national annual inventories.

The matched-panel analysis holds the expanded population fixed at 153 pairs from 31 plants (132 pairs from 26 biological plants). With the full 26-compound panel, the biological-plant precursor index spans approximately 15–135% across ND = 0, DL/2 and DL. Restricting all three substitutions to the same 21 compounds used by the detection-supported imputation narrows this range to approximately 14–70%. The approximately 18% imputed estimate uses that 21-compound panel. Comparing 15–135% directly with 18% combines changes in censoring treatment and chemical coverage; the matched-panel comparison distinguishes these contributions.

`s16_matched_panel_precursor.py` uses only `data_clean/pairs_inf_eff.parquet` and `data_clean/plants.csv`; it does not refit imputation or DML. It writes the 12-row table to `results/s16_matched_panel.csv`, with plant changes and detection-support metadata alongside it. If the earlier `results/s06_precursor_budget.csv` exists, it also verifies the full-panel results against that calculation. An alternative destination can be supplied with `--output /path/to/table.csv`, and `--root /path/to/repository` selects another input repository.

`python -m unittest discover -s tests -v` checks numerical primitives and parsing invariants. The public-code preparation verified syntax for every script, ran these tests and reproduced the cleaning/pairing outputs against the original analysis. The full stochastic model suite was not refitted during packaging. Small numerical differences can arise across software/platform versions. The Supporting Information documents design limitations, benchmark scope, model settings and interval definitions.

The publication package retains the original analysis and plotting sources separately. This repository deliberately contains only the core analysis code and the input specification.
