# Canadian wastewater PFAS analysis

Core data-processing and statistical-analysis code for **Measurement bounds distinguish PFCA enrichment from precursor attribution across Canadian wastewater plants**.

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

Run commands from this repository root. Scripts are research analysis entry points, not import-safe library modules; several execute on import. Full fitting and permutations can take hours. The historical benchmark uses 12 worker processes; the new validation benchmark has an explicit worker argument. Keep raw inputs and outputs in separate directories as specified above.

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

The second argument `0` in the main comparison skips its earlier auxiliary permutation calculation; the studentized plant-label comparison is calculated by `s11_studentized_permutation.py`. Historical full-population main/stage/biosolids analyses use ten imputations. The revised primary analysis below uses twenty nested imputations. Historical imputed plant-stage summaries use five imputations; the revised stage figure uses concentration-compatible bounds. Default values and seeds are explicit in each script.

## Interpretation and reproducibility checks

The principal comparison uses 373 pairs from 54 plants. Expanded-panel precursor indices are computed separately by treatment group; reported aqueous indices concern 26 biological plants within the 31-plant expanded population. The biosolids oxidation index concerns 11 biological plants and has a different denominator. Precursor indices are screening ratios, not conversion yields or closed mass balances. Annualized sampled-day loads are conditional summaries, not national annual inventories.

The matched-panel analysis holds the expanded population fixed at 153 pairs from 31 plants (132 pairs from 26 biological plants). With the full 26-compound panel, the biological-plant precursor index spans approximately 15–135% across ND = 0, DL/2 and DL. Restricting all three substitutions to the same 21 compounds used by the detection-supported imputation narrows this range to approximately 14–70%. The approximately 18% imputed estimate uses that 21-compound panel. Comparing 15–135% directly with 18% combines changes in censoring treatment and chemical coverage; the matched-panel comparison distinguishes these contributions.

`s16_matched_panel_precursor.py` uses only `data_clean/pairs_inf_eff.parquet` and `data_clean/plants.csv`; it does not refit imputation or DML. It writes the 12-row table to `results/s16_matched_panel.csv`, with plant changes and detection-support metadata alongside it. If the earlier `results/s06_precursor_budget.csv` exists, it also verifies the full-panel results against that calculation. An alternative destination can be supplied with `--output /path/to/table.csv`, and `--root /path/to/repository` selects another input repository.

`python -m unittest discover -s tests -v` checks numerical primitives and parsing invariants. The public-code preparation verified syntax for every script, ran these tests and reproduced the cleaning/pairing outputs against the original analysis. The full stochastic model suite was not refitted during packaging. Small numerical differences can arise across software/platform versions. The Supporting Information documents design limitations, benchmark scope, model settings and interval definitions.

The publication package retains the original analysis and plotting sources separately. This repository deliberately contains only the core analysis code and the input specification.

## Strengthened primary analysis and validation

The primary comparison now fits concentration imputation **inside each outer
training-plant fold**. Model fits, training covariate medians, residual scales,
plant bootstrap samples and training random draws exclude held-out plants.
The thirteen endpoint estimates use twenty chains, three sweeps, 200 boosting
rounds and five outer plant folds. Barnard–Rubin intervals use the observed
within/between-imputation components with complete-data df = 53; df = 5 is an
additional reference-limited sensitivity, not a calibration guarantee.

Use the repository root as the working directory. Run the cleaning and pairing
steps `s01_load_clean.py` and `s02_build_wide.py` first. The historical
`s03_upgrade_effect.py 10 0` also creates the full-population completed data
needed for secondary comparisons and the semi-synthetic design; those completed
data are not an input to the nested primary fit. Run
`s16_matched_panel_precursor.py` before the independent measurement-bound check,
and `s05_process_stage_biosolids.py` before the load-scenario update.
All new fitted outputs remain under ignored `validation/results/` and
`validation/checkpoints/`. Each fit is single-threaded. Two to four worker
processes are generally sufficient; complete computation can take hours.

```sh
python validation/test_nested_boundary.py
python validation/test_overlap_analysis.py
python validation/test_masked_driver.py
python validation/nested_imputation.py --M 20 --sweeps 3 --rounds 200 --outer-folds 5 --seed 42000 --all-endpoints --tag field_nested_M20_s3_r200 --checkpoint-tag field_nested_M20_s3_r200
python validation/summarize_nested.py --tag field_nested_M20_s3_r200
python validation/masked_detection_validation.py fit --M 5 --sweeps 3 --rounds 200 --seed 42000 --tag masked_M5_s3_r200
python validation/masked_detection_validation.py analyze --tag masked_M5_s3_r200 --draws 100 --masking train_q20
python validation/masked_detection_validation.py analyze --tag masked_M5_s3_r200 --draws 100 --masking reported_DL3
python validation/masked_detection_validation.py continue --tag masked_M5_s3_r200 --final-sweeps 5
python validation/summarize_nested.py --tag masked_M5_s3_r200_S5
python validation/overlap_analysis.py --bootstrap 1999
python validation/overlap_full_profile_check.py
python validation/measurement_bounds.py
python validation/stage_measurement_bounds.py
python validation/independent_measurement_bounds_check.py
python validation/update_load_scenarios.py
```

The boundary test perturbs held-out limits, censoring and covariates and checks
that fitted training models and values are unchanged. It also checks exact
replay and continuation of a three-sweep chain to five sweeps. Reconstruction
tests mask quantified held-out values, using a threshold from training
detections or three times each reported laboratory limit. Such tests concern
the detection-selected population; genuine non-detect truth is unobserved.

The design comparisons use equal-plant, overlap and receiving-region-by-era
reference populations. Weight construction excludes effluent outcomes and
balance diagnostics cover all twenty-five original covariates. Their estimand
is a fixed-design plant-mean contrast, distinct from the primary DML
coefficient. The six primary-only plants and incomplete composition balance
remain important limitations.

Measurement ranges hold detections fixed and vary non-detects independently
over [0, DL]. They describe concentration-compatible family contrasts, shared-
primary-effluent stage differences and precursor loss/gain indices. They are
neither sampling confidence intervals nor causal effect bounds. The independent
check reconstructs intervals from raw cleaned concentration bounds without
calling the bound-producing aggregation functions.

The expanded benchmark has 200 independent replicates **per condition**, a
known seven-compound molar PFCA-sum effect and a separate null PFOS endpoint.
A Gaussian generator at reported laboratory limits and a clustered heavy-tailed
generator at tripled limits are each evaluated with a null and a non-null PFCA
effect. Imputation methods have
matched per-fit M=2, one sweep, 30 rounds and three outer folds; total fit counts
differ. This reduced experiment does not prove calibration of the richer field
analysis. The four planned conditions are reproduced with:

```sh
python validation/expanded_benchmark.py --output validation/results/expanded_benchmark/full200 --dgps gaussian_iid clustered_t3 --effects null effect --limits 1 3 --paired-settings --reps 200 --M 2 --sweeps 1 --rounds 30 --inner-folds 3 --outer-folds 3 --seed 20261101 --workers 2
python validation/expanded_benchmark.py --output validation/results/expanded_benchmark/full200 --summarize-only
```

The Gaussian setting uses the reported limits, and the combined-stress setting
uses tripled limits. `--paired-settings` pairs these two generators with their
respective limit multipliers; omitting it creates a different factorial design.
See `python validation/expanded_benchmark.py --help` for further controls.
Results are resumable only within the same
configuration. Run `--summarize-only` after all planned replicates finish to
derive finite-M interval metrics and paired Monte Carlo errors. Never pool
endpoints or partially completed conditions as extra independent replicates.

A separate known-nuisance design reference uses one thousand replicates in
each generator/effect condition. It adds a clustered Gaussian generator and
uses uncensored known-nuisance scores, keeping the same monitored plant design:

```sh
python validation/expanded_benchmark.py --output validation/results/expanded_benchmark/known_nuisance_design --dgps gaussian_iid clustered_gaussian clustered_t3 --effects null effect --limits 1 --methods known_h --reps 1000 --outer-folds 3 --seed 20261102 --workers 2
python validation/expanded_benchmark.py --output validation/results/expanded_benchmark/known_nuisance_design --summarize-only
```

This design reference is distinct from the four-condition imputation benchmark;
its known-nuisance scores do not use censoring or imputation.

Secondary stage/process/era/biosolids/precursor/qualifier/offset estimates retain
their historical full-population imputation settings. Stitched held-out
completions are used for plots and descriptive loads only, and must not be
fed into a new nuisance cross-fitting procedure labelled as nested.
