"""Plant-fold-trained censoring imputation and partially linear contrasts.

This is a validation extension of the existing analysis, not a new estimator.
Conditional models, residual scales, bootstrap samples and covariate fills use
outer-training plants only. Held-out bounds condition only their own draws.
Pooling is approximate: chained conditional models do not establish a proper
joint imputation distribution, and cluster intervals require empirical checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT
LN10 = np.log(10.0)


def digest(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        a = np.asarray(arr)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        if a.dtype.kind in "OU":
            h.update(json.dumps(a.tolist(), sort_keys=True).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fill_train_medians(Ztrain, Ztest):
    """Learn all missing-covariate fill values solely in the training fold."""
    Ztrain, Ztest = np.asarray(Ztrain, float).copy(), np.asarray(Ztest, float).copy()
    with np.errstate(all="ignore"):
        med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in Ztrain.T])
    Ztrain = np.where(np.isfinite(Ztrain), Ztrain, med)
    Ztest = np.where(np.isfinite(Ztest), Ztest, med)
    return Ztrain, Ztest, med


def tobit_sigma(obs_residual, censoring_residual):
    def nll(log_scale):
        scale = np.exp(log_scale)
        return -(stats.norm.logpdf(obs_residual, scale=scale).sum()
                 + special.log_ndtr(censoring_residual / scale).sum())
    fit = optimize.minimize_scalar(nll, bounds=(np.log(.05), np.log(5.)), method="bounded")
    return float(np.exp(fit.x))


def draw_upper(mu, sigma, upper, rng):
    """Stable inverse-CDF draw, including tails with Phi(b) below 1e-12."""
    mu, upper = np.asarray(mu), np.asarray(upper)
    log_probability = special.log_ndtr((upper - mu) / sigma) + np.log(rng.uniform(np.finfo(float).tiny, 1., len(mu)))
    z = mu + sigma * special.ndtri_exp(log_probability)
    return np.minimum(z, upper)  # floating-point roundoff at the upper boundary


def fit_aft(X, lo, hi, *, rounds=200, seed=0, params=None):
    config = dict(objective="survival:aft", eval_metric="aft-nloglik",
                  aft_loss_distribution="normal", aft_loss_distribution_scale=.8,
                  tree_method="hist", learning_rate=.05, max_depth=3,
                  min_child_weight=3, subsample=.8, colsample_bytree=.8,
                  reg_lambda=2., seed=int(seed), verbosity=0, nthread=1)
    if params:
        config.update(params)
    config["nthread"] = 1
    matrix = xgb.DMatrix(np.asarray(X), nthread=1)
    matrix.set_float_info("label_lower_bound", np.where(np.isfinite(lo), np.exp(lo), 0.))
    matrix.set_float_info("label_upper_bound", np.exp(hi))
    return xgb.train(config, matrix, num_boost_round=int(rounds))


def predict_mu(model, X):
    return model.predict(xgb.DMatrix(np.asarray(X), nthread=1), output_margin=True)


def impute_fold(train_lo, train_hi, test_lo, test_hi, Ztrain, Ztest,
                training_groups, *, n_sweeps=3, rounds=200, inner_folds=3,
                seed=0, bootstrap=True, aft_params=None, store_model_hashes=False,
                trace_zip=None,initial_train=None,initial_test=None,start_sweep=0,
                training_rng_state=None,test_rng_state=None):
    """Fit a conditional chain in training plants, applying it to both splits.

    Test rows never enter fitting or scale estimation. Separate RNG streams
    ensure changing test censoring cannot change subsequent training draws.
    Z must already contain any treatment indicator and training-only fills.
    """
    train_lo, train_hi, test_lo, test_hi = map(lambda a: np.asarray(a, float), (train_lo, train_hi, test_lo, test_hi))
    Ztrain, Ztest = np.asarray(Ztrain, float), np.asarray(Ztest, float)
    training_groups = np.asarray(training_groups)
    assert train_lo.shape == train_hi.shape and test_lo.shape == test_hi.shape
    assert train_lo.shape[1] == test_lo.shape[1]
    assert np.isfinite(train_hi).all() and np.isfinite(test_hi).all()
    assert np.isfinite(Ztrain).all() and np.isfinite(Ztest).all()
    ctrain, ctest = ~np.isfinite(train_lo), ~np.isfinite(test_lo)
    if start_sweep and any(value is None for value in (initial_train,initial_test,training_rng_state,test_rng_state)):
        raise ValueError('Continuation requires both completed matrices and both RNG states')
    Vtrain = np.where(ctrain, train_hi - np.log(2.), train_lo) if initial_train is None else np.asarray(initial_train,float).copy()
    Vtest = np.where(ctest, test_hi - np.log(2.), test_lo) if initial_test is None else np.asarray(initial_test,float).copy()
    assert Vtrain.shape==train_lo.shape and Vtest.shape==test_lo.shape
    ug = np.unique(training_groups)
    if len(ug) < 2:
        raise ValueError("At least two training plants are required.")
    # Never share a stream with held-out draws.
    rt = np.random.default_rng(np.random.SeedSequence([int(seed), 104729]))
    re = np.random.default_rng(np.random.SeedSequence([int(seed), 130363]))
    if training_rng_state is not None:rt.bit_generator.state=training_rng_state
    if test_rng_state is not None:re.bit_generator.state=test_rng_state
    splits = list(GroupKFold(min(inner_folds, len(ug))).split(Vtrain, groups=training_groups))
    audit = []
    trace_path=Path(trace_zip) if trace_zip else None
    trace_temporary=trace_path.with_suffix('.tmp.zip') if trace_path else None
    if trace_path:trace_path.parent.mkdir(parents=True,exist_ok=True)
    trace_archive=ZipFile(trace_temporary,'w',ZIP_DEFLATED,compresslevel=1) if trace_path else None
    for sweep in range(int(start_sweep),int(start_sweep+n_sweeps)):
        for j in range(Vtrain.shape[1]):
            # Fit the fixed target list even if training censoring is absent.
            # Test censoring must not determine which training models/RNG draws exist.
            Xtr = np.column_stack([Ztrain, np.delete(Vtrain, j, axis=1)])
            Xte = np.column_stack([Ztest, np.delete(Vtest, j, axis=1)])
            oof = np.empty(len(Xtr))
            fit_seed = int(seed + 1009 * sweep + 37 * j)
            for k, (itr, ite) in enumerate(splits):
                inner = fit_aft(Xtr[itr], train_lo[itr, j], train_hi[itr, j], rounds=rounds, seed=fit_seed + k, params=aft_params)
                oof[ite] = predict_mu(inner, Xtr[ite])
            observed = ~ctrain[:, j]
            sigma = tobit_sigma(train_lo[observed, j] - oof[observed], train_hi[~observed, j] - oof[~observed])
            if bootstrap:
                sampled = rt.choice(ug, len(ug), replace=True)
                ids = np.concatenate([np.flatnonzero(training_groups == g) for g in sampled])
            else:
                ids = np.arange(len(Xtr))
            model = fit_aft(Xtr[ids], train_lo[ids, j], train_hi[ids, j], rounds=rounds, seed=fit_seed + 7919, params=aft_params)
            mt = predict_mu(model, Xtr)
            me = predict_mu(model, Xte) if len(Xte) else np.empty(0)
            Vtrain[ctrain[:, j], j] = draw_upper(mt[ctrain[:, j]], sigma, train_hi[ctrain[:, j], j], rt)
            Vtest[ctest[:, j], j] = draw_upper(me[ctest[:, j]], sigma, test_hi[ctest[:, j], j], re)
            record = dict(sweep=sweep + 1, variable=j, sigma=sigma,
                          detected_training=int(observed.sum()), censored_training=int(ctrain[:, j].sum()),
                          training_design_hash=digest(Xtr, train_lo[:, j], train_hi[:, j]),
                          bootstrap_index_hash=digest(ids))
            if store_model_hashes:
                record["model_hash"] = hashlib.sha256(bytes(model.save_raw())).hexdigest()
            if trace_archive:
                record['trace_member']=f's{sweep:02d}_v{j:02d}.ubj'
                trace_archive.writestr(record['trace_member'],bytes(model.save_raw()))
            audit.append(record)
    if trace_archive:
        trace_archive.writestr('trace.json',json.dumps(dict(n_sweeps=n_sweeps,rounds=rounds,
            start_sweep=start_sweep,inner_folds=inner_folds,seed=seed,variables=Vtrain.shape[1],
            final_training_rng_state=rt.bit_generator.state,final_test_rng_state=re.bit_generator.state,
            training_bounds_hash=digest(train_lo,train_hi),training_covariate_hash=digest(Ztrain),
            training_groups_hash=digest(training_groups),conditional_models=audit),indent=2))
        trace_archive.close();trace_temporary.replace(trace_path)
    assert np.array_equal(Vtrain[~ctrain], train_lo[~ctrain])
    assert np.array_equal(Vtest[~ctest], test_lo[~ctest])
    assert np.isfinite(Vtrain).all() and np.isfinite(Vtest).all()
    assert (Vtrain[ctrain] <= train_hi[ctrain] + 1e-10).all()
    assert (Vtest[ctest] <= test_hi[ctest] + 1e-10).all()
    return Vtrain, Vtest, audit


def replay_imputation_trace(test_lo, test_hi, Ztest, trace_zip, *, predictive_draws=100, seed=0,
                            initial_values=None,initial_rng_state=None):
    """Repeat held-out predictive chains under fixed training-chain models.

    Returns (draw, row, variable). Predictive draws are not additional bootstrap
    imputations: they condition on the training models captured by one chain.
    """
    test_lo,test_hi,Ztest=map(lambda a:np.asarray(a,float),(test_lo,test_hi,Ztest))
    n,p=test_lo.shape
    assert test_lo.shape==test_hi.shape and Ztest.shape[0]==n
    censored=np.tile(~np.isfinite(test_lo),(predictive_draws,1))
    upper=np.tile(test_hi,(predictive_draws,1))
    initial=np.where(np.isfinite(test_lo),test_lo,test_hi-np.log(2.)) if initial_values is None else np.asarray(initial_values,float)
    V=np.tile(initial,(predictive_draws,1))
    Z=np.tile(Ztest,(predictive_draws,1));rng=np.random.default_rng(np.random.SeedSequence([int(seed),130363]))
    if initial_rng_state is not None:rng.bit_generator.state=initial_rng_state
    with ZipFile(trace_zip) as archive:
        metadata=json.loads(archive.read('trace.json'))
        if metadata.get('start_sweep',0) and (initial_values is None or initial_rng_state is None):
            raise ValueError('A continuation trace needs its starting values and RNG state')
        assert metadata['variables']==p
        for step in metadata['conditional_models']:
            j=step['variable'];c=censored[:,j]
            if not c.any():continue
            model=xgb.Booster(params={'nthread':1})
            model.load_model(bytearray(archive.read(step['trace_member'])))
            model.set_param({'nthread':1})
            X=np.column_stack([Z,np.delete(V,j,axis=1)])
            mu=predict_mu(model,X)
            V[c,j]=draw_upper(mu[c],step['sigma'],upper[c,j],rng)
    assert np.isfinite(V).all() and np.all(V[censored]<=upper[censored]+1e-10)
    return V.reshape(predictive_draws,n,p)


def endpoints_from_completed(V, compound_names, mol_weights, endpoints):
    p = len(compound_names); Vi, Ve = V[:, :p], V[:, p:]
    result = {}
    for key, members in endpoints.items():
        indices = [compound_names.index(c) for c in members]
        weights = np.array([mol_weights[c] for c in members])
        result[key] = np.log10((np.exp(Ve[:, indices]) / weights).sum(1) / (np.exp(Vi[:, indices]) / weights).sum(1))
    return result


def regressors(light=False, seed=0):
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    if light:
        return [ridge, HistGradientBoostingRegressor(max_depth=3, learning_rate=.08, max_iter=120, min_samples_leaf=10, random_state=seed)]
    return [ridge, ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=.5, n_jobs=1, random_state=seed),
            HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=250, min_samples_leaf=10, random_state=seed)]


def propensity(Wtrain, Wtest, dtrain, *, light=False, seed=0):
    if len(np.unique(dtrain)) < 2:
        return np.repeat(np.mean(dtrain), len(Wtest)).clip(.02, .98)
    models = [make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=2000))]
    if not light:
        models.append(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=.5, n_jobs=1, random_state=seed))
    predictions = [model.fit(Wtrain, dtrain).predict_proba(Wtest)[:, 1] for model in models]
    return np.mean(predictions, axis=0).clip(.02, .98)


def plr_scores(y, d, mh, eh, groups):
    ry, rd = y - mh, d - eh
    theta = np.sum(rd * ry) / np.sum(rd ** 2)
    psi = (ry - theta * rd) * rd
    J = np.mean(rd ** 2)
    cluster = pd.Series(psi).groupby(np.asarray(groups)).sum().values
    G = len(cluster)
    var = G / (G - 1) * np.sum(cluster ** 2) / len(y) ** 2 / J ** 2
    return float(theta), float(var), psi, rd


def pool(estimates):
    q = np.asarray([e["theta"] for e in estimates]); u = np.asarray([e["variance"] for e in estimates])
    M = len(q); within = u.mean(); between = q.var(ddof=1) if M > 1 else 0.
    theta = q.mean(); se = np.sqrt(within + (1 + 1 / M) * between)
    return dict(theta=float(theta), se=float(se), lo95=float(theta - 1.96 * se), hi95=float(theta + 1.96 * se),
                within_variance=float(within), between_variance=float(between),
                imputation_MCSE=float(np.sqrt(between / M)), M=M)


def barnard_rubin_df(M, within, between, df_complete):
    """Stable finite-M MI degrees of freedom, matching current mice source.

    https://raw.githubusercontent.com/amices/mice/master/R/barnard.rubin.R
    No lambda floor is imposed. For single complete/substituted datasets use
    the selected complete-data t reference directly, not this MI adjustment.
    """
    if M<2:raise ValueError('Barnard-Rubin MI degrees of freedom require M>=2')
    if within<0 or between<0:raise ValueError('Variances must be nonnegative')
    total=within+(1+1/M)*between
    if total<=0:raise ValueError('Pooled variance must be positive')
    lam=(1+1/M)*between/total
    old=(M-1)/lam**2 if lam>0 else float('inf')
    if np.isinf(df_complete):observed=float('inf');combined=old
    else:
        if df_complete<=0:raise ValueError('Complete-data degrees of freedom must be positive')
        tmp=(1-lam)*(1+df_complete)*df_complete
        observed=tmp/(df_complete+3)
        combined=(M-1)*tmp/((df_complete+3)*(M-1)+lam**2*tmp)
    relative=(1+1/M)*between/within if within>0 else float('inf')
    fmi=lam+2*(1-lam)/(combined+3)
    return dict(lambda_mi=float(lam),df_old=float(old),df_observed=float(observed),
                df_BR=float(combined),df_complete=float(df_complete),
                relative_increase_variance=float(relative),estimated_FMI=float(fmi))


def nested_dml(Vlo, Vhi, Z, d, groups, compound_names, mol_weights, endpoints,
               *, M=10, n_sweeps=3, rounds=200, outer_folds=5, inner_folds=3,
               seed=42000, light=False, checkpoint_dir=None, folds=None,
               return_scores=True, include_treatment=True, imputation_indices=None):
    """Nested imputation/held-out nuisance prediction with resumable fold fits."""
    Vlo, Vhi, Z, d, groups = map(np.asarray, (Vlo, Vhi, Z, d, groups))
    if folds is None:
        folds = list(GroupKFold(outer_folds).split(Z, groups=groups))
    n = len(d); p = len(compound_names)
    assert Vlo.shape == Vhi.shape == (n, 2 * p)
    completed = np.zeros(n, int)
    for tr, te in folds:
        assert not np.intersect1d(groups[tr], groups[te]).size
        completed[te] += 1
    assert (completed == 1).all()
    config = dict(M=M, n_sweeps=n_sweeps, rounds=rounds, inner_folds=inner_folds,
                  seed=seed, light=light, include_treatment=include_treatment,
                  compound_names=list(compound_names), endpoints=endpoints,
                  mol_weights={key:mol_weights[key] for key in compound_names},
                  data_hash=digest(Vlo, Vhi, Z, d, groups),
                  folds_hash=digest(*[np.asarray(te) for tr, te in folds]))
    checkpoint = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint:
        checkpoint.mkdir(parents=True, exist_ok=True)
        marker = checkpoint / 'configuration.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            # M may be extended without invalidating already completed imputations.
            assert {k:v for k,v in previous.items() if k != 'M'} == {k:v for k,v in config.items() if k != 'M'}, 'Checkpoint configuration mismatch'
        temporary_marker=marker.with_suffix(f'.{os.getpid()}.tmp.json')
        temporary_marker.write_text(json.dumps(config, indent=2));temporary_marker.replace(marker)
    estimates = []; scores = []; fold_audit = []; start = time.time()
    with threadpool_limits(limits=1):
        for m in (range(M) if imputation_indices is None else imputation_indices):
            assert 0 <= m < M
            yy = {key:np.zeros(n) for key in endpoints}; mm = {key:np.zeros(n) for key in endpoints}
            eh = np.zeros(n); Vi = np.zeros((n, p))
            for k, (train, test) in enumerate(folds):
                cache = checkpoint / f'm{m:03d}_fold{k:02d}.npz' if checkpoint else None
                audit_file = checkpoint / f'm{m:03d}_fold{k:02d}.json' if checkpoint else None
                if cache and cache.exists() and audit_file.exists():
                    saved = np.load(cache)
                    for key in endpoints:
                        yy[key][test] = saved['y_' + key]; mm[key][test] = saved['mh_' + key]
                    eh[test] = saved['eh']; Vi[test] = saved['Vi']
                    audit = json.loads(audit_file.read_text())
                else:
                    ztr, zte, med = fill_train_medians(Z[train], Z[test])
                    zdtr = np.column_stack([ztr, d[train]]) if include_treatment else ztr
                    zdte = np.column_stack([zte, d[test]]) if include_treatment else zte
                    vtr, vte, conditional = impute_fold(Vlo[train], Vhi[train], Vlo[test], Vhi[test], zdtr, zdte, groups[train],
                                                       n_sweeps=n_sweeps, rounds=rounds, inner_folds=inner_folds,
                                                       seed=seed + m * 1000003 + k * 10007)
                    Wtr, Wte = np.column_stack([ztr, vtr[:, :p] / LN10]), np.column_stack([zte, vte[:, :p] / LN10])
                    yt, ye = endpoints_from_completed(vtr, compound_names, mol_weights, endpoints), endpoints_from_completed(vte, compound_names, mol_weights, endpoints)
                    eh[test] = propensity(Wtr, Wte, d[train], light=light)
                    Vi[test] = vte[:, :p]
                    for key in endpoints:
                        yy[key][test] = ye[key]
                        mm[key][test] = np.mean([model.fit(Wtr, yt[key]).predict(Wte) for model in regressors(light=light)], axis=0)
                    audit = dict(imputation=m, fold=k, training_plants=sorted(np.unique(groups[train]).tolist()),
                                 heldout_plants=sorted(np.unique(groups[test]).tolist()),
                                 training_primary_plants=int(len(np.unique(groups[train][d[train] == 0]))),
                                 heldout_primary_plants=int(len(np.unique(groups[test][d[test] == 0]))),
                                 covariate_medians=med.tolist(), conditional_models=conditional,
                                 elapsed_seconds=time.time()-start)
                    if cache:
                        contents = {'eh':eh[test], 'Vi':Vi[test]}
                        contents.update({'y_'+key:yy[key][test] for key in endpoints})
                        contents.update({'mh_'+key:mm[key][test] for key in endpoints})
                        temp = cache.with_suffix('.tmp.npz');np.savez_compressed(temp, **contents);temp.replace(cache)
                        audit_file.write_text(json.dumps(audit, indent=2))
                fold_audit.append({key:value for key,value in audit.items() if key != 'conditional_models'})
                print(f'Nested MI {m+1}/{M} fold {k+1}/{len(folds)} elapsed {time.time()-start:.1f}s', flush=True)
            for key in endpoints:
                theta, variance, psi, rd = plr_scores(yy[key], d, mm[key], eh, groups)
                estimates.append(dict(imputation=m, outcome=key, theta=theta, variance=variance))
                if return_scores:
                    scores.append(dict(imputation=m, outcome=key, y=yy[key], mh=mm[key], eh=eh.copy(), psi=psi, rd=rd, Vi=Vi.copy()))
    summary = {key:pool([e for e in estimates if e['outcome'] == key]) for key in endpoints}
    return dict(summary=summary, estimates=estimates, scores=scores, fold_audit=fold_audit,
                config=config, elapsed_seconds=time.time()-start)


def field_data(source_root=None):
    # The local frozen source is preferred; a standalone code clone can use
    # its own scripts/data_clean directories or an explicit analysis-root env.
    default_source=FROZEN if FROZEN.exists() else Path(__file__).resolve().parents[1]
    source_root=Path(source_root or os.environ.get('ECCC_PFAS_ANALYSIS_ROOT',str(default_source))).resolve()
    sys.path.insert(0, str(source_root/'scripts'))
    spec=importlib.util.spec_from_file_location('_nested_source_design',source_root/'scripts/s03_upgrade_effect.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    pr = old.load_design()
    Z = pr[old.ZCOLS].values.astype(float)
    Z[pr.flow_na.to_numpy().astype(bool), old.ZCOLS.index('log_flow')] = np.nan
    Z[pr.T_na.to_numpy().astype(bool), old.ZCOLS.index('T')] = np.nan
    plants = pd.read_csv(source_root/'data_clean/plants.csv')
    plants['code2'] = plants.code.str.replace(r' \(.*\)', '', regex=True)
    plants = plants.drop_duplicates('code2').set_index('code2')
    Z[:, old.ZCOLS.index('pres')] = pd.to_numeric(pr.WWTP_CODE.map(plants.pct_res), errors='coerce')
    ilo, ihi = old.bounds(pr, 'INF');elo, ehi = old.bounds(pr, 'EFF')
    return dict(Vlo=np.hstack([ilo, elo]), Vhi=np.hstack([ihi, ehi]), Z=Z,
                d=pr.bio.values, groups=pr.WWTP_CODE.values, compound_names=old.CMP,
                mol_weights=old.MW, endpoints={'sumPFCA':old.PFCA, 'sumPFSA':old.PFSA}), pr


def save_result(result, prefix):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(outcome=key, **value) for key,value in result['summary'].items()]).to_csv(prefix.with_suffix('.csv'), index=False)
    pd.DataFrame(result['estimates']).to_csv(prefix.parent/(prefix.name+'_imputations.csv'), index=False)
    compact={key:value for key,value in result.items() if key != 'scores'}
    prefix.with_suffix('.json').write_text(json.dumps(compact, indent=2))
    if result['scores']:
        arrays={f'm{s["imputation"]}_{s["outcome"]}_{key}':s[key] for s in result['scores'] for key in ('y','mh','eh','psi','rd','Vi')}
        np.savez_compressed(prefix.parent/(prefix.name+'_scores.npz'), **arrays)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--M',type=int,default=10)
    parser.add_argument('--sweeps',type=int,default=3);parser.add_argument('--rounds',type=int,default=200)
    parser.add_argument('--outer-folds',type=int,default=5);parser.add_argument('--inner-folds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42000);parser.add_argument('--light',action='store_true')
    parser.add_argument('--tag',default='field_nested');parser.add_argument('--checkpoint-tag')
    parser.add_argument('--m-indices',type=int,nargs='+');parser.add_argument('--all-endpoints',action='store_true')
    args=parser.parse_args()
    data,pr=field_data();tag=args.tag
    if args.all_endpoints:
        data['endpoints'].update({key:[key] for key in data['compound_names']})
    result=nested_dml(**data,M=args.M,n_sweeps=args.sweeps,rounds=args.rounds,outer_folds=args.outer_folds,
                      inner_folds=args.inner_folds,seed=args.seed,light=args.light,
                      checkpoint_dir=OUT/'checkpoints'/(args.checkpoint_tag or tag),
                      imputation_indices=args.m_indices)
    save_result(result,OUT/'results'/tag)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
