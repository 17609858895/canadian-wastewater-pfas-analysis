"""Detection-limit-aware double machine learning (working name: LOD-DML).

Two parts
1. Survival-boosted chained imputation of left-censored log concentrations.
   Each censored variable is treated as a left-censored "survival time" and
   modelled by XGBoost-AFT (normal) on all other variables, covariates and the
   treatment (congenial imputation). sigma is re-estimated by censored
   (Tobit) likelihood on grouped out-of-fold means; censored entries are drawn
   from N(mu, sigma) truncated above at ln(DL). Plant-cluster bootstrap of the
   training rows supplies variation across imputations. Calibration is assessed
   empirically; imputation precedes nuisance cross-fitting.
2. Partially linear DML on each completed data set with plant-grouped
   cross-fitting; Rubin's rules across imputations; plant-level permutation
   comparison as a descriptive small-cluster diagnostic.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize, stats
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LN10 = np.log(10.0)


# ---------------------------------------------------------------- censoring
def tobit_sigma(resid_obs, z_cens):
    """MLE of sigma given residuals of observed values and standardized
    upper bounds (ln DL - mu) of censored values (mean fixed to model)."""
    def nll(ls):
        s = np.exp(ls)
        ll = stats.norm.logpdf(resid_obs, scale=s).sum()
        ll += stats.norm.logcdf(z_cens / s).sum()
        return -ll
    r = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(5.0)), method="bounded")
    return float(np.exp(r.x))


def draw_trunc_upper(mu, sigma, upper, rng):
    b = (upper - mu) / sigma
    u = rng.uniform(0, stats.norm.cdf(b))
    u = np.clip(u, 1e-12, None)
    return mu + sigma * stats.norm.ppf(u)


def fit_aft(X, lo, hi, params=None, seed=0, offset=None):
    """lo/hi on natural-log scale; lo=-inf for censored. offset: optional
    natural-log base margin (persistence anchor), learned part is the deviation."""
    p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
             aft_loss_distribution_scale=0.8, tree_method="hist", learning_rate=0.05,
             max_depth=3, min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=2.0, seed=seed, verbosity=0, nthread=2)
    if params:
        p.update(params)
    d = xgb.DMatrix(X)
    d.set_float_info("label_lower_bound", np.where(np.isfinite(lo), np.exp(lo), 0.0))
    d.set_float_info("label_upper_bound", np.exp(hi))
    if offset is not None:
        d.set_base_margin(offset)
    return xgb.train(p, d, num_boost_round=200)


def predict_mu(bst, X, offset=None):
    if isinstance(bst, TobitLinear):
        return bst.predict(X) + (0 if offset is None else offset)
    d = xgb.DMatrix(X)
    if offset is not None:
        d.set_base_margin(offset)
    return bst.predict(d, output_margin=True)


class TobitLinear:
    """Ridge-penalized linear left-censored (Tobit) regression; ablation engine."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, lo, hi):
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Xs = np.column_stack([np.ones(len(X)), (X - self.m) / self.s])
        obs = np.isfinite(lo)
        y0 = np.where(obs, lo, hi - np.log(2))
        b0 = np.linalg.lstsq(Xs, y0, rcond=None)[0]

        def nll(par):
            b, ls = par[:-1], par[-1]
            sg = np.exp(ls)
            mu = Xs @ b
            ll = stats.norm.logpdf(lo[obs], mu[obs], sg).sum() + stats.norm.logcdf((hi[~obs] - mu[~obs]) / sg).sum()
            return -ll + self.alpha * np.sum(b[1:] ** 2)

        r = optimize.minimize(nll, np.r_[b0, 0.0], method="L-BFGS-B")
        self.b = r.x[:-1]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.m) / self.s]) @ self.b


def _fit_engine(engine, X, lo, hi, aft_params, seed):
    if engine == "tobit":
        return TobitLinear().fit(X, lo, hi)
    return fit_aft(X, lo, hi, aft_params, seed)


def chained_impute(V_lo, V_hi, Z, groups, n_sweeps=3, seed=0, bootstrap=True, aft_params=None, engine="aft"):
    """V_lo/V_hi: (n, p) natural-log bounds (lo=-inf when censored, equal when
    observed). Z: (n, q) fully observed covariates incl. treatment.
    engine: 'aft' (XGBoost-AFT, default) or 'tobit' (linear ablation).
    Returns one completed (n, p) matrix of latent ln values."""
    rng = np.random.default_rng(seed)
    n, p = V_lo.shape
    cens = ~np.isfinite(V_lo)
    V = np.where(cens, V_hi - np.log(2.0), V_lo)  # init at DL/2
    ug = np.unique(groups)
    for s in range(n_sweeps):
        for j in range(p):
            if cens[:, j].sum() == 0:
                continue
            others = np.delete(V, j, axis=1)
            X = np.column_stack([Z, others])
            if bootstrap:
                bg = rng.choice(ug, size=len(ug), replace=True)
                idx = np.concatenate([np.where(groups == g)[0] for g in bg])
            else:
                idx = np.arange(n)
            # grouped OOF means for sigma, full-bootstrap model for draws
            oof = np.full(n, np.nan)
            gk = GroupKFold(n_splits=3)
            for tr, te in gk.split(X, groups=groups):
                b = _fit_engine(engine, X[tr], V_lo[tr, j], V_hi[tr, j], aft_params, seed + s)
                oof[te] = predict_mu(b, X[te])
            obs = ~cens[:, j]
            sig = tobit_sigma((V_lo[obs, j] - oof[obs]), (V_hi[~obs, j] - oof[~obs]))
            bst = _fit_engine(engine, X[idx], V_lo[idx, j], V_hi[idx, j], aft_params, seed + 100 + s)
            mu = predict_mu(bst, X)
            c = cens[:, j]
            V[c, j] = draw_trunc_upper(mu[c], sig, V_hi[c, j], rng)
    return V


# --------------------------------------------------------------------- DML
LIGHT = False


def _reg():
    if LIGHT:
        return [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
                HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=120, min_samples_leaf=10, random_state=0)]
    return [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
            ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0),
            HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=10, random_state=0)]


def _clf():
    if LIGHT:
        return [make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))]
    return [make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0)]


def crossfit_nuisance(W, y, d, groups, n_splits=5, d_only=False, m_hat=None):
    n = len(y)
    if m_hat is None and not d_only:
        m_hat = np.zeros(n)
        for tr, te in GroupKFold(n_splits).split(W, groups=groups):
            m_hat[te] = np.mean([r.fit(W[tr], y[tr]).predict(W[te]) for r in _reg()], axis=0)
    e_hat = np.zeros(n)
    for tr, te in GroupKFold(n_splits).split(W, groups=groups):
        if len(np.unique(d[tr])) < 2:
            e_hat[te] = d[tr].mean()
            continue
        e_hat[te] = np.mean([c.fit(W[tr], d[tr]).predict_proba(W[te])[:, 1] for c in _clf()], axis=0)
    return m_hat, np.clip(e_hat, 0.02, 0.98)


def plr(y, d, m_hat, e_hat, groups):
    ry, rd = y - m_hat, d - e_hat
    theta = np.sum(rd * ry) / np.sum(rd * rd)
    psi = (ry - theta * rd) * rd
    J = np.mean(rd * rd)
    g = pd.Series(psi).groupby(groups).sum().values
    G = len(g)
    var = (G / (G - 1)) * np.sum(g ** 2) / (len(y) ** 2) / J ** 2
    return theta, var


def rubin(thetas, vars_):
    thetas, vars_ = np.asarray(thetas), np.asarray(vars_)
    M = len(thetas)
    qbar = thetas.mean()
    ubar = vars_.mean()
    b = thetas.var(ddof=1) if M > 1 else 0.0
    T = ubar + (1 + 1 / M) * b
    return qbar, np.sqrt(T)


def permutation_p(y, d, W, groups, m_hat, n_perm=500, seed=0, plant_d=None):
    """Plant-level label permutation (keeps number of control plants)."""
    rng = np.random.default_rng(seed)
    _, e_hat = crossfit_nuisance(W, y, d, groups, d_only=True, m_hat=m_hat)
    th0, _ = plr(y, d, m_hat, e_hat, groups)
    ug = np.unique(groups)
    pdict = pd.Series(d, index=groups).groupby(level=0).first()
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(pdict.values), index=pdict.index)
        dp = perm.loc[groups].values.astype(float)
        _, ep = crossfit_nuisance(W, y, dp, groups, d_only=True, m_hat=m_hat)
        null.append(plr(y, dp, m_hat, ep, groups)[0])
    null = np.asarray(null)
    return th0, (1 + np.sum(np.abs(null) >= abs(th0))) / (1 + n_perm), null
