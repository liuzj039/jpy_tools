from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from numpy import linalg as la
from scipy.sparse import issparse

from typing import Collection, Tuple, Optional, Union


def _design_matrix(
    model: pd.DataFrame, batch_key: str, batch_levels: Collection[str]
) -> pd.DataFrame:
    """\
    Computes a simple design matrix.
    Parameters
    --------
    model
        Contains the batch annotation
    batch_key
        Name of the batch column
    batch_levels
        Levels of the batch annotation
    Returns
    --------
    The design matrix for the regression problem
    """
    import patsy

    design = patsy.dmatrix(
        "~ 0 + C(Q('{}'), levels=batch_levels)".format(batch_key),
        model,
        return_type="dataframe",
    )
    model = model.drop([batch_key], axis=1)
    numerical_covariates = model.select_dtypes("number").columns.values

    other_cols = [c for c in model.columns.values if c not in numerical_covariates]

    if other_cols:
        col_repr = " + ".join("Q('{}')".format(x) for x in other_cols)
        factor_matrix = patsy.dmatrix(
            "~ 0 + {}".format(col_repr), model[other_cols], return_type="dataframe"
        )

        design = pd.concat((design, factor_matrix), axis=1)

    if numerical_covariates is not None:

        for nC in numerical_covariates:
            design[nC] = model[nC]

    return design


def _standardize_data(
    model: pd.DataFrame,
    data: pd.DataFrame,
    batch_key: str,
    modelAll: pd.DataFrame,
    dataAll: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """\
    Standardizes the data per gene.
    The aim here is to make mean and variance be comparable across batches.
    Parameters
    --------
    model
        Contains the batch annotation
    data
        Contains the Data
    batch_key
        Name of the batch column in the model matrix
    Returns
    --------
    s_data
        Standardized Data
    design
        Batch assignment as one-hot encodings
    var_pooled
        Pooled variance per gene
    stand_mean
        Gene-wise mean
    """

    # compute the design matrix
    batch_items = model.groupby(batch_key).groups.items()
    batch_levels, batch_info = zip(*batch_items)
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    design = _design_matrix(model, batch_key, batch_levels)

    # compute pooled variance estimator
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = (data - np.dot(design, B_hat).T) ** 2
    var_pooled = np.dot(var_pooled, np.ones((int(n_array), 1)) / int(n_array))

    # Compute the means
    if np.sum(var_pooled == 0) > 0:
        print(f"Found {np.sum(var_pooled == 0)} genes with zero variance.")
    stand_mean = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array)))
    )
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T

    # need to be a bit careful with the zero variance genes
    # just set the zero variance genes to zero in the standardized data
    s_data = np.where(
        var_pooled == 0,
        0,
        ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array))))),
    )
    s_data = pd.DataFrame(s_data, index=data.index, columns=data.columns)

    stand_mean_all = np.dot(stand_mean[:, [0]], np.ones((1, dataAll.shape[1])))
    s_data_all = np.where(
        var_pooled == 0,
        0,
        (
            (dataAll - stand_mean_all)
            / np.dot(np.sqrt(var_pooled), np.ones((1, dataAll.shape[1])))
        ),
    )
    s_data_all = pd.DataFrame(s_data_all, index=dataAll.index, columns=dataAll.columns)

    design_all = _design_matrix(modelAll, batch_key, batch_levels)

    return (
        s_data,
        design,
        var_pooled,
        stand_mean,
        s_data_all,
        design_all,
        stand_mean_all,
    )


def _aprior(delta_hat):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m ** 3) / s2


def _it_sol(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray]:
    """\
    Iteratively compute the conditional posterior means for gamma and delta.

    gamma is an estimator for the additive batch effect, deltat is an estimator
    for the multiplicative batch effect. We use an EB framework to estimate these
    two. Analytical expressions exist for both parameters, which however depend on each other.
    We therefore iteratively evalutate these two expressions until convergence is reached.

    Parameters
    --------
    s_data
        Contains the standardized Data
    g_hat
        Initial guess for gamma
    d_hat
        Initial guess for delta
    g_bar, t_2, a, b
        Hyperparameters
    conv: float, optional (default: `0.0001`)
        convergence criterium

    Returns:
    --------
    gamma
        estimated value for gamma
    delta
        estimated value for delta
    """

    n = (1 - np.isnan(s_data)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0

    # They need to be initialized for numba to properly infer types
    g_new = g_old
    d_new = d_old
    # we place a normally distributed prior on gamma and and inverse gamma prior on delta
    # in the loop, gamma and delta are updated together. they depend on each other. we iterate until convergence.
    while change > conv:
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        sum2 = s_data - g_new.reshape((g_new.shape[0], 1)) @ np.ones(
            (1, s_data.shape[1])
        )
        sum2 = sum2 ** 2
        sum2 = sum2.sum(axis=1)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

        change = max(
            (abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max()
        )
        g_old = g_new  # .copy()
        d_old = d_new  # .copy()
        count = count + 1

    return g_new, d_new


def normalizeByCombatFocuesOnMock(adata, batchKey, normlizeKey, mockName, layer=None):
    from .singleCellTools import basic

    key = batchKey
    covariates = None
    if not layer:
        layer = "X"
    adataOrg = adata
    adata = basic.getPartialLayersAdata(adata, layer, [key, normlizeKey])
    if issparse(adata.X):
        adata.X = adata.X.A
    adata.obs["isMock"] = adata.obs[normlizeKey] == mockName
    adata._sanitize()

    X = adata.X.T
    dataAll = pd.DataFrame(data=X, index=adata.var_names, columns=adata.obs_names)
    modelAll = adata.obs[[key] + (covariates if covariates else [])]
    batch_info_all = modelAll.groupby(key).indices.values()
    n_batches_all = np.array([len(v) for v in batch_info_all])
    n_batch_all = len(batch_info_all)
    n_array_all = float(sum(n_batches_all))

    adata = adata[adata.obs["isMock"]]
    X = adata.X.T
    data = pd.DataFrame(data=X, index=adata.var_names, columns=adata.obs_names)
    model = adata.obs[[key] + (covariates if covariates else [])]
    batch_info = model.groupby(key).indices.values()
    n_batches = np.array([len(v) for v in batch_info])
    n_batch = len(batch_info)
    n_array = float(sum(n_batches))

    (
        s_data,
        design,
        var_pooled,
        stand_mean,
        s_data_all,
        design_all,
        stand_mean_all,
    ) = _standardize_data(model, data, key, modelAll, dataAll)

    batch_design = design[design.columns[:n_batch]]
    # first estimate of the additive batch effect
    gamma_hat = (
        la.inv(batch_design.T @ batch_design) @ batch_design.T @ s_data.T
    ).values
    delta_hat = []

    # first estimate for the multiplicative batch effect
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(s_data.iloc[:, batch_idxs].var(axis=1))

    # empirically fix the prior hyperparameters
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)
    # a_prior and b_prior are the priors on lambda and theta from Johnson and Li (2006)
    a_prior = list(map(_aprior, delta_hat))
    b_prior = list(map(_bprior, delta_hat))

    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        # temp stores our estimates for the batch effect parameters.
        # temp[0] is the additive batch effect
        # temp[1] is the multiplicative batch effect
        gamma, delta = _it_sol(
            s_data.iloc[:, batch_idxs].values,
            gamma_hat[i],
            delta_hat[i].values,
            gamma_bar[i],
            t2[i],
            a_prior[i],
            b_prior[i],
        )

        gamma_star.append(gamma)
        delta_star.append(delta)

    bayesdata = s_data_all
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    batch_design_all = design_all[design_all.columns[:n_batch]]

    for j, batch_idxs in enumerate(batch_info_all):
        # we basically substract the additive batch effect, rescale by the ratio
        # of multiplicative batch effect to pooled variance and add the overall gene
        # wise mean
        dsq = np.sqrt(delta_star[j, :])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, n_batches_all[j])))
        numer = np.array(
            bayesdata.iloc[:, batch_idxs]
            - np.dot(batch_design_all.iloc[batch_idxs], gamma_star).T
        )
        bayesdata.iloc[:, batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = (
        bayesdata * np.dot(vpsq, np.ones((1, bayesdata.shape[1]))) + stand_mean_all
    )

    adataOrg.X = bayesdata.values.T
