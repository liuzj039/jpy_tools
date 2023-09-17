from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from io import StringIO
from joblib import Parallel, delayed
import sh
import h5py
from tqdm import tqdm
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Sequence,
    Literal,
    Any,
    Tuple,
    Iterator,
    Mapping,
    Callable,
)
import collections
import scipy.sparse as ss

def ad2df(ad:sc.AnnData, layer=None, forceDense = False) -> pd.DataFrame:
    '''> If the data is sparse, return a sparse dataframe, otherwise return a dense dataframe
    
    Parameters
    ----------
    ad : sc.AnnData
        the AnnData object
    layer
        the layer to convert to a dataframe. If None, then the .X layer is used.
    forceDense, optional
        if True, will force the conversion to a dense dataframe.
    
    Returns
    -------
        A pandas dataframe
    
    '''
    if forceDense:
        df = ad.to_df(layer)
    else:
        if layer is None:
            if ss.issparse(ad.X):
                df = pd.DataFrame.sparse.from_spmatrix(ad.X, index = ad.obs.index, columns=ad.var.index)
            else:
                df = ad.to_df()
        else:
            if ss.issparse(ad.layers[layer]):
                df = pd.DataFrame.sparse.from_spmatrix(ad.layers[layer], index = ad.obs.index, columns=ad.var.index)
            else:
                df = ad.to_df(layer)
    return df

def initLayer(ad:sc.AnnData, layer=None, total=1e4, needScale=False, logbase=None):
    """
    overwrite layer: `raw`, `normalize_log`, `normalize_log_scale`, 'X'
    """
    ad.layers['raw'] = ad.X.copy() if layer == None else ad.layers[layer].copy()
    ad.layers['normalize_log'] = ad.layers['raw'].copy()
    sc.pp.normalize_total(ad, total, layer='normalize_log')
    sc.pp.log1p(ad, layer='normalize_log', base=logbase)
    if needScale:
        ad.layers['normalize_log_scale'] = ad.layers['normalize_log'].copy()
        sc.pp.scale(ad, layer='normalize_log_scale')
    ad.X = ad.layers['normalize_log'].copy()

def getOverlap(ad_a: anndata.AnnData, ad_b: anndata.AnnData, copy=False):
    ls_geneOverlap = list(ad_a.var.index & ad_b.var.index)
    logger.info(f"Used Gene Counts: {len(ls_geneOverlap)}")
    if copy:
        return ad_a[:, ls_geneOverlap].copy(), ad_b[:, ls_geneOverlap].copy()
    else:
        return ad_a[:, ls_geneOverlap], ad_b[:, ls_geneOverlap]

def splitAdata(
    adata: anndata.AnnData,
    batchKey: str,
    copy=True,
    axis: Literal[0, "cell", 1, "feature"] = 0,
    needName=False,
    disableBar=False
) -> Iterator[anndata.AnnData]:
    '''It splits an AnnData object into multiple AnnData objects based on a key in the obs or var dataframe

    Parameters
    ----------
    adata : anndata.AnnData
        the AnnData object you want to split
    batchKey : str
        the column name in adata.obs or adata.var that contains the batch information
    copy, optional
        whether to copy the adata object. If you're going to modify the adata object, you should set this to True.
    axis : Literal[0, "cell", 1, "feature"], optional
        0 or "cell" for splitting cells, 1 or "feature" for splitting features
    needName, optional
        if True, the function will return a tuple of (batchName, adata)
    disableBar, optional
        Whether to disable the progress bar.

    '''
    if axis in [0, "cell"]:
        assert batchKey in adata.obs.columns, f"{batchKey} not detected in adata"
        indexName = "index" if (not adata.obs.index.name) else adata.obs.index.name
        adata.obs["__group"] = adata.obs[batchKey]
        batchObsLs = (
            adata.obs.filter(["__group"])
            .reset_index()
            .groupby("__group")[indexName]
            .agg(list)
        )
        for batchObs in tqdm(batchObsLs, disable=disableBar):
            if needName:
                if copy:
                    yield adata[batchObs].obs.iloc[0].loc["__group"], adata[
                        batchObs
                    ].copy()
                else:
                    yield adata[batchObs].obs.iloc[0].loc["__group"], adata[batchObs]
            else:
                if copy:
                    yield adata[batchObs].copy()
                else:
                    yield adata[batchObs]

    elif axis in [1, "feature"]:
        assert batchKey in adata.var.columns, f"{batchKey} not detected in adata"
        indexName = "index" if (not adata.var.index.name) else adata.var.index.name
        adata.var["__group"] = adata.var[batchKey]
        batchVarLs = (
            adata.var.filter(["__group"])
            .reset_index()
            .groupby("__group")[indexName]
            .agg(list)
        )
        del adata.var["__group"]
        for batchVar in tqdm(batchVarLs, disable=disableBar):
            if needName:
                if copy:
                    yield adata[batchVar].var.iloc[0].loc["__group"], adata[
                        batchVar
                    ].copy()
                else:
                    yield adata[batchVar].var.iloc[0].loc["__group"], adata[batchVar]
            else:
                if copy:
                    yield adata[:, batchVar].copy()
                else:
                    yield adata[:, batchVar]

    else:
        assert False, "Unknown `axis` parameter"

def testAllCountIsInt(adata: anndata.AnnData, layer: Optional[str] = None) -> bool:
    """
    Test whether all counts is int
    """
    import scipy.sparse as sp

    if layer == "X":
        layer = None

    testColCounts = min([10, adata.shape[0]])
    if not layer:
        X_subset = adata.X[:testColCounts]
    else:
        X_subset = adata.layers[layer][:testColCounts]

    err = (
        "Make sure that adata.layer contains unnormalized count data"
        + f"\tLayer:{layer}"
    )
    if sp.issparse(X_subset):
        assert (X_subset.astype(int) != X_subset).nnz == 0, err
    else:
        assert np.all(X_subset.astype(int) == X_subset), err


def getPartialLayersAdata(
    adata: anndata.AnnData,
    layers: Optional[Union[str, List[str]]] = None,
    obsInfoLs: Optional[Sequence[str]] = None,
    varInfoLs: Optional[Sequence[str]] = None,
) -> anndata.AnnData:
    """
    get a subset of adata. Only contains one layer expression matrix, and several obs information.

    Parameters
    ----------
    adata : anndata.AnnData
    layers : Optional[Union[str, List[str]]], optional
        None will be parsed as 'X', by default None
    obsInfoLs : Optional[Sequence[str]], optional
        by default None
    varInfoLs : Optional[Sequence[str]], optional
        by default None

    Returns
    -------
    anndata.AnnData
        if data type of `layers` is list, all element in 'X' of returned adata will be set as 0
    """
    import scipy.sparse as ss

    if not obsInfoLs:
        obsInfoLs = []
    else:
        assert sum([x in adata.obs.columns for x in obsInfoLs]) == len(
            obsInfoLs
        ), "Requested feature not located in adata.obs"
    if not varInfoLs:
        varInfoLs = []
    else:
        assert sum([x in adata.var.columns for x in varInfoLs]) == len(
            varInfoLs
        ), "Requested feature not located in adata.var"

    if not layers:
        layers = "X"
    if isinstance(layers, list):
        dt_layerMtx = {}
        for layer in layers:
            ar_mtx = adata.X if layer == "X" else adata.layers[layer]
            dt_layerMtx[layer] = ar_mtx
        subAd = anndata.AnnData(
            ss.csr_matrix(np.zeros(adata.shape)),
            adata.obs[obsInfoLs],
            adata.var[varInfoLs],
            layers=dt_layerMtx,
        )

    elif isinstance(layers, str):
        layer = layers
        mtxAr = adata.X if layer == "X" else adata.layers[layer]
        subAd = anndata.AnnData(mtxAr, adata.obs[obsInfoLs], adata.var[varInfoLs])

    else:
        assert False, f"unsupported layers data type: {type(layers)}"

    return subAd.copy()

def setadataColor(adata, label, colorDt=None, hex=True):
    adata.obs[label] = adata.obs[label].astype("category")
    if colorDt:
        if not hex:
            from matplotlib.colors import to_hex

            colorDt = {x: to_hex(y) for x, y in colorDt.items()}

        _dt = getadataColor(adata, label)
        _dt.update(colorDt)
        colorDt = _dt
        adata.uns[f"{label}_colors"] = [
            colorDt[x] for x in adata.obs[label].cat.categories
        ]
    else:
        if f"{label}_colors" not in adata.uns:
            sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)

    return adata

def getadataColor(adata, label):
    if f"{label}_colors" not in adata.uns:
        setadataColor(adata, label)
    return {
        x: y
        for x, y in zip(adata.obs[label].cat.categories, adata.uns[f"{label}_colors"])
    }

def mergeData(ad, obsKey, layer="raw"):
    testAllCountIsInt(ad, layer)
    ls_keyProduct = ad.obs[obsKey].value_counts().sort_index().index.to_list()
    if isinstance(obsKey, str):
        ad.obs["temp_merge"] = ad.obs[obsKey].copy()
        obsKey = [obsKey]
    else:
        ad.obs["temp_merge"] = ad.obs[obsKey].apply(lambda x: tuple(x), axis=1)
    lsSr_onehot = []
    for col in ls_keyProduct:
        if isinstance(col, str):
            sr_col = (
                pd.Series(index=ad.obs.index, name="||".join([str(col)]))
                .fillna(0)
                .astype(int)
            )
        else:
            sr_col = (
                pd.Series(index=ad.obs.index, name="||".join([str(x) for x in col])).fillna(0).astype(int)
            )
        sr_col.where(ad.obs["temp_merge"] != col, 1, inplace=True)
        lsSr_onehot.append(sr_col)
    df_oneHot = pd.concat(lsSr_onehot, axis=1)
    ad_merge = sc.AnnData(
        df_oneHot.values.T @ ad.layers[layer],
        obs=pd.DataFrame(index=df_oneHot.columns),
        var=pd.DataFrame(index=ad.var.index),
    )
    ad_merge.obs = ad_merge.obs.index.to_series().str.split("\|\|", expand=True)
    ad_merge.obs.columns = obsKey
    del ad.obs["temp_merge"]
    return ad_merge

def getMetaCells(
    ad, ls_obs, layer="raw", skipSmallGroup=True, target_metacell_size=5e4, **kwargs
):
    """
    get meta-cell from adata
    """
    import metacells as mc

    if isinstance(ls_obs, str):
        ls_obs = [ls_obs]
    dtAd_meta = {}
    ls_changeObs = []
    for ls_label, df in ad.obs.groupby(ls_obs):
        if isinstance(ls_label, str):
            ls_label = [ls_label]

        ad_sub = ad[df.index].copy()
        ad_sub.X = ad_sub.layers[layer].copy()
        logger.info(f"{ls_label} info:: {len(ad_sub)} cells, {ad_sub.X.sum()} UMIs")

        if ad_sub.X.sum() / 2 < target_metacell_size:
            ls_changeObs.append(ls_label)
            if skipSmallGroup:
                logger.warning(f"{ls_label} is too small, skip")
                continue
            else:
                _target_metacell_size = ad_sub.X.sum() // 2
                logger.warning(
                    f"{ls_label} is too small, set target_metacell_size to {_target_metacell_size}"
                )
                mc.pl.divide_and_conquer_pipeline(
                    ad_sub, target_metacell_size=_target_metacell_size, **kwargs
                )
        else:
            mc.pl.divide_and_conquer_pipeline(
                ad_sub, target_metacell_size=target_metacell_size, **kwargs
            )
        ad_subMeta = mc.pl.collect_metacells(ad_sub, name="metacells")
        dt_metaLinkWithOrg = (
            ad_sub.obs["metacell"]
            .groupby(ad_sub.obs["metacell"])
            .apply(lambda sr: sr.index.to_list())
            .to_dict()
        )
        ad_subMeta.obs["metacell_link"] = ad_subMeta.obs.index.map(
            lambda x: dt_metaLinkWithOrg[int(x)]
        )

        # ad_sub.obs.index = ad_sub.obs.index.astype(str) + '-' + '-'.join(ls_label)
        for obsName, label in zip(ls_obs, ls_label):
            ad_subMeta.obs[obsName] = label
        ad_subMeta.layers[layer] = ad_subMeta.X.copy()
        dtAd_meta["-".join(ls_label)] = ad_subMeta
    ad_meta = sc.concat(dtAd_meta, index_unique="-")
    for obsName in ls_obs:
        if isinstance(ad.obs[obsName].dtype, pd.CategoricalDtype):
            ad_meta.obs[obsName] = (
                ad_meta.obs[obsName]
                .astype("category")
                .cat.reorder_categories(ad.obs[obsName].cat.categories)
            )
    ad_meta.obs["metacell_link"] = ad_meta.obs["metacell_link"].str.join(",")
    print(ad_meta.obs[ls_obs].value_counts())
    print(f"These {ls_changeObs} groups are skipped or changed due to small size")
    sns.histplot(ad_meta.to_df(layer).sum(1))
    plt.xlabel("metacell UMI counts")
    plt.show()

    sns.histplot(ad_meta.obs["metacell_link"].str.split(",").map(len))
    plt.xlabel("metacell cell counts")
    plt.show()
    return ad_meta