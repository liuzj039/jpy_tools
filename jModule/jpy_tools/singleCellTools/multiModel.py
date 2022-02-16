"""
multiModle tools
"""
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
from concurrent.futures import ProcessPoolExecutor as Mtp
from concurrent.futures import ThreadPoolExecutor as MtT
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
from xarray import corr
from . import basic

def addDfToObsm(adata, copy=False, **dataDt):
    """addDfToObsm, add data to adata.obsm

    Args:
        adata ([anndata])
        copy (bool, optional)
        dataDt: {label: dataframe}, dataframe must have the same dimension

    Returns:
        adata if copy=True, otherwise None
    """
    adata = adata.copy() if copy else adata
    for label, df in dataDt.items():
        if (adata.obs.index != df.index).all():
            logger.error(f"dataset {label} have a wrong shape/index")
            0 / 0
        if label in adata.obsm:
            logger.warning(f"dataset {label} existed! Overwrite")
        adata.uns[f"{label}_label"] = df.columns.values
        adata.obsm[label] = df.values
    if copy:
        return adata


def getMatFromObsm(
    adata: anndata.AnnData,
    keyword: str,
    minCell: int = 5,
    useGeneLs: Union[list, pd.Series, np.ndarray] = [],
    normalize=True,
    logScale=True,
    ignoreN=False,
    clear=False,
    raw=False,
    strCommand=None,
) -> anndata.AnnData:
    """
    use MAT deposited in obsm replace the X MAT

    params:
        adata:
            version 1.0 multiAd
        keyword:
            stored in obsm
        minCell:
            filter feature which expressed not more than <minCell> cells
        useGeneLs:
            if not specified useGeneLs, all features will be output, otherwise only features associated with those gene will be output
        normalize:
            normalize the obtained Mtx or not
        logScale:
            log-transformed or not
        ignoreN:
            ignore ambiguous APA/Splice info
        clear:
            data not stored in obs or var will be removed
        raw:
            return the raw dataset stored in the obsm. This parameter is prior to all others
        strCommand:
            use str instead of specified params:
            "n": set normalize True
            "s": set logScale True
            'N': set ignoreN True
            'c' set clear True
            '': means all is False
            This parameter is prior to all others except raw
    return:
        anndata
    """
    if clear:
        transformedAd = anndata.AnnData(
            X=adata.obsm[keyword].copy(),
            obs=adata.obs,
            var=pd.DataFrame(index=adata.uns[f"{keyword}_label"]),
        )
    else:
        transformedAd = anndata.AnnData(
            X=adata.obsm[keyword].copy(),
            obs=adata.obs,
            var=pd.DataFrame(index=adata.uns[f"{keyword}_label"]),
            obsp=adata.obsp,
            obsm=adata.obsm,
            uns=adata.uns,
        )

    if raw:
        return transformedAd

    if strCommand != None:
        normalize = True if "n" in strCommand else False
        logScale = True if "s" in strCommand else False
        ignoreN = True if "N" in strCommand else False
        clear = True if "c" in strCommand else False

    logger.info(
        f"""
    final mode: 
        normalize: {normalize}, 
        logScale: {logScale}, 
        ignoreN: {ignoreN}, 
        clear: {clear}
    """
    )

    sc.pp.filter_genes(transformedAd, min_cells=minCell)

    if normalize:
        sc.pp.normalize_total(transformedAd, target_sum=1e4)
    if logScale:
        sc.pp.log1p(transformedAd)

    useGeneLs = list(useGeneLs)
    if not useGeneLs:
        transformedAd = transformedAd
    else:
        transformedAdFeatureSr = transformedAd.var.index
        transformedAdFeatureFilterBl = (
            transformedAdFeatureSr.str.split("_").str[0].isin(useGeneLs)
        )
        transformedAd = transformedAd[:, transformedAdFeatureFilterBl]

    if ignoreN:
        transformedAdFeatureSr = transformedAd.var.index
        transformedAdFeatureFilterBl = (
            ~transformedAdFeatureSr.str.split("_").str[1].isin(["N", "Ambiguous"])
        )

        transformedAd = transformedAd[:, transformedAdFeatureFilterBl]

    return transformedAd


def transformEntToAd(ent) -> anndata.AnnData:
    """
    parse mofa
    transformEntToAd parse trained ent object to anndata

    Args:
        ent ([entry_point]): only one group

    Returns:
        anndata: the X represents the sample-factor weights,
                the layer represents the feature-factor weight and variance-factor matrix,
                the uns['mofaR2_total] stored the total variance of factors could be explained
    """
    factorOrderLs = np.argsort(
        np.array(ent.model.calculate_variance_explained()).sum(axis=(0, 1))
    )[::-1]

    sampleWeightDf = pd.DataFrame(ent.model.getExpectations()["Z"]["E"]).T
    sampleWeightDf = sampleWeightDf.reindex(factorOrderLs).reset_index(drop=True)
    sampleWeightDf.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
    sampleWeightDf.columns = ent.data_opts["samples_names"][0]
    mofaAd = basic.creatAnndataFromDf(sampleWeightDf)

    for label, featureSr, data in zip(
        ent.data_opts["views_names"],
        ent.data_opts["features_names"],
        ent.model.getExpectations()["W"],
    ):
        df = pd.DataFrame(data["E"]).T
        featureSr = pd.Series(featureSr)
        featureSr = featureSr.str.rstrip(label)
        if label in ["APA", "fullySpliced"]:
            featureSr = featureSr + label
        df.columns = featureSr
        df = df.reindex(factorOrderLs).reset_index(drop=True)
        df.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
        addDfToObsm(mofaAd, **{label: df})

    r2Df = pd.DataFrame(ent.model.calculate_variance_explained()[0]).T
    r2Df = r2Df.reindex(factorOrderLs).reset_index(drop=True)
    r2Df.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
    r2Df.columns = ent.data_opts["views_names"]
    addDfToObsm(mofaAd, mofaR2=r2Df)

    mofaAd.uns["mofaR2_total"] = {
        x: y
        for x, y in zip(
            ent.data_opts["views_names"],
            ent.model.calculate_variance_explained(True)[0],
        )
    }
    return mofaAd
