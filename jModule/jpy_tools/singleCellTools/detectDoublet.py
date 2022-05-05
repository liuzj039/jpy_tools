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

def byDoubletFinder(
    adata: anndata.AnnData, copy: bool = False, doubletRatio: float = 0.075
) -> Optional[anndata.AnnData]:
    """
    use doubletFinder detect doublets.


    Args:
        adata (anndata.AnnData): X must is raw counts
        copy (bool, optional): copy adata or not. Defaults to False.
        doubletRatio (float, optional): expected doublet ratio. Defaults to 0.075.

    Returns:
        Optional[anndata.AnnData]: anndata if copy
    """
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri
    import anndata2ri
    import rpy2.robjects as ro

    adata = adata.copy() if copy else adata
    logger.info("start to transfer adata to R")

    with localconverter(anndata2ri.converter):
        ro.globalenv["adata"] = adata
    logger.info("start to preprocess adata")
    ro.r(
        f"""
    library(Seurat)
    library(DoubletFinder)
    seuratObj <- as.Seurat(adata, counts="X", data = NULL)
    seuratObj <- SCTransform(seuratObj, )
    seuratObj <- RunPCA(seuratObj)
    seuratObj <- RunUMAP(seuratObj, dims = 1:10)
    seuratObj <- FindNeighbors(seuratObj, dims = 1:10)
    seuratObj <- FindClusters(seuratObj, resolution = 0.6)
    1
    """
    )
    logger.info("start to calculate DF parameters")
    ro.r(
        f"""
    sweep.res.seuratObj <- paramSweep_v3(seuratObj, PCs = 1:10, sct = TRUE)
    sweep.stats.seuratObj <- summarizeSweep(sweep.res.seuratObj, GT = FALSE)
    annotationsDf <- seuratObj@meta.data$seurat_clusters
    homotypic.prop <- modelHomotypic(annotationsDf)
    nExp_poi <- round({doubletRatio}*nrow(seuratObj@meta.data)) 
    nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
    1
    """
    )
    logger.info("start to calculate doublet score")
    ro.r(
        f"""
    seuratObj <- doubletFinder_v3(seuratObj, PCs = 1:10, pN = 0.25, pK = 0.09, nExp = nExp_poi, reuse.pANN = FALSE, sct = TRUE)
    seuratObj <- doubletFinder_v3(seuratObj, PCs = 1:10, pN = 0.25, pK = 0.09, nExp = nExp_poi.adj, reuse.pANN = paste('pANN_0.25_0.09_', nExp_poi, sep=''), sct = TRUE)
    1
    """
    )
    logger.info("start to intergrate result with adata")
    with localconverter(ro.default_converter + pandas2ri.converter):
        finalDf = ro.r("seuratObj@meta.data")
    colNameSr = list(
        ro.r(
            "c(paste('DF.classifications_0.25_0.09_', nExp_poi, sep=''), paste('DF.classifications_0.25_0.09_', nExp_poi.adj, sep=''))"
        )
    )
    finalDf = finalDf.filter(colNameSr).rename(
        {
            x: y
            for x, y in zip(
                colNameSr, ["doubletFinder_raw", "doubletFinder_adjusted"]
            )
        },
        axis=1,
    )
    adata.obs = adata.obs.join(finalDf.copy(deep=True))

    if copy:
        return adata


def byScDblFinder(
    adata: anndata.AnnData,
    layer: str = "X",
    copy: bool = False,
    batch_key: Optional[str] = None,
    doubletRatio: Optional[float] = None,
    skipCheck: bool = False,
    dropDoublet: bool = True,
) -> Optional[anndata.AnnData]:
    """
    use ScDblFinder detect doublets.

    Parameters
    ----------
    adata : anndata.AnnData
        anndata
    layer : str, optional
        use this layer. must be raw counts. Defaults to X
    copy : bool, optional
        copy adata or not. Defaults to False.
    doubletRatio : float, optional
        expected doublet ratio. Defaults to 0.1

    Returns
    -------
    Optional[anndata.AnnData]
        anndata if copy
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_set_seed, ad2so

    r_set_seed(39)
    R = ro.r
    Seurat = importr('Seurat')

    ls_obsInfo = []
    if not batch_key:
        batch_key = R("NULL")
    else:
        ls_obsInfo.append(batch_key)
    if not doubletRatio:
        doubletRatio = R("NULL")

    scDblFinder = importr("scDblFinder")

    if not skipCheck:
        basic.testAllCountIsInt(adata, layer)

    tempAd = basic.getPartialLayersAdata(adata, layer, obsInfoLs=ls_obsInfo)
    tempAd.layers["counts"] = tempAd.X

    logger.info("start to transfer adata to R")
    so = ad2so(tempAd, layer='counts')
    tempAdr = Seurat.as_SingleCellExperiment(so)
    del tempAd
    del so

    logger.info("start to calculate doublet score")

    tempAdr = scDblFinder.scDblFinder(tempAdr, samples=batch_key, dbr=doubletRatio)

    logger.info("start to intergrate result with adata")
    scDblFinderResultDf = r2py(tempAdr.slots["colData"])

    adata.obsm["scDblFinder"] = (
        scDblFinderResultDf.reindex(adata.obs.index)
        .filter(regex=r"^scDblFinder[\w\W]*")
        .copy(deep=True)
    )
    adata.obsm["scDblFinder"].columns = adata.obsm["scDblFinder"].columns.astype(
        str
    )

    if dropDoublet:
        logger.info(f"before filter: {len(adata)}")
        adata._inplace_subset_obs(
            adata.obsm["scDblFinder"]["scDblFinder.class"] == "singlet"
        )
        logger.info(f"after filter: {len(adata)}")

    if copy:
        return adata