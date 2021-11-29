"""
Normalization tools
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
from ..otherTools import setSeed


def normalizeMultiAd(multiAd, removeAmbiguous=True):
    """
    normalize illumina and nanopore data separately, each cell's sum counts will equals to 3e4
    """
    multiCountAd = multiAd[:, ~multiAd.var.index.str.contains("_")]
    multiOtherAd = multiAd[:, multiAd.var.index.str.contains("_")]
    sc.pp.normalize_total(multiCountAd, target_sum=1e4)
    sc.pp.normalize_total(multiOtherAd, target_sum=2e4)
    multiAd = sc.concat([multiCountAd, multiOtherAd], axis=1)
    if removeAmbiguous:
        multiAd = multiAd[
            :,
            ~(
                multiAd.var.index.str.contains("Ambiguous")
                | multiAd.var.index.str.contains("_N_")
            ),
        ]
    return multiAd


def normalizeByScran(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    logScaleOut: bool = True,
    needNormalizePre: bool = True,
    resolutionPre: float = 0.7,
    clusterInfo: Optional[str] = None,
    copy: bool = False,
) -> anndata.AnnData:
    """
    normalizeByScran: use scran normalize raw counts

    Parameters
    ----------
    adata : anndata.AnnData
        X stores raw counts
    logScaleOut : bool, optional
        log-transform the output or not. Defaults to True.
    needNormalizePre: bool, optional
        wheather need normalize adata.X before pre-clustering, if False, the input adata.X must NOT be log-scaled.
    resolutionPre: float, optional
        the clustering resolution of leiden before input to scran.
    clusterInfo: str, optional
        the column name of clusterInfo which stored in adata.obs
        if set, <resolutionPre> and <needNormalizePre> parameters will be ignored. Default to None
    copy: bool, optional
        Default to False

    Returns
    -------
    anndata.AnnData
        anndata: update scran in adata.layers; update sizeFactors in adata.obs
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot
    from scipy.sparse import csr_matrix, isspmatrix

    R = ro.r
    importr("scran")
    logger.info("Initialization")
    adata = adata.copy() if copy else adata
    layer = "X" if layer is None else layer

    if not clusterInfo:
        adataPP = basic.getPartialLayersAdata(adata, layer)

        if needNormalizePre:
            basic.testAllCountIsInt(adataPP, None)
            sc.pp.normalize_per_cell(adataPP, counts_per_cell_after=1e6)
        else:
            logger.warning(
                "Not perfom normalize step, you should ensure the input data is not log-transformed"
            )

        sc.pp.log1p(adataPP)

        logger.info("pre-clustering")
        sc.pp.pca(adataPP, n_comps=15)
        sc.pp.neighbors(adataPP)
        sc.tl.leiden(adataPP, key_added="groups", resolution=resolutionPre)

        logger.info("transfer data to R")
        inputGroupDf_r = py2r(adataPP.obs["groups"])
    else:
        logger.info("transfer data to R")
        inputGroupDf_r = py2r(adata.obs[clusterInfo])

    se = py2r(adata)

    logger.info("calculate size factor")
    sizeFactorSr_r = R.sizeFactors(
        R.computeSumFactors(
            se,
            clusters=inputGroupDf_r,
            **{"min.mean": 0.1, "assay.type": layer},
        )
    )
    sizeFactorSr = r2py(sizeFactorSr_r).copy()

    logger.info("process result")
    rawMtx = adata.X if layer == "X" else adata.layers[layer]
    rawMtx = rawMtx.A if isspmatrix(rawMtx) else rawMtx
    adata.obs["sizeFactor"] = sizeFactorSr
    adata.layers["scran"] = rawMtx / adata.obs["sizeFactor"].values.reshape([-1, 1])

    basic.setLayerInfo(adata, scran="raw")
    if logScaleOut:
        logger.warning("output is logScaled")
        basic.setLayerInfo(adata, scran="log")
        sc.pp.log1p(adata, layer="scran")

    return adata if copy else None


def normalizeByScranMultiBatchNorm(
    ad: anndata.AnnData,
    batchKey: str,
    layer: Optional[str] = None,
    geneMinCells: int = 1,
    threads: int = 64,
    argsToScran: Dict = {},
    **argsToMultiBatchNorm,
):
    """
    use multiBatchNorm with computeSumFactors to normalize adata

    Parameters
    ----------
    ad : anndata.AnnData
    batchKey : str
        column name
    layer : Optional[str], optional
        must be raw, by default None
    geneMinCells : int, optional
        by default 1
    threads : int, optional
        by default 64
    argsToScran : Dict, optional
        transfer to `normalize.normalizeByScran`, by default {}
    **argsToMultiBatchNorm:
        transfer to `batchelor.multiBatchNorm`

    Returns
    -------
    anndata:
        layers['scranMbn'] will be updated by log-normalized data
    """
    from rpy2.robjects.packages import importr
    from ..rTools import (
        py2r,
        r2py,
        r_set_seed,
    )

    batchelor = importr("batchelor")
    r_set_seed(39)

    if not layer:
        layer = "X"

    adOrg = ad
    logger.info(f"input data shape: {adOrg.shape}")
    ls_ad = list(basic.splitAdata(ad, batchKey))
    if layer != "X":
        for _ad in ls_ad:
            _ad.X = _ad.layers[layer].copy()
    [sc.pp.filter_genes(x, min_cells=geneMinCells) for x in ls_ad]

    if threads > 1:
        with Mtp(threads) as mtp:
            ls_results = []
            for _ad in ls_ad:
                ls_results.append(
                    mtp.submit(
                        normalize.normalizeByScran, _ad, copy=True, **argsToScran
                    )
                )
        ls_results = [x.result() for x in ls_results]
    else:
        ls_results = []
        for _ad in ls_ad:
            ls_results.append(normalize.normalizeByScran(_ad, copy=True, **argsToScran))

    ad = sc.concat(ls_results)
    ad.layers["counts"] = ad.X.copy()

    _ad = basic.getPartialLayersAdata(
        ad, layers=["counts"], obsInfoLs=["sizeFactor", batchKey]
    )
    adR = py2r(_ad)
    adR = batchelor.multiBatchNorm(
        adR, batch="index", normalize_all=True, **argsToMultiBatchNorm
    )
    ad = r2py(adR)
    adOrg = adOrg[:, ad.var.index].copy()
    adOrg.layers["scranMbn"] = ad.layers["logcounts"]
    basic.setLayerInfo(adOrg, scranMbn="log")
    logger.info(f"output data shape: {adOrg.shape}")
    return adOrg


def normalizeBySCT(
    adata: anndata.AnnData,
    layer: Union[Literal["X"], str] = "X",
    regress_out: Sequence[str] = ["log10_umi"],
    method: Literal[
        "theta_ml", "glmgp", "fix-slope", "theta_ml", "theta_lbfgs", "alpha_lbfgs"
    ] = "theta_ml",
    vst_flavor: Literal["v1", "v2"] = "v1",
    res_clip_range: Literal["seurat", "default"] = "seurat",
    batch_key: Optional[str] = None,
    min_cells: int = 5,
    n_top_genes: int = 3000,
    n_genes: int = 2000,
    n_cells: int = 5000,
    correct_counts: bool = True,
    log_scale_correct: bool = False,
    threads: int = 12,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Normalization and variance stabilization of scRNA-seq data using regularized
    negative binomial regression [Hafemeister19]_.
    sctransform uses Pearson residuals from regularized negative binomial regression to
    correct for the sequencing depth. After regressing out total number of UMIs (and other
    variables if given) it ranks the genes based on their residual variances and therefore
    also acts as a HVG selection method.
    This function replaces `sc.pp.normalize_total` and `sc.pp.highly_variable_genes` and requires
    raw counts in `adata.X`.

    Parameters
    ----------
    adata : anndata.AnnData
    layer : Union[Literal[, optional
        raw count, by default "X"
    regress_out : Sequence[str], optional
        by default ["log10_umi"].
    method : Literal[, optional
        Literal["theta_ml", "glmgp", "fix-slope", "theta_ml", "theta_lbfgs", "alpha_lbfgs"], by default "theta_ml"
    vst_flavor : Literal[, optional
        Literal["v1", "v2"], by default "v1". if `v2`, method will force to `fix-slope`
    res_clip_range : Literal[, optional
        Literal["seurat", "default"], by default "seurat"
    batch_key : Optional[str], optional
        Useless now, by default None
    min_cells : int, optional
        by default 5
    n_top_genes : int, optional
        by default 3000
    n_genes : int, optional
        gene counts used for `vst`, by default 2000
    n_cells : int, optional
        cell counts used for `vst`, by default 5000
    correct_counts : bool, optional
        by default True
    log_scale_correct : bool, optional
        by default False
    threads : int, optional
        by default 12
    copy : bool, optional
            by default False

    Returns
    -------
    Optional[anndata.AnnData]
        [description]
    """

    import scipy.sparse as ss
    import pysctransform
    import scanpy as sc
    from scanpy.preprocessing import filter_genes
    import rpy2.robjects as ro

    setSeed()
    layer = "X" if not layer else layer

    # check if observations are unnormalized using first 10
    basic.testAllCountIsInt(adata, layer)

    if copy:
        adata = adata.copy()
    # sctransform only worked on sparse matrix
    if layer == "X":
        if not ss.issparse(adata.X):
            adata.X = ss.csr_matrix(adata.X)
    else:
        if not ss.issparse(adata.layers[layer]):
            adata.layers[layer] = ss.csr_matrix(adata.layers[layer])

    assert regress_out, "regress_out cannot be emtpy"

    filter_genes(adata, min_cells=min_cells)
    n_cells = min(n_cells, len(adata))
    if vst_flavor == "v2":
        method = "fix-slope"
        exclude_poisson = True
    else:
        exclude_poisson = False

    ls_cellAttr = []
    ls_cellAttr.extend([x for x in regress_out if x != "log10_umi"])
    if batch_key:
        ls_cellAttr.append(batch_key)
    df_cellAttr = adata.obs[ls_cellAttr]

    mtx = adata.X if layer == "X" else adata.layers[layer]
    vst_out = pysctransform.vst(
        mtx.T,
        gene_names=adata.var_names.tolist(),
        cell_names=adata.obs_names.tolist(),
        latent_var=regress_out,
        batch_var=batch_key,
        method=method,
        n_cells=n_cells,
        n_genes=n_genes,
        exclude_poisson=exclude_poisson,
        correct_counts=correct_counts,
        cell_attr=df_cellAttr,
        min_cells=min_cells,
        threads=threads,
        verbosity=1,
    )
    residuals = pysctransform.get_hvg_residuals(vst_out, n_top_genes, res_clip_range)

    ro.numpy2ri.deactivate()
    ro.pandas2ri.deactivate()

    adata.layers["sct_residuals"] = vst_out["residuals"].T
    adata.var["highly_variable"] = adata.var.index.isin(residuals.columns)
    if correct_counts:
        adata.layers["sct_corrected"] = vst_out["corrected_counts"].T
        basic.setLayerInfo(adata, sct_corrected="raw")
        if log_scale_correct:
            sc.pp.log1p(adata, layer="sct_corrected")
            basic.setLayerInfo(adata, sct_corrected="log-normalized")
    if copy:
        return adata


def integrateBySeurat(
    ad: anndata.AnnData,
    batch_key,
    n_top_genes=5000,
    layer="raw",
    reduction: Literal["cca", "rpca", "rlsi"] = "cca",
    normalization_method: Literal["LogNormalize", "SCT"] = "LogNormalize",
) -> sc.AnnData:
    """
    Integrate by Seurat [Hafemeister19]_.

    Parameters
    ----------
    ad : anndata.AnnData
    batch_key : str
    n_top_genes : int
    layer : str, must be raw counts

    Notes
    -------
    ad will be updated as following rules:
        ad.obsm['seurat_integrated_data']: integrated log-transformed data
        ad.obs['X_pca_seurat']: PCA of integrated data

    Returns
    -------
    anndata.AnnData: ad_combined
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import ad2so, so2ad
    from cool import F

    rBase = importr("base")
    rUtils = importr("utils")
    R = ro.r
    setSeed()

    importr("Seurat")
    sc.pp.highly_variable_genes(
        ad,
        layer=layer,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
    )
    ls_features = ad.var.loc[ad.var["highly_variable"]].index.to_list() | F(
        lambda z: [x.replace("_", "-") for x in z]
    )  # seurat always use dash to separate gene names
    lsR_features = R.c(*ls_features)

    so = ad2so(ad)

    ro.globalenv["so"] = so
    ro.globalenv["batch_key"] = batch_key
    ro.globalenv["n_top_genes"] = n_top_genes
    ro.globalenv["lsR_features"] = lsR_features
    ro.globalenv["reduction"] = reduction
    ro.globalenv["normalization.method"] = normalization_method

    R(
        """
    so.list <- SplitObject(so, split.by = batch_key)
    """
    )
    if normalization_method == "LogNormalize":
        R(
            """
        so.list <- lapply(X = so.list, FUN = function(x) {
            x <- NormalizeData(x)
        })
        """
        )
    elif normalization_method == "SCT":
        R(
            """
        so.list <- lapply(X = so.list, FUN = SCTransform, method = "glmGamPoi", residual.features = lsR_features)
        so.list <- PrepSCTIntegration(object.list = so.list, anchor.features = lsR_features)
        """
        )

    else:
        assert False, f"unknown normalization method : {normalization_method}"

    R(
        """
    so.anchors <- FindIntegrationAnchors(object.list = so.list, anchor.features = lsR_features, reduction = reduction, normalization.method = normalization.method)
    so.combined <- IntegrateData(anchorset = so.anchors, normalization.method = normalization.method)
    """
    )

    so_combined = R("so.combined")
    ad_combined = so2ad(so_combined)
    sc.pp.scale(ad_combined)
    sc.tl.pca(ad_combined)

    ad.obsm["seurat_integrated_data"] = ad_combined.to_df("integrated_data").copy()
    ad.obsm["X_pca_seurat"] = ad_combined.obsm["X_pca"].copy()
    return ad_combined
