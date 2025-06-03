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

# from xarray import corr
# import muon as mu
from . import basic
from ..otherTools import setSeed
from ..rTools import rcontext


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
    threads=1,
    calculate=True,
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
    bp = importr("BiocParallel")
    logger.info("Initialization")
    adata = adata.copy() if copy else adata
    layer = "X" if layer is None else layer
    if threads == 1:
        BPPARAM = bp.SerialParam()
    else:
        BPPARAM = bp.MulticoreParam(threads)

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

    # se = py2r(adata)

    mtx_R = py2r(adata.layers[layer].T)
    logger.info("calculate size factor")
    sizeFactorSr_r = R.calculateSumFactors(
        mtx_R, clusters=inputGroupDf_r, BPPARAM=BPPARAM, **{"min.mean": 0.1}
    )
    # sizeFactorSr_r = R.sizeFactors(
    #     R.computeSumFactors(
    #         se,
    #         clusters=inputGroupDf_r,
    #         **{"min.mean": 0.1, "assay.type": layer},
    #     )
    # )
    sizeFactorSr = r2py(sizeFactorSr_r).copy()

    logger.info("process result")
    adata.obs["sizeFactor"] = sizeFactorSr
    if calculate:
        rawMtx = adata.X if layer == "X" else adata.layers[layer]
        rawMtx = rawMtx.A if isspmatrix(rawMtx) else rawMtx
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


@rcontext
def normalizeBySCT_r(
    ad,
    *,
    layer="raw",
    nTopGenes=3000,
    ls_gene=None,
    vstFlavor="v2",
    returnOnlyVarGenes=False,
    doCorrectUmi=True,
    returnMuon=False,
    returnSo=False,
    rEnv=None,
    debug=False,
    runSctOnly=False,
    **dt_kwargsToSct,
):
    """`normalizeBySCT_r` is a function that takes in a single-cell RNA-seq dataset and returns a
    normalized version of the dataset.

    Parameters
    ----------
    ad
        AnnData object
    layer, optional
        the layer to use for normalization. By default, "raw" is used.
    nTopGenes, optional
        number of genes to use for normalization
    ls_gene
        list of genes to use for normalization. If None, use the top nTopGenes genes.
    vstFlavor, optional
        "v2" or "v1"
    returnOnlyVarGenes, optional
        if True, return only the variable genes
    doCorrectUmi, optional
        whether to correct for UMI counts
    returnMuon, optional
        if True, return the muon object.

    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import pickle
    from ..rTools import ad2so, so2ad, so2md
    from ..otherTools import F

    importr("DescTools")
    rBase = importr("base")
    rUtils = importr("utils")
    Seurat = importr("Seurat")
    magrittr = importr("magrittr")

    R = ro.r
    setSeed()

    def setSctVstToAd(ad, rEnv):
        model = R("levels(x = so_sct[['SCT']])")
        assert len(list(model)) == 1, "model must be 1"
        R(
            """
        SCTModel_to_vst <- function(SCTModel) {
            feature.params <- c("theta", "(Intercept)",  "log_umi")
            feature.attrs <- c("residual_mean", "residual_variance" )
            vst_out <- list()
            vst_out$model_str <- slot(object = SCTModel, name = "model")
            vst_out$model_pars_fit <- as.matrix(x = slot(object = SCTModel, name = "feature.attributes")[, feature.params])
            vst_out$gene_attr <- slot(object = SCTModel, name = "feature.attributes")[, feature.attrs]
            vst_out$cell_attr <- slot(object = SCTModel, name = "cell.attributes")
            vst_out$arguments <- slot(object = SCTModel, name = "arguments")
            return(vst_out)
            }

        model = levels(x = so_sct[['SCT']])

        clip_range = SCTResults(object = so_sct[["SCT"]], slot = "clips", model = "model1")$sct
        vst_out <- SCTModel_to_vst(SCTModel = slot(object = so_sct[['SCT']], name = "SCTModel.list")[[model]])
        """
        )
        vst_out = rEnv["vst_out"]
        ad.uns["sct_vst_pickle"] = str(pickle.dumps(vst_out))
        ad.uns["sct_clip_range"] = list(rEnv["clip_range"])

    so = ad2so(ad, layer=layer, lightMode=True)
    if debug:
        import pdb

        pdb.set_trace()
    dt_kwargsToSct["variable.features.n"] = nTopGenes
    dt_kwargsToSct["residual.features"] = (
        R("NULL") if ls_gene is None else R.c(*[x.replace("_", "-") for x in ls_gene])
    )
    dt_kwargsToSct["vst.flavor"] = vstFlavor
    dt_kwargsToSct["return.only.var.genes"] = returnOnlyVarGenes
    dt_kwargsToSct["do.correct.umi"] = doCorrectUmi
    dtR_kwargsToSct = R.list(**dt_kwargsToSct)

    rEnv["so"] = so
    rEnv["dtR_kwargsToSct"] = dtR_kwargsToSct
    R(
        """
    dtR_kwargsToSct$object = so
    so_sct = DoCall(SCTransform, dtR_kwargsToSct)
    ls_hvg <- so_sct[['SCT']]@var.features
    """
    )
    so_sct = rEnv["so_sct"]
    setSctVstToAd(ad, rEnv)
    ls_hvg = list(rEnv["ls_hvg"])

    ls_var = R("VariableFeatures")(so_sct, assay="SCT") >> F(list)
    dt_var = {ls_var[i]: i for i in range(len(ls_var))}
    ad.var["highly_variable"] = ad.var.index.isin(dt_var)
    ad.var["highly_variable_rank"] = ad.var.index.map(lambda x: dt_var.get(x, np.nan))
    if runSctOnly:
        return ad

    md_sct = so2md(so_sct)
    md_sct["SCT_scale.data"].var["highly_variable"] = md_sct[
        "SCT_scale.data"
    ].var.index.isin(ls_hvg)
    md_sct["SCT_scale.data"].X = md_sct["SCT_scale.data"].layers["SCT_scale.data"]
    md_sct.uns["sct_vst_pickle"] = ad.uns["sct_vst_pickle"]
    md_sct.uns["sct_clip_range"] = ad.uns["sct_clip_range"]

    sc.tl.pca(md_sct["SCT_scale.data"])
    ad.obsm["X_pca_sct"] = md_sct["SCT_scale.data"].obsm["X_pca"].copy()
    ad.uns["pca_sct"] = md_sct["SCT_scale.data"].uns["pca"].copy()
    ad.obsm["SCT_data"] = md_sct["SCT"].layers["SCT_data"]
    ad.uns["SCT_data_features"] = md_sct["SCT"].var.index.to_list()

    # ad_sct = so2ad(so_sct, verbose=0)
    # ad_sct.var['highly_variable'] = ad_sct.var.index.isin(ls_hvg)
    # # ad_sct = ad_sct[:, ad_sct.obsm['SCT_scale.data'].columns]
    # # ad_sct.X = ad_sct.obsm['SCT_scale.data']
    # ad_sctForPca = sc.AnnData(ad_sct.obsm['SCT_scale.data'])
    # ad_sctForPca.var['highly_variable'] = ad_sctForPca.var.index.isin(ls_hvg)
    # sc.tl.pca(ad_sctForPca)
    # ad.obsm['X_pca_sct'] = ad_sctForPca.obsm['X_pca'].copy()
    # ad.uns['pca_sct'] = ad_sctForPca.uns['pca'].copy()

    # ad.var['highly_variable'] = ad.var.index.isin(ls_hvg)
    # ad.obsm['SCT_data'] = ad_sct.layers['SCT_data']
    # ad.uns['SCT_data_features'] = ad_sct.var.index.to_list()
    # ad.obsm['SCT_scale.data'] = ad_sct.obsm['SCT_scale.data']

    if returnMuon:
        logger.warning(f"Deprecated")
    if returnSo:
        return so_sct
    else:
        return md_sct


def getHvgGeneFromSctAdata(ls_ad, nTopGenes=3000, nTopGenesEachAd=3000):
    """> get the top  HVGs that are shared across all adatas

    Parameters
    ----------
    ls_ad
        a list of adata objects, must be `data` slot. Gene listed in all adata objects will be used.
    nTopGenes, optional
        the total number of genes to use for the analysis
    nTopGenesEachAd, optional
        the number of HVGs to use from each adata object

    Returns
    -------
        A list of genes that are highly variable across all adatas.

    """
    import pickle

    for ad in ls_ad:
        assert (
            "highly_variable_rank" in ad.var.columns
        ), "adata must have highly_variable_rank"
    ls_allGenes = []
    for ad in ls_ad:
        vst_out = pickle.loads(eval(ad.uns["sct_vst_pickle"]))
        ls_allGenes.extend(list(vst_out[2].rownames))
    ls_allGenes = (
        pd.Series(ls_allGenes)
        .value_counts()
        .loc[lambda x: x == len(ls_ad)]
        .index.to_list()
    )

    ls_allHvg = []
    for ad in ls_ad:
        ls_allHvg.extend(
            ad.var.sort_values("highly_variable_rank")
            .loc[lambda _: _.highly_variable]
            .index[:nTopGenesEachAd]
            .to_list()
        )
    ls_allHvg = [x for x in ls_allHvg if x in ls_allGenes]
    assert (
        len(set(ls_allHvg)) > nTopGenes
    ), "nTopGenes must be smaller than total number of HVGs"
    ls_hvgCounts = pd.Series(ls_allHvg).value_counts()

    ls_usedHvg = []
    for hvgCounts in range(len(ls_ad), 0, -1):
        ls_currentCountsHvg = ls_hvgCounts[ls_hvgCounts == hvgCounts].index.to_list()
        if (len(ls_usedHvg) + len(ls_currentCountsHvg)) > nTopGenes:
            break
        ls_usedHvg.extend(ls_currentCountsHvg)

    needAnotherCounts = nTopGenes - len(ls_usedHvg)
    df_remainGeneRank = pd.DataFrame(index=list(set(ls_allGenes) - set(ls_usedHvg)))
    for i, ad in enumerate(ls_ad):
        df_remainGeneRank[f"{i}"] = ad.var["highly_variable_rank"]
    df_remainGeneRank = df_remainGeneRank.sort_index()
    df_remainGeneRank["count"] = pd.notna(df_remainGeneRank).sum(1)
    df_remainGeneRank["median"] = df_remainGeneRank.drop(columns="count").apply(
        "median", axis=1
    )
    df_remainGeneRank = df_remainGeneRank.sort_values(
        ["count", "median"], ascending=[False, True]
    )
    ls_usedHvg.extend(df_remainGeneRank.iloc[:needAnotherCounts].index.to_list())
    return ls_usedHvg, df_remainGeneRank


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
    gene_attr = vst_out["gene_attr"]
    adata.var["sct_residuals"] = gene_attr["residual_variance"].reindex(adata.var_names)
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


def getSctResiduals(
    ad, ls_gene, layer="raw", forceOverwrite=False, sctVstPickle=None, sctClipRange=None
):
    import pickle
    from ..rTools import py2r, r2py
    from ..otherTools import F

    basic.testAllCountIsInt(ad, layer)
    if sctVstPickle is None:
        assert "sct_vst_pickle" in ad.uns, "sct_vst_pickle not found in adata.uns"
        sctVstPickle = ad.uns["sct_vst_pickle"]
    if sctClipRange is None:
        assert "sct_clip_range" in ad.uns, "sct_clip_range not found in adata.layers"
        sctClipRange = ad.uns["sct_clip_range"]

    import rpy2.robjects as ro

    R = ro.r

    if ("sct_residual" not in ad.obsm.keys()) or forceOverwrite:
        ls_gene = ls_gene
    else:
        if isinstance(ad.obsm["sct_residual"], pd.DataFrame):
            ls_genePrev = list(ad.obsm["sct_residual"].columns)
        else:
            ls_genePrev = ad.uns['sct_residual_keys']
        ls_geneRaw = ls_gene
        ls_gene = list(set(ls_gene) - set(ls_genePrev))
    if len(ls_gene) == 0:
        return None

    fcR_getResiduals = R("sctransform::get_residuals")
    vst_out = pickle.loads(eval(sctVstPickle))
    ls_clipRange = [float(x) for x in list(sctClipRange)]
    df_residuals = (
        fcR_getResiduals(
            vst_out,
            umi=ad[:, ls_gene].to_df(layer).T
            >> F(py2r)
            >> F(R("data.matrix"))
            >> F(R("Matrix::Matrix")),
            res_clip_range=R.c(*ls_clipRange),
        )
        >> F(R("as.data.frame"))
        >> F(r2py)
    )
    df_residuals = df_residuals.T
    df_residuals = df_residuals - df_residuals.mean()
    if ("sct_residual" not in ad.obsm.keys()) or forceOverwrite:
        ad.obsm["sct_residual"] = df_residuals
    else:
        if isinstance(ad.obsm["sct_residual"], pd.DataFrame):
            df_prevRes = ad.obsm["sct_residual"]
        else:
            df_prevRes = pd.DataFrame(
                ad.obsm["sct_residual"], columns=ad.uns['sct_residual_keys'], index=ad.obs_names
            )

        df_residuals = pd.concat(
            [df_prevRes, df_residuals], axis=1
        )
        df_residuals = df_residuals[ls_geneRaw]
        ad.obsm["sct_residual"] = df_residuals.values
        ad.uns['sct_residual_keys'] = ls_geneRaw


@rcontext
def integrateBySeurat(
    ad: anndata.AnnData,
    batch_key,
    n_top_genes=5000,
    layer="raw",
    reduction: Literal["cca", "rpca", "rlsi"] = "cca",
    normalization_method: Literal["LogNormalize", "SCT"] = "LogNormalize",
    k_score=30,
    dims=30,
    k_filter=200,
    k_weight=100,
    identify_top_genes_by_seurat=False,
    dt_integrateDataParams={},
    saveSeurat=None,
    returnData: Optional[Literal["ad", "so", "both"]] = "ad",
    rEnv=None,
) -> sc.AnnData:
    """`integrateBySeurat` takes an AnnData object, a batch key, and a few other parameters, and returns a
    Seurat object with the integrated data

    Parameters
    ----------
    ad : anndata.AnnData
        anndata.AnnData
    batch_key
        the column name of the batch key in ad.obs
    n_top_genes, optional
        number of top genes to use for integration
    layer, optional
        the layer of the adata object to use for integration. Default is "raw"
    reduction : Literal["cca", "rpca", "rlsi"], optional
        The dimensionality reduction method to use.
    normalization_method : Literal["LogNormalize", "SCT"], optional
        Literal["LogNormalize", "SCT"] = "LogNormalize"
    k_score, optional
        number of nearest neighbors to use for the score
    dims, optional
        number of dimensions to reduce to
    k_filter, optional
        number of genes to use for filtering
    k_weight, optional
        number of nearest neighbors to use for weighting
    identify_top_genes_by_seurat, optional
        If True, Seurat will identify the top genes for you. If False, hvg is identified by scanpy.
    saveSeurat
        if you want to save the Seurat object, give it a path.

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
    from ..otherTools import F

    rBase = importr("base")
    rUtils = importr("utils")
    R = ro.r
    setSeed()

    importr("Seurat")
    if identify_top_genes_by_seurat:
        lsR_features = n_top_genes
    else:
        if isinstance(n_top_genes, int):
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
        else:
            ls_features = n_top_genes | F(lambda z: [x.replace("_", "-") for x in z])
        lsR_features = R.c(*ls_features)

    so = ad2so(ad, layer=layer, ls_obs=[batch_key])

    rEnv["so"] = so
    rEnv["batch_key"] = batch_key
    rEnv["n_top_genes"] = n_top_genes
    rEnv["lsR_features"] = lsR_features
    rEnv["reduction"] = reduction
    rEnv["normalization.method"] = normalization_method
    rEnv["k.score"] = k_score
    rEnv["dims"] = dims
    rEnv["k.filter"] = k_filter
    rEnv["k.weight"] = k_weight
    rEnv["dtR_integrateDataParams"] = R.list(**dt_integrateDataParams)
    R(
        """
    so.list <- SplitObject(so, split.by = batch_key)
    """
    )
    if normalization_method == "LogNormalize":
        if identify_top_genes_by_seurat:
            R(
                """
            so.list <- lapply(X = so.list, FUN = function(x) {
                x <- NormalizeData(x)
                x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = n_top_genes)
            })
            lsR_features <- SelectIntegrationFeatures(object.list = so.list, nfeatures = n_top_genes)
            """
            )
        else:
            R(
                """
            so.list <- lapply(X = so.list, FUN = function(x) {
                x <- NormalizeData(x)
            })
            """
            )
        if reduction == "rpca":
            R(
                """
            so.list <- lapply(X = so.list, FUN = function(x) {
                x <- ScaleData(x, features = lsR_features, verbose = FALSE)
                x <- RunPCA(x, features = lsR_features, verbose = FALSE)
            })
            """
            )
    elif normalization_method == "SCT":
        if identify_top_genes_by_seurat:
            R(
                """
            so.list <- lapply(X = so.list, FUN = SCTransform, variable.features.n = lsR_features, vst.flavor = 'v2')
            lsR_features <- SelectIntegrationFeatures(object.list = so.list, nfeatures = n_top_genes)
            so.list <- PrepSCTIntegration(object.list = so.list, anchor.features = lsR_features)
            """
            )
        else:
            R(
                """
            so.list <- lapply(X = so.list, FUN = SCTransform, residual.features = lsR_features, vst.flavor = 'v2')
            so.list <- PrepSCTIntegration(object.list = so.list, anchor.features = lsR_features)
            """
            )
        if reduction == "rpca":
            R(
                """
            so.list <- lapply(X = so.list, FUN = function(x) {
                x <- RunPCA(x, features = lsR_features, verbose = FALSE)
            })
            """
            )
    else:
        assert False, f"unknown normalization method : {normalization_method}"
    R(
        """
    so.anchors <- FindIntegrationAnchors(object.list = so.list, anchor.features = lsR_features, reduction = reduction, normalization.method = normalization.method, dims = 1:dims, k.score = k.score, k.filter = k.filter)
    dtR_integrateDataParams$anchorset <- so.anchors
    dtR_integrateDataParams$`normalization.method` <- normalization.method
    dtR_integrateDataParams$dims <- 1:dims
    dtR_integrateDataParams$`k.weight` <- k.weight
    so.combined <- DescTools::DoCall(IntegrateData, dtR_integrateDataParams)
    DefaultAssay(so.combined) <- 'integrated'
    so.combined <- ScaleData(so.combined, verbose = FALSE)
    """
    )

    if not saveSeurat is None:
        rEnv["saveSeurat"] = saveSeurat
        R("saveRDS(so.combined, file = saveSeurat)")  # save seurat object
    so_combined = R("so.combined")
    if returnData == "so":
        return so_combined
    else:
        ad_combined = so2ad(so_combined)
        ad_combined = ad_combined[ad.obs.index]
        if normalization_method == "LogNormalize":
            ad.obsm["seurat_integrated_data"] = ad_combined.to_df(
                "integrated_data"
            ).copy()
            ad_combined.X = ad_combined.layers["integrated_data"].copy()
            sc.pp.scale(ad_combined)
        else:
            ad.obsm["seurat_integrated_data"] = ad_combined.to_df(
                "integrated_data"
            ).copy()
            ad.obsm["seurat_integrated_scale.data"] = ad_combined.obsm[
                "integrated_scale.data"
            ].copy()
            ad_combined.X = ad_combined.obsm["integrated_scale.data"].copy()

        sc.tl.pca(ad_combined, use_highly_variable=False)
        ad.obsm["X_pca_seurat"] = ad_combined.obsm["X_pca"].copy()
        ad.uns["pca_seurat"] = ad_combined.uns["pca"].copy()
        if returnData is None:
            return
        elif returnData == "ad":
            return ad_combined
        else:
            return ad_combined, so


class NormAnndata(object):
    def __init__(
        self, ad: sc.AnnData, rawLayer: str = "raw", lastResKey: str = "lastNorm"
    ):
        """
        Initialize a Normalizer object with the given AnnData object and raw layer name.

        Parameters:
        ad (sc.AnnData): AnnData object containing the raw data.
        rawLayer (str): Name of the layer containing the raw data.
        """
        self.ad = ad
        self.rawLayer = rawLayer
        self.lastResKey = lastResKey

    @property
    def lastRes(self):
        lastResKey = self.lastResKey
        if lastResKey in self.ad.uns.keys():
            return self.ad.uns[lastResKey]
        else:
            self.ad.uns[lastResKey] = {}
            return self.ad.uns[lastResKey]

    def __repr__(self):
        contents = (
            f"NormAnndata object, rawLayer: {self.rawLayer}, lastRes: {self.lastRes}\n"
            + self.ad.__repr__()
        )
        return contents

    def getSizeFactorByScran(
        self,
        resolution: float = 2,
        preCluster: Optional[str] = None,
        threads: int = 1,
        dt_kwargs2scran: dict = {"min.mean": 0.1},
    ):
        """The `getSizeFactorByScran` function calculates size factors for each cell in an AnnData object using the scran package in R, and stores the results in the AnnData object.

        Parameters
        ----------
        resolution : float, optional
            The `resolution` parameter is used to specify the resolution for clustering cells. It is a float value that determines the granularity of the clustering. Higher values result in fewer clusters, while lower values result in more clusters. The default value is 2.
        preCluster : Optional[str]
            The `preCluster` parameter is used to specify the cluster information for the cells in the `anndata` object. If `preCluster` is not specified, the function will use the `connectivity` stored in the `anndata` object and the specified `resolution` to cluster the cells
        threads : int, optional
            The `threads` parameter specifies the number of threads to be used for parallel processing. It determines how many parallel processes can be executed simultaneously.
        dt_kwargs2scran : dict
            The `dt_kwargs2scran` parameter is a dictionary that contains additional arguments to be passed to the `calculateSumFactors` function from the `scran` package in R. These arguments control the behavior of the size factor calculation. In the code snippet, the default value for `dt_kwargs

        """
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from scipy.sparse import csr_matrix, isspmatrix
        import scipy.sparse as ss
        from ..rTools import py2r, r2py

        R = ro.r
        importr("scran")
        bp = importr("BiocParallel")

        ad = self.ad

        if preCluster is None:
            logger.info(
                f"cluster information is not specified, `connectivity` stored in anndata and reslution {resolution} will be used to cluster cells"
            )
            logger.warning(
                "I strongly recommend you to specify `preCluster` to avoid repeated clustering"
            )
            ad_bc = ad
            ad = sc.AnnData(ad_bc.layers["normalize_log"].copy(), obs=ad.obs.copy())
            sc.pp.highly_variable_genes(ad, n_top_genes=3000)
            sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
            sc.pp.neighbors(ad)
            sc.tl.leiden(ad, resolution=resolution, key_added="scran_used_cluster")
            ad_bc.obs["scran_used_cluster"] = ad.obs["scran_used_cluster"].copy()
            ad = ad_bc
        else:
            ad.obs["scran_used_cluster"] = self.ad.obs[preCluster]

        if threads == 1:
            BPPARAM = bp.SerialParam()
        else:
            BPPARAM = bp.MulticoreParam(threads)

        dfR_clusterInfo = py2r(ad.obs["scran_used_cluster"])
        mtx = ad.layers[self.rawLayer]
        if ss.issparse(mtx):
            if not isinstance(mtx, ss.csr_matrix):
                mtx = ss.csr_matrix(mtx)

        mtxR = py2r(mtx.T)
        logger.info("calculate size factor")
        vtR_sizeFactor = R.calculateSumFactors(
            mtxR, clusters=dfR_clusterInfo, BPPARAM=BPPARAM, **dt_kwargs2scran
        )
        sr_sizeFactor = r2py(vtR_sizeFactor).copy()
        logger.info("process result")
        ad.obs["scran_sizeFactor"] = sr_sizeFactor
        ad.layers["scran_norm_count"] = csr_matrix(
            ad.layers["raw"].multiply(
                1 / ad.obs["scran_sizeFactor"].values.reshape(-1, 1)
            )
        )

    def normByApr(self, nTopGenes=3000, batchKey=None, onlyRunPca=True, comps=50):
        import gc
        import scipy.sparse as ss

        ad = self.ad
        ad.X = ad.layers[self.rawLayer].copy()
        if batchKey is None:
            sc.experimental.pp.highly_variable_genes(
                ad, n_top_genes=nTopGenes, batch_key=batchKey
            )
        else:
            ls_geneExpInAllSample = basic.filterGeneBasedOnSample(
                ad, batchKey, layer=self.rawLayer
            )
            _ad = ad[:, ls_geneExpInAllSample]
            df_hvg = sc.experimental.pp.highly_variable_genes(
                _ad, n_top_genes=nTopGenes, batch_key=batchKey, inplace=False
            )
            ad.var["highly_variable"] = df_hvg.reindex(ad.var.index)[
                "highly_variable"
            ].fillna(False)

        ad_apr = ad[:, ad.var["highly_variable"]].copy()
        del ad_apr.uns

        if batchKey is None:
            sc.experimental.pp.normalize_pearson_residuals(ad_apr)
        else:
            lsAd_apr = []
            for _ad in basic.splitAdata(ad_apr, batchKey, copy=True):
                sc.experimental.pp.normalize_pearson_residuals(_ad)
                lsAd_apr.append(_ad)
            ad_apr = sc.concat(lsAd_apr)
            ad_apr = ad_apr[ad.obs.index]

        if onlyRunPca:
            ad_apr.var["highly_variable"] = True
            sc.tl.pca(ad_apr, n_comps=comps, use_highly_variable=True)
            ad.obsm["X_pca"] = ad_apr.obsm["X_pca"].copy()
            ad.uns["pca"] = ad_apr.uns["pca"].copy()
        else:
            df_apr = pd.DataFrame.sparse.from_spmatrix(
                ss.csc_matrix(ad_apr.X),
                index=ad_apr.obs.index,
                columns=ad_apr.var.index,
            )
            ls_otherGenes = [x for x in ad.var.index if x not in ad_apr.var.index]
            df_others = pd.DataFrame.sparse.from_spmatrix(
                ss.csc_matrix((ad.shape[0], len(ls_otherGenes))),
                index=ad.obs.index,
                columns=ls_otherGenes,
            )
            df_final = pd.concat([df_apr, df_others], axis=1)
            df_final = df_final.reindex(columns=ad.var.index)
            ad.layers["APR"] = df_final
            ad.X = ad.layers["APR"].copy()
            sc.tl.pca(ad, n_comps=comps, use_highly_variable=True)

        self.lastRes["normMethod"] = "APR"
        gc.collect()

    def normBySct(
        self,
        batchKey=None,
        layer=None,
        nTopGenes=3000,
        ls_gene=None,
        vstFlavor="v2",
        rEnv=None,
        comps=50,
        njobs=4,
        sctOnly=False,
        **dt_kwargsToSct,
    ):
        from joblib import Parallel, delayed

        layer = self.rawLayer if layer is None else layer
        if batchKey is None:
            ad_sct = normalizeBySCT_r(
                self.ad,
                layer=layer,
                nTopGenes=nTopGenes,
                ls_gene=ls_gene,
                vstFlavor=vstFlavor,
                returnOnlyVarGenes=True,
                doCorrectUmi=False,
                rEnv=rEnv,
                runSctOnly=True,
                **dt_kwargsToSct,
            )
            self.ad.var["highly_variable"] = ad_sct.var["highly_variable"]
            self.ad.var["highly_variable_rank"] = ad_sct.var["highly_variable_rank"]
            self.ad.uns["sct_vst_pickle"] = ad_sct.uns["sct_vst_pickle"]
            self.ad.uns["sct_clip_range"] = ad_sct.uns["sct_clip_range"]
            ls_hvg = ad_sct.var.loc[ad_sct.var["highly_variable"]].index.to_list()
            self.getSctRes(ls_gene=ls_hvg, forceOverwrite=True)

        else:
            self.ad.uns["sctModels"] = {}

            # from concurrent.futures import ProcessPoolExecutor
            # executor = ProcessPoolExecutor(njobs)
            # ls_task = []
            # lsAd_sct = []
            # ls_sample = []
            # for sample, _ad in basic.splitAdata(self.ad, batchKey, copy=True, needName=True):
            #     task = executor.submit(normalizeBySCT_r, _ad, layer=layer, nTopGenes=nTopGenes, ls_gene=ls_gene, vstFlavor=vstFlavor, returnOnlyVarGenes=True, doCorrectUmi=False, rEnv=rEnv, runSctOnly=True, **dt_kwargsToSct)
            #     ls_task.append(task)
            #     lsAd_sct.append(_ad)
            #     ls_sample.append(sample)

            # for task, _ad, sample in zip(ls_task, lsAd_sct, ls_sample):
            #     # ad_sct = normalizeBySCT_r(_ad, layer=layer, nTopGenes=nTopGenes, ls_gene=ls_gene, vstFlavor=vstFlavor, returnOnlyVarGenes=True, doCorrectUmi=False, rEnv=rEnv, runSctOnly=True, **dt_kwargsToSct)
            #     ad_sct = task.result()
            #     _ad.var['highly_variable'] = ad_sct.var['highly_variable']
            #     _ad.var['highly_variable_rank'] = ad_sct.var['highly_variable_rank']
            #     _ad.uns["sct_vst_pickle"] = ad_sct.uns["sct_vst_pickle"]
            #     _ad.uns["sct_clip_range"] = ad_sct.uns["sct_clip_range"]
            #     self.ad.uns['sctModels'][f'{sample}_sct_vst_pickle'] = ad_sct.uns["sct_vst_pickle"]
            #     self.ad.uns['sctModels'][f'{sample}_sct_clip_range'] = ad_sct.uns["sct_clip_range"]
            #     # self.ad.uns[f'{sample}_sct_vst_pickle'] = ad_sct.uns["sct_vst_pickle"]
            #     # self.ad.uns[f'{sample}_sct_clip_range'] = ad_sct.uns["sct_clip_range"]
            # executor.shutdown()

            lsAd_sct = []
            ls_sample = []
            for sample, _ad in basic.splitAdata(
                self.ad, batchKey, copy=True, needName=True
            ):
                lsAd_sct.append(_ad)
                ls_sample.append(sample)
            lsAd_sct = Parallel(n_jobs=njobs)(
                delayed(normalizeBySCT_r)(
                    _ad,
                    layer=layer,
                    nTopGenes=nTopGenes,
                    ls_gene=ls_gene,
                    vstFlavor=vstFlavor,
                    returnOnlyVarGenes=True,
                    doCorrectUmi=False,
                    rEnv=rEnv,
                    runSctOnly=True,
                    **dt_kwargsToSct,
                )
                for _ad in lsAd_sct
            )

            df_hvgRank = pd.DataFrame(index=self.ad.var.index)
            for ad_sct, sample in zip(lsAd_sct, ls_sample):
                self.ad.uns["sctModels"][f"{sample}_sct_vst_pickle"] = ad_sct.uns[
                    "sct_vst_pickle"
                ]
                self.ad.uns["sctModels"][f"{sample}_sct_clip_range"] = ad_sct.uns[
                    "sct_clip_range"
                ]
                df_hvgRank[sample] = ad_sct.var['highly_variable_rank'].copy()

            ls_hvg, _ = getHvgGeneFromSctAdata(
                lsAd_sct, nTopGenes=nTopGenes, nTopGenesEachAd=nTopGenes
            )
            self.ad.var["highly_variable"] = self.ad.var.index.isin(ls_hvg)
            self.ad.varm["highly_variable_rank_sct"] = df_hvgRank

            if sctOnly:
                pass
            else:
                self.getSctRes(ls_gene=ls_hvg, forceOverwrite=True, batchKey=batchKey)

        if sctOnly:
            pass
        else:
            ad_resi = sc.AnnData(self.ad.obsm["sct_residual"])
            ad_resi.var["highly_variable"] = True

            sc.tl.pca(ad_resi, n_comps=comps, use_highly_variable=True)
            self.ad.obsm["X_pca"] = ad_resi.obsm["X_pca"].copy()
            self.ad.uns["pca"] = ad_resi.uns["pca"].copy()

    def getSctRes(self, ls_gene, layer=None, forceOverwrite=False, batchKey=None):
        layer = self.rawLayer if layer is None else layer
        if batchKey is None:
            getSctResiduals(
                self.ad, ls_gene, layer=layer, forceOverwrite=forceOverwrite
            )
        else:
            dt_sctModels = self.ad.uns.get("sctModels", {})
            if dt_sctModels == {}:
                logger.warning("sctModels is not found in adata.uns['sctModels']")
                dt_sctModels = self.ad.uns

            lsAd = []
            for sample, _ad in basic.splitAdata(
                self.ad, batchKey, copy=True, needName=True
            ):
                getSctResiduals(
                    _ad,
                    ls_gene,
                    layer=layer,
                    forceOverwrite=forceOverwrite,
                    sctVstPickle=dt_sctModels[f"{sample}_sct_vst_pickle"],
                    sctClipRange=dt_sctModels[f"{sample}_sct_clip_range"],
                )
                lsAd.append(_ad)
            
            _ls = []
            for _ad in lsAd:
                if isinstance(_ad.obsm["sct_residual"], pd.DataFrame):
                    _ls.append(_ad.obsm["sct_residual"])
                else:
                    _ls.append(
                        pd.DataFrame(
                            _ad.obsm["sct_residual"],
                            columns=_ad.uns['sct_residual_keys'],
                            index=_ad.obs.index
                        )
                    )
            df_resi = pd.concat(
               _ls, axis=0, join='inner'
            ).reindex(index=self.ad.obs.index)
            self.ad.obsm["sct_residual"] = df_resi.copy()
            self.ad.uns['sct_residual_genes'] = df_resi.columns.to_list()

    def getHvgGeneFromSctAdata(self, batchKey, nTopGenes=3000, nTopGenesEachAd=3000):
        """> get the top  HVGs that are shared across all adatas

        Parameters
        ----------
        ls_ad
            a list of adata objects, must be `data` slot. Gene listed in all adata objects will be used.
        nTopGenes, optional
            the total number of genes to use for the analysis
        nTopGenesEachAd, optional
            the number of HVGs to use from each adata object

        Returns
        -------
            A list of genes that are highly variable across all adatas.

        """
        import pickle
        ead = self.ad
        df_hvgRank = ead.varm["highly_variable_rank_sct"]
        dt_sctModels = ead.uns['sctModels']
        ls_sample = ead.obs[batchKey].unique().tolist()
        
        ls_allGenes = []
        for sample in ls_sample:
            vst_out = pickle.loads(eval(dt_sctModels[f"{sample}_sct_vst_pickle"]))
            ls_allGenes.extend(list(vst_out[2].rownames))
        ls_allGenes = (
            pd.Series(ls_allGenes)
            .value_counts()
            .loc[lambda x: x == len(ls_sample)]
            .index.to_list()
        )

        ls_allHvg = []
        for sample in ls_sample:
            ls_allHvg.extend(
                df_hvgRank.sort_values(sample)
                .dropna()
                .index[:nTopGenesEachAd]
                .to_list()
            )
        ls_allHvg = [x for x in ls_allHvg if x in ls_allGenes]
        assert (
            len(set(ls_allHvg)) > nTopGenes
        ), "nTopGenes must be smaller than total number of HVGs"
        ls_hvgCounts = pd.Series(ls_allHvg).value_counts()

        ls_usedHvg = []
        for hvgCounts in range(len(ls_sample), 0, -1):
            ls_currentCountsHvg = ls_hvgCounts[ls_hvgCounts == hvgCounts].index.to_list()
            if (len(ls_usedHvg) + len(ls_currentCountsHvg)) > nTopGenes:
                break
            ls_usedHvg.extend(ls_currentCountsHvg)

        needAnotherCounts = nTopGenes - len(ls_usedHvg)
        df_remainGeneRank = pd.DataFrame(index=list(set(ls_allGenes) - set(ls_usedHvg)))
        for i, sample in enumerate(ls_sample):
            df_remainGeneRank[f"{i}"] = df_hvgRank[sample]
        df_remainGeneRank = df_remainGeneRank.sort_index()
        df_remainGeneRank["count"] = pd.notna(df_remainGeneRank).sum(1)
        df_remainGeneRank["median"] = df_remainGeneRank.drop(columns="count").apply(
            "median", axis=1
        )
        df_remainGeneRank = df_remainGeneRank.sort_values(
            ["count", "median"], ascending=[False, True]
        )
        ls_usedHvg.extend(df_remainGeneRank.iloc[:needAnotherCounts].index.to_list())
        return ls_usedHvg, df_remainGeneRank

    def getScviEmbedding(
        self,
        nTopGenes=3000,
        layer=None,
        batchKey=None,
        hvgSpan=0.3,
        nLayers=2,
        nLatent=30,
        geneLikelihood="nb",
        earlyStop=True,
        categoricalCovariateKeys=None,
        dt_kwargsToSetup={},
        dt_kwargsToModel={},
        dt_kwargsToTrain={},
    ):
        import scvi

        scvi.settings.seed = 39

        layer = self.rawLayer if layer is None else layer
        sc.pp.highly_variable_genes(
            self.ad,
            n_top_genes=nTopGenes,
            layer=layer,
            batch_key=batchKey,
            flavor="seurat_v3",
            span=hvgSpan
        )
        ad_scvi = self.ad[:, self.ad.var["highly_variable"]].copy()
        scvi.model.SCVI.setup_anndata(
            ad_scvi,
            layer=layer,
            batch_key=batchKey,
            categorical_covariate_keys=categoricalCovariateKeys,
            **dt_kwargsToSetup,
        )
        # model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        model = scvi.model.SCVI(
            ad_scvi,
            n_layers=nLayers,
            n_latent=nLatent,
            gene_likelihood=geneLikelihood,
            **dt_kwargsToModel,
        )
        model.train(early_stopping=earlyStop, **dt_kwargsToTrain)
        ad_scvi.obsm["X_scvi"] = model.get_latent_representation()
        self.ad.obsm["X_scvi"] = ad_scvi.obsm["X_scvi"].copy()

    def integrateBySeurat(
        self,
        batch_key,
        n_top_genes=5000,
        layer="raw",
        reduction: Literal["cca", "rpca", "rlsi"] = "cca",
        normalization_method: Literal["LogNormalize", "SCT"] = "LogNormalize",
        k_score=30,
        dims=30,
        k_filter=200,
        k_weight=100,
        identify_top_genes_by_seurat=False,
        dt_integrateDataParams={},
        saveSeurat=None,
        returnData: Optional[Literal["ad", "so", "both"]] = None,
        rEnv=None,
    ):
        ad = self.ad
        if normalization_method == "SCT":
            logger.warning(
                "SCT normalization will be re-run and the previous SCT results will be overwritten"
            )
        return integrateBySeurat(
            ad,
            batch_key,
            n_top_genes=n_top_genes,
            layer=layer,
            reduction=reduction,
            normalization_method=normalization_method,
            k_score=k_score,
            dims=dims,
            k_filter=k_filter,
            k_weight=k_weight,
            identify_top_genes_by_seurat=identify_top_genes_by_seurat,
            dt_integrateDataParams=dt_integrateDataParams,
            saveSeurat=saveSeurat,
            returnData=returnData,
            rEnv=rEnv,
        )
