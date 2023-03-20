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
import muon as mu
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
    returnMuon = False,
    returnSo = False,
    rEnv = None,
    debug = False,
    runSctOnly = False,
    **dt_kwargsToSct,
):
    '''`normalizeBySCT_r` is a function that takes in a single-cell RNA-seq dataset and returns a
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
    
    '''
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

    R = ro.r
    setSeed()
    def setSctVstToAd(ad, rEnv):
        model = R("levels(x = so_sct[['SCT']])")
        assert len(list(model)) == 1, "model must be 1"
        R("""
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
        """)
        vst_out = rEnv['vst_out']
        ad.uns['sct_vst_pickle'] = str(pickle.dumps(vst_out))
        ad.uns['sct_clip_range'] = list(rEnv['clip_range'])



    so = ad2so(ad, layer=layer, lightMode=True)
    if debug:
        import pdb;pdb.set_trace()
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
    
    ls_var =  R("VariableFeatures")(so_sct, assay='SCT') >> F(list)
    dt_var = {ls_var[i]:i for i in range(len(ls_var))}
    ad.var['highly_variable'] = ad.var.index.isin(dt_var)
    ad.var['highly_variable_rank'] = ad.var.index.map(lambda x:dt_var.get(x, np.nan))
    if runSctOnly:
        return ad

    
    md_sct = so2md(so_sct)
    md_sct['SCT_scale.data'].var['highly_variable'] = md_sct['SCT_scale.data'].var.index.isin(ls_hvg)
    md_sct['SCT_scale.data'].X = md_sct['SCT_scale.data'].layers['SCT_scale.data']
    md_sct.uns["sct_vst_pickle"] = ad.uns["sct_vst_pickle"]
    md_sct.uns["sct_clip_range"] = ad.uns["sct_clip_range"]

    sc.tl.pca(md_sct['SCT_scale.data'])
    ad.obsm['X_pca_sct'] = md_sct['SCT_scale.data'].obsm['X_pca'].copy()
    ad.uns['pca_sct'] = md_sct['SCT_scale.data'].uns['pca'].copy()
    ad.obsm['SCT_data'] = md_sct['SCT'].layers['SCT_data']
    ad.uns['SCT_data_features'] = md_sct['SCT'].var.index.to_list()
    


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
    '''> get the top  HVGs that are shared across all adatas

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

    '''
    for ad in ls_ad:
        assert 'highly_variable_rank' in ad.var.columns, "adata must have highly_variable_rank"
    ls_allGenes = []
    for ad in ls_ad:
        ls_allGenes.extend(ad.var.index.to_list())
    ls_allGenes = pd.Series(ls_allGenes).value_counts().loc[lambda x: x == len(ls_ad)].index.to_list()
    

    ls_allHvg = []
    for ad in ls_ad:
        _t = True
        ls_allHvg.extend(ad.var.query("highly_variable == @_t").sort_values('highly_variable_rank').index[:nTopGenesEachAd].to_list())
    ls_allHvg = [x for x in ls_allHvg if x in ls_allGenes]
    assert len(set(ls_allHvg)) > nTopGenes, "nTopGenes must be smaller than total number of HVGs"
    ls_hvgCounts = pd.Series(ls_allHvg).value_counts()

    ls_usedHvg = []
    for hvgCounts in range(len(ls_ad), 0, -1):
        ls_currentCountsHvg = ls_hvgCounts[ls_hvgCounts == hvgCounts].index.to_list()
        if (len(ls_usedHvg) + len(ls_currentCountsHvg)) > nTopGenes:
            break
        ls_usedHvg.extend(ls_currentCountsHvg)

    needAnotherCounts = nTopGenes - len(ls_usedHvg)
    df_remainGeneRank = pd.DataFrame(index=list(set(ls_allGenes) - set(ls_usedHvg)))
    for i,ad in enumerate(ls_ad):
        df_remainGeneRank[f"{i}"] = ad.var['highly_variable_rank']
    df_remainGeneRank = df_remainGeneRank.sort_index()
    df_remainGeneRank['count'] = pd.notna(df_remainGeneRank).sum(1)
    df_remainGeneRank['median'] = df_remainGeneRank.drop(columns='count').apply('median', axis=1)
    df_remainGeneRank = df_remainGeneRank.sort_values(['count', 'median'], ascending=[False, True])
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

def getSctResiduals(ad, ls_gene, layer='raw', forceOverwrite=False):
    import pickle
    from ..rTools import py2r, r2py
    from ..otherTools import F

    basic.testAllCountIsInt(ad, layer)
    assert 'sct_vst_pickle' in ad.uns, "sct_vst_pickle not found in adata.uns"
    assert 'sct_clip_range' in ad.uns, "sct_clip_range not found in adata.layers"
    import rpy2.robjects as ro
    R = ro.r

    if ('sct_residual' not in ad.obsm.keys()) or forceOverwrite:
        ls_gene = ls_gene
    else:
        ls_gene = list(set(ls_gene) - set(ad.obsm['sct_residual'].columns))
    if len(ls_gene) == 0:
        return None

    fcR_getResiduals = R("sctransform::get_residuals")
    vst_out = pickle.loads(eval(ad.uns['sct_vst_pickle']))
    ls_clipRange = list(ad.uns['sct_clip_range'])
    df_residuals = fcR_getResiduals(vst_out, umi=ad[:, ls_gene].to_df(layer).T >> F(py2r) >> F(R("data.matrix")) >> F(R("Matrix::Matrix")), res_clip_range=R.c(*ls_clipRange)) >> F(R("as.data.frame")) >> F(r2py)
    df_residuals = df_residuals.T
    df_residuals = df_residuals - df_residuals.mean()
    if ('sct_residual' not in ad.obsm.keys()) or forceOverwrite:
        ad.obsm['sct_residual'] = df_residuals
    else:
        ad.obsm['sct_residual'] = pd.concat([ad.obsm['sct_residual'], df_residuals], axis=1)


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
    rEnv = None,
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
    """
    )
    if not saveSeurat is None:
        rEnv["saveSeurat"] = saveSeurat
        R("saveRDS(so.combined, file = saveSeurat)")  # save seurat object
    so_combined = R("so.combined")
    ad_combined = so2ad(so_combined)
    ad_combined = ad_combined[ad.obs.index]
    if normalization_method == "LogNormalize":
        ad.obsm["seurat_integrated_data"] = ad_combined.to_df("integrated_data").copy()
        ad_combined.X = ad_combined.layers["integrated_data"].copy()
        sc.pp.scale(ad_combined)
    else:
        ad.obsm["seurat_integrated_data"] = ad_combined.to_df("integrated_data").copy()
        ad.obsm["seurat_integrated_scale.data"] = ad_combined.obsm["integrated_scale.data"].copy()
        ad_combined.X = ad_combined.obsm["integrated_scale.data"].copy()

    sc.tl.pca(ad_combined, use_highly_variable=False)
    ad.obsm["X_pca_seurat"] = ad_combined.obsm["X_pca"].copy()
    return ad_combined
