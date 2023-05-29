"""
recipe
"""
from logging import addLevelName, log
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
from . import basic
from . import normalize
from . import normalize


def sct(ad, n_top_genes=3000):
    """
    sct + pca + neighbors + umap

    `.X` must be raw data

    ----------
    `ad` will be updated.
    ad.X will be `normalize_log` layer
    """
    basic.testAllCountIsInt(ad, None)
    ad.layers["raw"] = ad.X.copy()
    ad.layers["normalize_log"] = ad.layers["raw"].copy()
    sc.pp.normalize_total(ad, 1e4, layer="normalize_log")
    sc.pp.log1p(ad, layer="normalize_log")
    normalize.normalizeBySCT(
        ad,
        layer="raw",
        min_cells=1,
        log_scale_correct=True,
        n_top_genes=n_top_genes,
        n_genes=n_top_genes,
    )
    ad.X = ad.layers["sct_residuals"].copy()
    sc.tl.pca(ad)
    sc.pp.neighbors(ad, n_pcs=50)
    sc.tl.umap(ad, min_dist=0.2)
    ad.X = ad.layers["normalize_log"].copy()


def singleBatch(
    ad, method: Literal["total", "sct", "scran", "sct_r"], n_top_genes=3000
) -> sc.AnnData:
    """
    sct|scran|total + pca + neighbors + umap

    `.X` must be raw data

    ----------
    `ad` will be updated.
    ad.X will be `normalize_log` layer
    """
    method = method.lower()
    basic.testAllCountIsInt(ad, None)
    ad.layers["raw"] = ad.X.copy()
    ad.layers["normalize_log"] = ad.layers["raw"].copy()
    sc.pp.normalize_total(ad, 1e4, layer="normalize_log")
    sc.pp.log1p(ad, layer="normalize_log")
    if method == "sct":
        normalize.normalizeBySCT(
            ad,
            layer="raw",
            min_cells=1,
            log_scale_correct=True,
            n_top_genes=n_top_genes,
            n_genes=n_top_genes,
        )
        ad.X = ad.layers["sct_residuals"].copy()
    elif method == 'sct_r':
        normalize.normalizeBySCT_r(ad, layer='raw', nTopGenes=n_top_genes)
        ad.obsm['X_pca'] = ad.obsm['X_pca_sct']
        ad.uns['pca'] = ad.uns['pca_sct']
    elif method == "scran":
        sc.pp.highly_variable_genes(
            ad, layer="raw", flavor="seurat_v3", n_top_genes=n_top_genes
        )
        normalize.normalizeByScran(ad, layer="raw")
        ad.X = ad.layers["scran"].copy()
        sc.pp.scale(ad)
    elif method == "total":
        sc.pp.highly_variable_genes(
            ad, layer="raw", flavor="seurat_v3", n_top_genes=n_top_genes
        )
        ad.X = ad.layers["normalize_log"].copy()
        sc.pp.scale(ad)
    else:
        assert False, "Unsupported"
    
    if method != 'sct_r':
        sc.tl.pca(ad)

    sc.pp.neighbors(ad, n_pcs=50)
    sc.tl.umap(ad, min_dist=0.2)
    ad.X = ad.layers["normalize_log"].copy()
    return ad


type_method = Literal["harmony", "scanorama", "scvi", "seurat"]


def multiBatch(
    ad,
    batch: str,
    method: Union[List[type_method], type_method],
    normalization: Optional[Literal["total", "sct", "scran"]] = "total",
    n_top_genes=5000,
    ls_remove_cateKey=[],
    scale_individual=False,
    max_epochs=None,
    reduction: Literal["cca", "rpca", "rlsi"] = "cca",
    batch_size = 512,
    dt_params2Model = {}
) -> sc.AnnData:
    """
    sct + pca + harmony|scanorama|scvi + neighbors + umap

    `.X` must be raw data

    ----------
    `ad` will be updated.
    ad.X will be `normalize_log` layer
    """
    import scanpy.external as sce

    if isinstance(method, str):
        ls_method = [method]
    else:
        ls_method = method
    normalization = normalization.lower()

    basic.testAllCountIsInt(ad, None)
    # if not batch:
    #     batch = "_batch"
    #     ad.obs[batch] = "same"
    ad.layers["raw"] = ad.X.copy()
    ad.layers["normalize_log"] = ad.layers["raw"].copy()
    sc.pp.normalize_total(ad, 1e4, layer="normalize_log")
    sc.pp.log1p(ad, layer="normalize_log")
    sc.pp.highly_variable_genes(ad, "raw", n_top_genes=n_top_genes, flavor="seurat_v3")
    ad.X = ad.layers["normalize_log"].copy()
    if normalization == "total":
        if scale_individual:
            basic.scIB_scale_batch(ad, batch)
        else:
            sc.pp.scale(ad, max_value=10)
    elif normalization == "sct":
        if method != ['seurat']:
            ls_adataAfterSCT = []
            for _ad in basic.splitAdata(ad, batch):
                normalize.normalizeBySCT(
                    _ad,
                    layer="raw",
                    n_top_genes=n_top_genes,
                    n_genes=int(min((max(2 * n_top_genes, 0.5 * _ad.shape[1])), ad.shape[1])),
                    log_scale_correct=True,
                )
                ls_adataAfterSCT.append(_ad)
            ad = sc.concat(ls_adataAfterSCT)
            ad.X = ad.layers["sct_residuals"].copy()
            sc.pp.highly_variable_genes(
                ad, "raw", n_top_genes=n_top_genes, flavor="seurat_v3"
            )
    elif normalization == "scran":
        ls_adataAfterScran = []
        for _ad in basic.splitAdata(ad, batch):
            normalize.normalizeByScran(_ad, layer="raw")
            ls_adataAfterScran.append(_ad)
        ad = sc.concat(ls_adataAfterScran)
        ad.X = ad.layers["scran"].copy()
        sc.pp.scale(ad)
    else:
        assert False, "Unsupported"

    sc.tl.pca(ad)
    for method in ls_method:
        if method == "harmony":
            sce.pp.harmony_integrate(
                ad, batch, adjusted_basis="X_harmony", max_iter_harmony=50
            )
            ad.obsm["X_integrated"] = ad.obsm["X_harmony"].copy()
        elif method == "scanorama":
            sce.pp.scanorama_integrate(ad, batch, adjusted_basis="X_scanorama")
            ad.obsm["X_integrated"] = ad.obsm["X_scanorama"].copy()
        elif method == "scvi":
            import scvi

            ad_forScvi = basic.getPartialLayersAdata(
                ad, "raw", [batch, *ls_remove_cateKey], ["highly_variable"]
            )
            ad_forScvi = ad_forScvi[:, ad_forScvi.var["highly_variable"]].copy()
            scvi.model.SCVI.setup_anndata(
                ad_forScvi,
                batch_key=batch,
                categorical_covariate_keys=ls_remove_cateKey,
            )

            scvi.settings.seed = 39
            scvi.settings.num_threads = 36
            model = scvi.model.SCVI(ad_forScvi, **dt_params2Model)
            model.train(max_epochs=max_epochs, early_stopping=True, batch_size = batch_size)
            model.history["elbo_train"].plot()
            ad.obsm["X_scvi"] = model.get_latent_representation(ad_forScvi).copy()
            ad.obsm["X_integrated"] = ad.obsm["X_scvi"].copy()
        elif method == "seurat":
            from .normalize import integrateBySeurat

            seuratNormalizeMethod = {"total": "LogNormalize", "sct": "SCT"}[
                normalization
            ]
            integrateBySeurat(
                ad,
                batch,
                layer="raw",
                n_top_genes=n_top_genes,
                reduction=reduction,
                normalization_method=seuratNormalizeMethod,
            )
            ad.obsm["X_integrated"] = ad.obsm["X_pca_seurat"].copy()
        else:
            assert False, "Unsupported"
        sc.pp.neighbors(ad, use_rep="X_integrated")
        sc.tl.umap(ad, min_dist=0.2)
        ad.obsm[f"X_umap_{method}_{normalization}"] = ad.obsm["X_umap"].copy()
        ad.X = ad.layers["normalize_log"].copy()
        sc.pl.embedding(ad, f"X_umap_{method}_{normalization}", color=batch)
    return ad

def aprRecipe(ad, obsKey, layer, n_top_genes=5000):
    """
    This function performs APR (analytic pearson residual) on the input AnnData object
    ad: AnnData object containing gene expression data
    obsKey: name of the observation key to group cells by


    ## details
    The function first creates a new AnnData object containing only genes detected in all samples
    It then splits the data by the observation key and calculates the Pearson residuals for each sample
    The Pearson residuals are converted to a DataFrame and added to a list
    The PCA is performed on the combined DataFrame of all samples
    The PCA results are added to the original AnnData object as uns and obsm attributes
    """
    ls_detectedGeneInAllSamples = basic.filterGeneBasedOnSample(ad, obsKey, layer=layer)
    ad_apr = ad[:, ls_detectedGeneInAllSamples].copy()
    sc.experimental.pp.highly_variable_genes(ad_apr, n_top_genes=n_top_genes, batch_key=obsKey, layer=layer, chunksize=5000, subset=True)

    lsDf_apr = []
    for sample, _ad in basic.splitAdata(ad_apr, obsKey, needName=True):
        x = sc.experimental.pp.normalize_pearson_residuals(_ad, inplace=False, layer=layer)
        df_apr = pd.DataFrame(x['X'], index=_ad.obs.index, columns=_ad.var.index)
        lsDf_apr.append(df_apr)
    df_apr = pd.concat(lsDf_apr)
    tp_aprPca = sc.tl.pca(df_apr.loc[ad.obs.index].values, return_info=True)

    ad.uns['pca'] = {}
    ad.obsm['X_pca'] = tp_aprPca[0]
    ad.uns['pca']['variance_ratio'] = tp_aprPca[2]
    ad.uns['pca']['variance'] = tp_aprPca[3]