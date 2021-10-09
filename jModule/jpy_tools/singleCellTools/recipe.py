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


def sct(ad):
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
    normalize.normalizeBySCT(ad, layer="raw", min_cells=10, log_scale_correct=True)
    ad.X = ad.layers["sct_residuals"].copy()
    sc.tl.pca(ad)
    sc.pp.neighbors(ad, n_pcs=50)
    sc.tl.umap(ad)
    ad.X = ad.layers["normalize_log"].copy()


def multiBatch(
    ad,
    batch: Optional[str],
    method: Optional[Literal["harmony", "scanorama", "scvi"]] = None,
    nomalization: Optional[Literal["total", "SCT"]] = "total",
    n_top_genes = 5000
):
    """
    sct + pca + harmony|scanorama|scvi + neighbors + umap

    `.X` must be raw data

    ----------
    `ad` will be updated.
    ad.X will be `normalize_log` layer
    """
    import scanpy.external as sce

    basic.testAllCountIsInt(ad, None)
    if not batch:
        batch = "_batch"
        ad.obs[batch] = "same"
    ad.layers["raw"] = ad.X.copy()
    ad.layers["normalize_log"] = ad.layers["raw"].copy()
    sc.pp.normalize_total(ad, 1e4, layer="normalize_log")
    sc.pp.log1p(ad, layer="normalize_log")
    sc.pp.highly_variable_genes(ad, "raw", n_top_genes=n_top_genes, flavor="seurat_v3")
    ad.X = ad.layers["normalize_log"].copy()
    if nomalization == "total":
        if method == "harmony":
            sc.pp.scale(ad, max_value=10)
        elif method == "scanorama":
            basic.scIB_scale_batch(ad, batch)
    elif nomalization == "SCT":
        ls_adataAfterSCT = []
        for _ad in basic.splitAdata(ad, batch):
            normalize.normalizeBySCT(
                _ad, layer="raw", n_top_genes=n_top_genes, n_genes=n_top_genes, log_scale_correct=True
            )
            ls_adataAfterSCT.append(_ad)
        ad = sc.concat(ls_adataAfterSCT)
        ad.X = ad.layers["sct_residuals"].copy()
        sc.pp.highly_variable_genes(ad, "raw", n_top_genes=n_top_genes, flavor="seurat_v3")

    sc.tl.pca(ad)
    if method == "harmony":
        sce.pp.harmony_integrate(ad, batch, adjusted_basis="X_harmony", max_iter_harmony=50)
        ad.obsm["X_integrated"] = ad.obsm["X_harmony"].copy()
    elif method == "scanorama":
        sce.pp.scanorama_integrate(ad, batch, adjusted_basis="X_scanorama")
        ad.obsm["X_integrated"] = ad.obsm["X_scanorama"].copy()
    elif method == "scvi":
        import scvi

        ad_forScvi = basic.getPartialLayersAdata(
            ad, "raw", [batch], ["highly_variable"]
        )
        ad_forScvi = ad_forScvi[:, ad_forScvi.var["highly_variable"]].copy()
        scvi.data.setup_anndata(ad_forScvi, batch_key=batch)

        scvi.settings.seed = 39
        scvi.settings.num_threads = 56
        model = scvi.model.SCVI(ad_forScvi)
        model.train(early_stopping=True)
        ad.obsm["X_scvi"] = model.get_latent_representation(ad_forScvi).copy()
        ad.obsm["X_integrated"] = ad.obsm["X_scvi"].copy()
    sc.pp.neighbors(ad, use_rep="X_integrated")
    sc.tl.umap(ad)
    ad.obsm[f"X_umap_{method}"] = ad.obsm["X_umap"].copy()
    ad.X = ad.layers["normalize_log"].copy()
    sc.pl.embedding(ad, f"X_umap_{method}", color=batch)
    return ad
