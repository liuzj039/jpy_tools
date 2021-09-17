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

def getDEG(
    adata,
    df_deInfo,
    groupby,
    groups=None,
    bayesCutoff=3,
    nonZeroProportion=0.1,
    markerCounts=None,
    keyAdded=None,
):
    adata.obs[groupby] = adata.obs[groupby].astype("category")
    if not groups:
        cats = list(adata.obs[groupby].cat.categories)

    markers = {}
    cats = adata.obs[groupby].cat.categories
    for i, c in enumerate(cats):
        cid = "{} vs Rest".format(c)
        cell_type_df = df_deInfo.loc[df_deInfo.comparison == cid]

        cell_type_df = cell_type_df[cell_type_df.lfc_mean > 0]

        cell_type_df = cell_type_df[cell_type_df["bayes_factor"] > bayesCutoff]
        cell_type_df = cell_type_df[
            cell_type_df["non_zeros_proportion1"] > nonZeroProportion
        ]
        if not markerCounts:
            markers[c] = cell_type_df.index.tolist()
        else:
            markers[c] = cell_type_df.index.tolist()[:markerCounts]

    if not keyAdded:
        keyAdded = f"scvi_marker_{groupby}"
    adata.uns[keyAdded] = markers


def labelTransferByScanvi(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    needLoc: bool = False,
    ls_removeCateKey: Optional[List[str]] = [],
    dt_params2Model={},
    cutoff: float = 0.9,
    keyAdded: Optional[str] = None,
    max_epochs: int = 1000,
    threads: int = 24,
    mode: Literal["merge", "online"] = "online",
) -> Optional[anndata.AnnData]:
    """
    annotate queryAd based on refAd annotation result.

    Parameters
    ----------
    refAd : anndata.AnnData
    refLabel : str
    refLayer : str
        raw count
    queryAd : anndata.AnnData
    queryLayer : str
        raw count
    needLoc : bool, optional
        if True, and `copy` is False, integrated anndata will be returned. by default False
    ls_removeCateKey : Optional[List[str]], optional
        These categories will be removed, the first one must be 'batch', by default []
    dt_params2Model : dict, optional
        by default {}
    cutoff : float, optional
        by default 0.9
    keyAdded : Optional[str], optional
        by default None
    max_epochs : int, optional
        by default 1000
    threads : int, optional
        by default 24
    mode: Literal['merge', 'online']
        by default 'online'

    Returns
    -------
    Optional[anndata.AnnData]
        based on needloc
    """
    import pandas as pd
    import numpy as np
    import scanpy as sc
    import scvi

    scvi.settings.seed = 39
    scvi.settings.num_threads = threads

    queryAdOrg = queryAd
    refAd = basic.getPartialLayersAdata(
        refAd, refLayer, [refLabel, *ls_removeCateKey]
    )
    queryAd = basic.getPartialLayersAdata(queryAd, queryLayer, ls_removeCateKey)
    refAd, queryAd = basic.getOverlap(refAd, queryAd)
    if not ls_removeCateKey:
        ls_removeCateKey = ["_batch"]

    queryAd.obs[refLabel] = "unknown"
    ad_merge = sc.concat([refAd, queryAd], label="_batch", keys=["ref", "query"])
    ad_merge.X = ad_merge.X.astype(int)
    sc.pp.highly_variable_genes(
        ad_merge,
        flavor="seurat_v3",
        n_top_genes=3000,
        batch_key="_batch",
        subset=True,
    )

    refAd = refAd[:, ad_merge.var.index].copy()
    queryAd = queryAd[:, ad_merge.var.index].copy()

    if mode == "online":
        # train model
        scvi.data.setup_anndata(
            refAd,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )
        scvi.data.setup_anndata(
            queryAd,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )

        scvi_model = scvi.model.SCVI(refAd, **dt_params2Model)
        scvi_model.train(max_epochs=max_epochs)

        lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, "unknown")
        lvae.train(max_epochs=max_epochs)

        # plot result on training dataset
        refAd.obs[f"labelTransfer_scanvi_{refLabel}"] = lvae.predict(refAd)
        refAd.obsm["X_scANVI"] = lvae.get_latent_representation(refAd)
        sc.pp.neighbors(refAd, use_rep="X_scANVI")
        sc.tl.umap(refAd)
        sc.pl.umap(refAd, color=refLabel, legend_loc="on data")
        df_color = basic.getadataColor(refAd, refLabel)
        refAd = basic.setadataColor(
            refAd, f"labelTransfer_scanvi_{refLabel}", df_color
        )
        sc.pl.umap(
            refAd, color=f"labelTransfer_scanvi_{refLabel}", legend_loc="on data"
        )

        # online learning
        lvae_online = scvi.model.SCANVI.load_query_data(
            queryAd,
            lvae,
        )
        lvae_online._unlabeled_indices = np.arange(queryAd.n_obs)
        lvae_online._labeled_indices = []
        lvae_online.train(max_epochs=max_epochs, plan_kwargs=dict(weight_decay=0.0))

    elif mode == "merge":
        scvi.data.setup_anndata(
            ad_merge,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )
        scvi_model = scvi.model.SCVI(ad_merge, **dt_params2Model)
        scvi_model.train(max_epochs=max_epochs)

        lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, "unknown")
        lvae.train(max_epochs=max_epochs)

        ad_merge.obsm["X_scANVI"] = lvae.get_latent_representation(ad_merge)
        sc.pp.neighbors(ad_merge, use_rep="X_scANVI")
        sc.tl.umap(ad_merge)
        sc.pl.umap(ad_merge, color=refLabel, legend_loc="on data")

        lvae_online = lvae

    else:
        assert False, "Unknown `mode`"

    # plot result on both dataset
    ad_merge.obs[f"labelTransfer_scanvi_{refLabel}"] = lvae_online.predict(ad_merge)
    ad_merge.obsm["X_scANVI"] = lvae_online.get_latent_representation(ad_merge)
    sc.pp.neighbors(ad_merge, use_rep="X_scANVI")
    sc.tl.umap(ad_merge)
    df_color = basic.getadataColor(refAd, refLabel)
    ad_merge = basic.setadataColor(
        ad_merge, f"labelTransfer_scanvi_{refLabel}", df_color
    )
    sc.pl.umap(ad_merge, color="_batch")
    sc.pl.umap(ad_merge, color=refLabel)
    sc.pl.umap(
        ad_merge, color=f"labelTransfer_scanvi_{refLabel}", legend_loc="on data"
    )

    # get predicted labels
    if not keyAdded:
        keyAdded = f"labelTransfer_scanvi_{refLabel}"
    queryAdOrg.obsm[f"{keyAdded}_score"] = lvae_online.predict(queryAd, soft=True)
    queryAdOrg.obs[keyAdded] = queryAdOrg.obsm[f"{keyAdded}_score"].pipe(
        lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown")
    )
    if needLoc:
        return ad_merge