from logging import log
import scvi
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
from ..rTools import rcontext
from ..otherTools import F


@rcontext
def cellTypeAnnoByIci(
    ad,
    layer,
    ad_bulk,
    bulkLayer,
    label,
    dt_kwargsToComputeSpec={},
    specThreads=12,
    minSpecScore=0.15,
    ar_informationRange=np.linspace(0, 50, 51),
    nSamples=200,
    informationSelect: Literal["auto", "manual"] = "auto",
    dt_kwargsToComputeIciScore={},
    iciThreads=4,
    keyAdded="ici",
    rEnv=None,
):
    """`cellTypeAnnoByIci` is a function that takes in a single cell RNA-seq dataset and a bulk RNA-seq
    dataset, and returns a dataframe with the ICI score for each cell

    Parameters
    ----------
    ad
        AnnData object
    layer
        the layer of the adata object that you want to annotate. log-normalized
    ad_bulk
        AnnData object of the bulk data
    bulkLayer
        the layer of the bulk adata object that you want to use to compute the ICI score. log-normalized
    label
        the name of the cell type recorded the annotation results
    dt_kwargsToComputeSpec
        dict of kwargs to pass to `compute_spec_table`
    specThreads, optional
        number of threads to use for computing specificity
    minSpecScore
        minimum specificity score to be considered a cell type
    ar_informationRange
        the range of information values to consider.
    nSamples, optional
        number of samples to use for computing the ICI score
    informationSelect : Literal["auto", "manual"], optional
        Literal["auto", "manual"] = "auto"
    dt_kwargsToComputeIciScore
        dict of kwargs to pass to `compute_ici_scores`
    iciThreads, optional
        number of threads to use for computing ICI scores
    keyAdded, optional
        the key to add to the adata object
    rEnv
        R environment to use. If None, will create a new one.

    """
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from ..rTools import py2r, r2py

    ICITools = importr("ICITools")
    future = importr("future")
    tibble = importr("tibble")
    future.plan(strategy="multiprocess", workers=specThreads)
    R = ro.r
    print(2)
    assert informationSelect in [
        "auto",
        "manual",
    ], "informationSelect must be either 'auto' or 'manual'"
    df_exp = (
        ad_bulk.to_df(bulkLayer)
        .join(ad_bulk.obs[label].rename("Cell_Type"))
        .rename_axis("Sample_Name")
        .reset_index()
        .melt(["Sample_Name", "Cell_Type"], var_name="Locus", value_name="Expression")
        .reindex(columns=["Locus", "Cell_Type", "Sample_Name", "Expression"])
    )
    df_exp = df_exp.sort_values(["Sample_Name", "Locus"])
    df_exp["Cell_Type"] = df_exp["Cell_Type"].astype(str)
    df_exp["Sample_Name"] = df_exp["Sample_Name"].astype(str)

    dfR_exp = py2r(df_exp)
    dfR_exp = R.as_tibble(dfR_exp)
    dfR_geneSpec = ICITools.compute_spec_table(dfR_exp, **dt_kwargsToComputeSpec)
    df_geneSpec = r2py(dfR_geneSpec)
    sns.displot(df_geneSpec["spec"])
    plt.show()
    plt.close()
    print("Gene counts:")
    print(
        df_geneSpec.query("spec >= @minSpecScore")
        .value_counts("Cell_Type")
        .to_markdown()
    )

    df_scExp = ad.to_df(layer).rename_axis(columns="Locus").T.reset_index()
    dfR_scExp = py2r(df_scExp)

    rEnv["dfR_geneSpec"] = dfR_geneSpec
    rEnv["dfR_scExp"] = dfR_scExp
    rEnv["ar_informationRange"] = py2r(ar_informationRange)
    rEnv["minSpecScore"] = minSpecScore
    rEnv["nSamples"] = nSamples
    R(
        """
    library(dplyr)
    library(ggplot2)
    info <- gather_information_level_data(dfR_scExp, 
                                        spec_table = dfR_geneSpec, 
                                        information_range = ar_informationRange, 
                                        min_spec_score = minSpecScore, 
                                        n_samples = nSamples)

    ici_signal <- extract_signal_from_info_data(info)
    ici_var <- extract_var_from_info_data(info)

    info_summary <- dplyr::inner_join(ici_var, ici_signal)
    """
    )
    df_infoSummary = r2py(R("info_summary"))
    sns.lineplot(
        data=df_infoSummary.melt("information_level", ["signal", "variability"]),
        x="information_level",
        y="value",
        hue="variable",
    )
    plt.show()
    plt.close()
    if informationSelect == "auto":
        informationLevel = (
            df_infoSummary.groupby("information_level")["signal"].mean().idxmax()
        )
    elif informationSelect == "manual":
        informationLevel = float(input())
    future.plan(strategy="multiprocess", workers=iciThreads)

    dfR_iciScore = ICITools.compute_ici_scores(
        expression_data=dfR_scExp,
        spec_table=dfR_geneSpec,
        sig=True,
        n_iterations=1000,
        information_level=int(informationLevel),
        min_spec_score=minSpecScore,
        **dt_kwargsToComputeIciScore,
    )
    df_iciScore = r2py(dfR_iciScore)
    df_iciScore = df_iciScore.pivot_table(index="Cell", columns="Cell_Type")
    for x in ["ici_score", "ici_score_norm", "p_adj", "p_val"]:
        ad.obsm[f"{keyAdded}_{x}"] = df_iciScore.loc[:, x].copy()


def getOverlapBetweenPrividedMarkerAndSpecificGene(
    adata: anndata.AnnData,
    markerDt: Mapping[str, Sequence[str]],
    key: str = "rank_genes_groups",
    min_in_group_fraction: float = 0.5,
    max_out_group_fraction: float = 0.25,
    minFc: float = 0.585,
) -> Mapping[str, Sequence[str]]:
    minLog2Fc = np.exp2(minFc)
    specificMarkerDf = sc.get.rank_genes_groups_df(adata, None, key=key).query(
        "logfoldchanges >= @minLog2Fc & pct_nz_group >= @min_in_group_fraction & pct_nz_reference <= @max_out_group_fraction"
    )
    specificMarkerLs = set(specificMarkerDf["names"])
    markerDt = {x: (set(y) & specificMarkerLs) for x, y in markerDt.items()}

    delKeyLs = []
    for x, y in markerDt.items():
        if not y:
            logger.warning(f"Specific genes dont have any overlap with cell type <{x}>")
            delKeyLs.append(x)
    [markerDt.pop(x) for x in delKeyLs]

    return markerDt


def cellTypeAnnoByCorr(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    cutoff: float = 0.5,
    groups: List = [],
    method: Literal["pearson", "spearman"] = "spearman",
    keyAdded: str = None,
):
    """
    annotate queryAd based on refAd annotation result.
    ----------
    refAd : anndata.AnnData
    refLabel : str
    refLayer : str
        must be log-transformed
    queryAd : anndata.AnnData
    queryLayer : str
        must be log-transformed
    cutoff : float, optional
        by default 0.01
    method : Literal["pearson", "spearman"]
        by default "spearman"
    groups : List, optional
        by default []
    keyAdded : str, optional
        by default None
    """
    from scipy.stats import mstats

    if (not refLayer) | (refLayer == "X"):
        df_refGene = basic.mergeadata(refAd, refLabel, [], "mean").to_df()
    else:
        df_refGene = basic.mergeadata(refAd, refLabel, [refLayer], "mean").to_df(
            refLayer
        )

    if groups:
        df_refGene = df_refGene.reindex(index=groups)

    df_refGene = df_refGene.reindex(index=groups)

    if not queryLayer:
        queryLayer = "X"
    df_queryGene = queryAd.to_df() if queryLayer == "X" else queryAd.to_df(queryLayer)

    sr_overlap = df_queryGene.columns & df_refGene.columns
    df_refGene = df_refGene.reindex(columns=sr_overlap)
    df_queryGene = df_queryGene.reindex(columns=sr_overlap)
    ix_Ref = df_refGene.index

    if method == "spearman":
        ar_Ref = mstats.rankdata(df_refGene, axis=1)
        ar_query = mstats.rankdata(df_queryGene, axis=1)
    elif method == "pearson":
        ar_Ref = df_refGene.values
        ar_query = df_queryGene.values
    else:
        assert False, f"unimplemented method: {method}"

    ar_corr = np.corrcoef(ar_Ref, ar_query)[-len(ar_query) :, : len(ar_Ref)]
    df_corr = pd.DataFrame(ar_corr, index=queryAd.obs.index, columns=ix_Ref)

    if not keyAdded:
        keyAdded = f"corr_{method}_{refLabel}"

    queryAd.obsm[f"{keyAdded}_corr"] = df_corr
    queryAd.obs[f"{keyAdded}"] = np.select(
        [df_corr.max(1) > cutoff], [df_corr.idxmax(1)], "unknown"
    )


def labelTransferByCellId(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    query_batch_key: Optional[str] = None,
    markerCount: int = 200,
    n_top_genes: int = 5000,
    ls_use_gene: Optional[List[str]] = None,
    cutoff: float = 2.0,
    nmcs: int = 30,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    annotate queryAd based on refAd annotation result.

    Parameters
    ----------
    refAd : anndata.AnnData
    refLabel : str
        column's name in refAd.obs
    refLayer : str
        must be raw data
    queryAd : anndata.AnnData
    queryLayer : str
        must be raw data
    markerCount : int
        Gene number extracted from refAd. These gene will be used to annotate queryAd
    cutoff : float, optional
        by default 2.0
    copy : bool, optional
        by default False

    Returns
    -------
    Optional[anndata.AnnData]
        if copy is False, queryAd will be updated as following rules:
            obsm will be updated by f"cellid_{refLabel}_labelTranferScore".
            obs will be updated by f"cellid_{refLabel}_labelTranfer"
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot, rHelp
    from . import geneEnrichInfo

    rBase = importr("base")
    rUtils = importr("utils")
    cellId = importr("CelliD")
    R = ro.r
    queryAd_org = queryAd.copy() if copy else queryAd
    refAd, queryAd = basic.getOverlap(refAd, queryAd, copy=True)
    refAd.X = refAd.layers[refLayer] if refLayer not in ["X", None] else refAd.X
    queryAd.X = (
        queryAd.layers[queryLayer] if queryLayer not in ["X", None] else queryLayer.X
    )
    ad_integrated = sc.concat(
        {"ref": refAd, "query": queryAd}, label="batch_cellid", index_unique="-"
    )
    if not ls_use_gene:
        sc.pp.highly_variable_genes(
            ad_integrated,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            batch_key="batch_cellid",
            subset=True,
        )
        ls_useGene = ad_integrated.var.index.to_list()
    else:
        ls_useGene = ls_use_gene

    sc.pp.normalize_total(refAd, 1e4)
    sc.pp.normalize_total(queryAd, 1e4)
    refAd = refAd[:, ls_useGene].copy()
    queryAd = queryAd[:, ls_useGene].copy()

    VectorR_Refmarker = geneEnrichInfo.getEnrichedGeneByCellId(
        refAd,
        "X",
        refLabel,
        markerCount,
        copy=True,
        returnR=True,
        nmcs=nmcs,
    )

    if not query_batch_key:
        _ad = basic.getPartialLayersAdata(queryAd, ["X"])
        sc.pp.scale(_ad, max_value=10)
        adR_query = py2r(_ad)
        adR_query = cellId.RunMCA(adR_query, slot="X", nmcs=nmcs)
        df_labelTransfered = r2py(
            rBase.data_frame(
                cellId.RunCellHGT(
                    adR_query, VectorR_Refmarker, dims=py2r(np.arange(1, 1 + nmcs))
                ),
                check_names=False,
            )
        ).T
    else:
        lsDf_labelTransfered = []
        for _ad in basic.splitAdata(queryAd, query_batch_key):
            _ad = basic.getPartialLayersAdata(_ad, ["X"])
            sc.pp.scale(_ad, max_value=10)
            adR_query = py2r(_ad)
            adR_query = cellId.RunMCA(adR_query, slot="X", nmcs=nmcs)
            df_labelTransfered = r2py(
                rBase.data_frame(
                    cellId.RunCellHGT(
                        adR_query, VectorR_Refmarker, dims=py2r(np.arange(1, 1 + nmcs))
                    ),
                    check_names=False,
                )
            ).T
            lsDf_labelTransfered.append(df_labelTransfered)
        df_labelTransfered = pd.concat(lsDf_labelTransfered).reindex(
            queryAd_org.obs.index
        )

    queryAd_org.obsm[f"cellid_{refLabel}_labelTranferScore"] = df_labelTransfered
    queryAd_org.obs[f"cellid_{refLabel}_labelTranfer"] = queryAd_org.obsm[
        f"cellid_{refLabel}_labelTranferScore"
    ].pipe(lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown"))
    if copy:
        return queryAd_org


def labelTransferByScanpy(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    features: Optional[None] = None,
    method: Literal["harmony", "scanorama"] = "harmony",
    cutoff: float = 0.5,
    npcs: int = 30,
    copy: bool = False,
    needLoc: bool = False,
) -> Optional[anndata.AnnData]:
    """
    annotate queryAd based on refAd annotation result.

    Parameters
    ----------
    refAd : anndata.AnnData
    refLabel : str
    refLayer : str
        log-transformed normalized
    queryAd : anndata.AnnData
    queryLayer : str
        log-transformed normalized
    features : Optional[None]
        list, used gene to DR
    method : Literal['harmony', 'scanorama'], optional
        by default 'harmony'
    cutoff : float, optional
        by default 0.5
    npcs : int, optional
        by default 30
    copy : bool, optional
        by default False
    needLoc: bool, optional
        if True, and `copy` is False, intregated anndata will be returned

    Returns
    -------
    Optional[anndata.AnnData]
    """
    from sklearn.metrics.pairwise import cosine_distances
    import scanpy.external as sce

    def label_transfer(dist, labels):
        lab = pd.get_dummies(labels).to_numpy().T
        class_prob = lab @ dist
        norm = np.linalg.norm(class_prob, 2, axis=0)
        class_prob = class_prob / norm
        class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
        return class_prob

    queryAdOrg = queryAd.copy() if copy else queryAd
    refAd = basic.getPartialLayersAdata(refAd, refLayer, [refLabel])
    queryAd = basic.getPartialLayersAdata(queryAd, queryLayer)

    ls_overlapGene = refAd.var.index & queryAd.var.index
    queryAd = queryAd[:, ls_overlapGene]
    refAd = refAd[:, ls_overlapGene]

    ad_concat = sc.concat(
        [queryAd, refAd],
        label="batch",
        keys=["query", "ref"],
        index_unique="-batch-",
    )
    if not features:
        features = basic.scIB_hvg_batch(ad_concat, "batch")
    ad_concat.var["highly_variable"] = ad_concat.var.index.isin(features)
    sc.pp.scale(ad_concat)
    sc.tl.pca(ad_concat)
    if method == "harmony":
        sce.pp.harmony_integrate(ad_concat, key="batch")
        useObsmKey = "X_pca_harmony"
    elif method == "scanorama":
        sce.pp.scanorama_integrate(ad_concat, key="batch")
        useObsmKey = "X_scanorama"
    else:
        assert False, f"Unrealized method ：{method}"

    ad_concat.obsm["temp"] = ad_concat.obsm[useObsmKey][:, :npcs]
    sc.pp.neighbors(ad_concat, n_pcs=20, use_rep="temp")
    sc.tl.umap(ad_concat)
    sc.pl.umap(ad_concat, color="batch")
    refAd.obs.index = refAd.obs.index + "-batch-ref"
    ad_concat.obs = ad_concat.obs.join(refAd.obs[refLabel])

    # import pdb; pdb.set_trace()
    dt_color = basic.getadataColor(refAd, refLabel)
    dt_color["unknown"] = "#111111"
    ad_concat = basic.setadataColor(ad_concat, refLabel, dt_color)
    sc.pl.umap(ad_concat, color=refLabel, legend_loc="on data")

    distances = 1 - cosine_distances(
        ad_concat[ad_concat.obs["batch"] == "ref"].obsm["temp"],
        ad_concat[ad_concat.obs["batch"] == "query"].obsm["temp"],
    )
    del ad_concat.obsm["temp"]

    class_prob = label_transfer(
        distances, ad_concat[ad_concat.obs["batch"] == "ref"].obs[refLabel]
    )
    df_labelScore = pd.DataFrame(
        class_prob,
        columns=np.sort(
            ad_concat[ad_concat.obs["batch"] == "ref"].obs[refLabel].unique()
        ),
    )
    df_labelScore.index = ad_concat[ad_concat.obs["batch"] == "query"].obs.index
    df_labelScore = df_labelScore.rename(index=lambda x: x.rstrip("-batch-query"))
    queryAdOrg.obsm[
        f"labelTransfer_score_scanpy_{method}_{refLabel}"
    ] = df_labelScore.reindex(queryAdOrg.obs.index)
    queryAdOrg.obs[f"labelTransfer_scanpy_{method}_{refLabel}"] = queryAdOrg.obsm[
        f"labelTransfer_score_scanpy_{method}_{refLabel}"
    ].pipe(lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown"))
    if copy:
        return queryAdOrg
    if needLoc:
        return ad_concat

@rcontext
def labelTransferBySeurat(
    ad_ref: sc.AnnData,
    refLabel: str,
    refLayer,
    ad_query: sc.AnnData,
    queryLayer,
    nTopGenes=2000,
    kScore=30,
    dims=30,
    kWeight=100,
    refIsIntegrated=False,
    rEnv=None,
) -> anndata.AnnData:
    """
    annotate queryAd based on refAd annotation result.

    Parameters
    ----------
    ad_ref : sc.AnnData
    refLabel : str
    refLayer : str, optional
        raw
    ad_query : [type], optional
    queryLayer : str, optional
        raw
    nTopGenes : int, optional
        by default 2000
    kScore : int, optional
        by default 30
    dims : int, optional
        by default 30
    kWeight : int, optional
        by default 100
    refIsIntegrated: bool
        if True, refLayer is already integrated by seurat

    Returns
    -------
    anndata.AnnData
        intregated anndata
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rGet, rSet, so2ad, ad2so
    from ..otherTools import setSeed
    from .normalize import integrateBySeurat

    rBase = importr("base")
    rUtils = importr("utils")
    R = ro.r
    seurat = importr("Seurat")
    seuratObject = importr("SeuratObject")
    setSeed(0)

    if not refIsIntegrated:
        ad_ref.layers["_raw"] = ad_ref.layers[refLayer].copy()
        ad_query.layers["_raw"] = ad_query.layers[queryLayer].copy()
        ad_concat = sc.concat(
            {"ref": ad_ref, "query": ad_query}, label="seurat_", index_unique="-"
        )
        sc.pp.highly_variable_genes(
            ad_concat,
            layer="_raw",
            n_top_genes=nTopGenes,
            batch_key="seurat_",
            subset=True,
            flavor="seurat_v3",
        )
        ls_features = ad_concat.var.index.str.replace("_", "-").to_list()
        so_ref = ad2so(ad_ref, refLayer, ls_obs=refLabel)
        so_query = ad2so(ad_query, queryLayer, ls_obs=[])
        lsR_features = R.c(*ls_features)
        rEnv["so_ref"] = so_ref
        rEnv["so_query"] = so_query
        rEnv["lsR_features"] = lsR_features
        rEnv["k.score"] = kScore
        rEnv["dims"] = dims
        rEnv["k.weight"] = kWeight

        R(
            f"""
        anchors <- FindTransferAnchors(reference = so_ref, query = so_query, dims = 1:dims, k.score=k.score,features=lsR_features)
        predictions <- TransferData(anchorset = anchors, refdata = so_ref${refLabel}, dims = 1:dims, k.weight=k.weight)
        """
        )

        df_pred = r2py(rEnv["predictions"])

        ad_query.obsm[f"labelTransfer_seurat_{refLabel}"] = df_pred
        ad_query.obs[f"labelTransfer_seurat_{refLabel}"] = df_pred["predicted.id"]

        ad_ref.obs[f"labelTransfer_seurat_{refLabel}"] = ad_ref.obs[refLabel]
        ad_concat = sc.concat(
            {"ref": ad_ref, "query": ad_query}, label="seurat_", index_unique="-"
        )
        del ad_ref.obs[f"labelTransfer_seurat_{refLabel}"]

        ad_concat = integrateBySeurat(
            ad_concat,
            batch_key="seurat_",
            n_top_genes=nTopGenes,
            layer="_raw",
            k_score=kScore,
            dims=dims,
            k_weight=kWeight,
        )

        sc.pp.neighbors(ad_concat)
        sc.tl.umap(ad_concat)

        sc.pl.umap(
            ad_concat, color=[f"labelTransfer_seurat_{refLabel}", "seurat_"], wspace=0.5
        )
        ax = sc.pl.umap(ad_concat, size=12e4 / len(ad_concat), show=False)
        _ad = ad_concat[ad_concat.obs.eval('seurat_ == "query"')]
        sc.pl.umap(
            _ad,
            color=f"labelTransfer_seurat_{refLabel}",
            size=12e4 / len(ad_concat),
            ax=ax,
        )
    else:
        ad_ref = ad_ref.copy()
        ad_ref.var.index = ad_ref.var.index.str.replace("_", "-")
        ad_ref = ad_ref[:, ad_ref.obsm["seurat_integrated_data"].columns]
        ad_ref.layers["seurat_integrated_data"] = ad_ref.obsm["seurat_integrated_data"]
        if "seurat_integrated_scale.data" in ad_ref.obsm:
            assert (
                False
            ), "Not support SCT normalization, please rerun SCT and NOT transfer `seuratObject` to `AnnData` until label transfer is finished"
        else:
            refScaleLayer = None

        so_ref = ad2so(
            ad_ref,
            layer=refLayer,
            dataLayer="seurat_integrated_data",
            scaleLayer=refScaleLayer,
        )
        so_query = ad2so(ad_query, layer=queryLayer)
        rEnv["so_ref"] = so_ref
        rEnv["so_query"] = so_query
        rEnv["k.score"] = kScore
        rEnv["dims"] = dims
        rEnv["k.weight"] = kWeight
        rEnv["lsR_features"] = R.c(*ad_ref.var.index.str.replace("_", "-").to_list())
        R(
            f"""
        anchors <- FindTransferAnchors(reference = so_ref, query = so_query, dims = 1:dims, k.score=k.score, features=lsR_features)
        predictions <- TransferData(anchorset = anchors, refdata = so_ref${refLabel}, dims = 1:dims, k.weight=k.weight)
        """
        )

        df_pred = r2py(rEnv["predictions"])

        ad_query.obsm[f"labelTransfer_seurat_{refLabel}"] = df_pred
        ad_query.obs[f"labelTransfer_seurat_{refLabel}"] = df_pred["predicted.id"]

        ad_ref.obs[f"labelTransfer_seurat_{refLabel}"] = ad_ref.obs[refLabel]
        ad_concat = sc.concat(
            {"ref": ad_ref, "query": ad_query}, label="seurat_", index_unique="-"
        )

    return ad_concat


def labelTransferByScanvi(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    needLoc: bool = False,
    ls_removeCateKey: Optional[List[str]] = [],
    dt_params2SCVIModel={},
    dt_params2SCANVIModel={},
    cutoff: float = 0.95,
    keyAdded: Optional[str] = None,
    max_epochs: int = 1000,
    max_epochs_scanvi: Optional[int] = None,
    threads: int = 24,
    mode: Literal["merge", "online"] = "online",
    n_top_genes=3000,
    early_stopping: bool = True,
    batch_size_ref: int = 128,
    batch_size_query: int = 128,
    hvgBatch='',
) -> Optional[anndata.AnnData]:
    """
    Performs label transfer from a reference dataset to a query dataset using the scanvi library.
        
    Parameters
    ----------
    refAd: anndata.AnnData
        The reference dataset.
    refLabel: str
        The categorical column name in the reference dataset to use for labels.
    refLayer: str
        The layer to filter the reference dataset by.
    queryAd: anndata.AnnData
        The query dataset.
    queryLayer: str
        The layer to filter the query dataset by.
    needLoc: bool, optional
        If True, modifies the original queryAd object in place and returns None. Otherwise, returns an AnnData object with the transferred labels added.
        Default is False.
    ls_removeCateKey: List[str], optional
        A list of categorical column names to remove from both refAd and queryAd before performing the label transfer.
        Default is [].
    dt_params2SCVIModel: dict, optional
        A dictionary of parameters to pass to the SCVI model when training it.
        Default is {}.
    dt_params2SCANVIModel: dict, optional
        A dictionary of parameters to pass to the SCANVI model when training it.
        Default is {}.
    cutoff: float, optional
        The minimum probability required for a cell to be labeled.
        Must be a float value between 0 and 1.
        Default is 0.95.
    keyAdded: str, optional
        The name of the column to use for the transferred labels in the returned AnnData object.
        Default is None.
    max_epochs: int, optional
        The maximum number of epochs to use for training the SCVI and SCANVI models.
        Default is 1000.
    threads: int, optional
        The number of threads to use.
        Default is 24.
    mode: str, optional
        The mode of operation, either "merge" or "online".
        Default is "online".
    n_top_genes: int, optional
        The number of top genes to use.
        Default is 3000.
    early_stopping: bool, optional
        Whether to use early stopping when training the models.
        Default is True.
    batch_size_ref: int, optional
        an integer specifying the batch size to use when training the SCANVI model on the reference data.
        Default is 128.
    batch_size_query: int, optional
        an integer specifying the batch size to use when training the SCANVI model on the query data.
        Default is 128.
    hvgBatch: str, optional
        a string specifying a batch column name to use for highly variable gene selection.
    """
    scvi.settings.seed = 39
    scvi.settings.num_threads = threads


    queryAdOrg = queryAd
    refAd = basic.getPartialLayersAdata(refAd, refLayer, [refLabel, *ls_removeCateKey, hvgBatch] >> F(filter, lambda x: x) >> F(set) >> F(list))
    queryAd = basic.getPartialLayersAdata(queryAd, queryLayer, [*ls_removeCateKey, hvgBatch] >> F(filter, lambda x: x) >> F(set) >> F(list))
    refAd, queryAd = basic.getOverlap(refAd, queryAd)
    if not ls_removeCateKey:
        ls_removeCateKey = ["_batch"]
    if not hvgBatch:
        hvgBatch = ls_removeCateKey[0]
    queryAd.obs[refLabel] = "unknown"
    refAd.obs["_batch"] = "ref"
    queryAd.obs["_batch"] = "query"
    ad_merge = sc.concat([refAd, queryAd], label="_batch", keys=["ref", "query"])
    ad_merge.X = ad_merge.X.astype(int)
    sc.pp.highly_variable_genes(
        ad_merge,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key=hvgBatch,
        subset=True,
    )

    refAd = refAd[:, ad_merge.var.index].copy()
    queryAd = queryAd[:, ad_merge.var.index].copy()

    if mode == "online":
        # train model
        scvi.model.SCVI.setup_anndata(
            refAd,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )
        scvi.model.SCVI.setup_anndata(
            queryAd,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )

        scvi_model = scvi.model.SCVI(refAd, **dt_params2SCVIModel)
        scvi_model.train(
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            batch_size=batch_size_ref,
        )

        lvae = scvi.model.SCANVI.from_scvi_model(
            scvi_model, "unknown", **dt_params2SCANVIModel
        )
        lvae.train(max_epochs=max_epochs_scanvi, batch_size=batch_size_ref)
        lvae.history["elbo_train"].plot()
        plt.yscale("log")
        plt.show()
        # plot result on training dataset
        refAd.obs[f"labelTransfer_scanvi_{refLabel}"] = lvae.predict(refAd)
        refAd.obsm["X_scANVI"] = lvae.get_latent_representation(refAd)
        sc.pp.neighbors(refAd, use_rep="X_scANVI")
        sc.tl.umap(refAd, min_dist=0.2)

        ax = sc.pl.umap(refAd, color=refLabel, show=False)
        sc.pl.umap(refAd, color=refLabel, legend_loc="on data", ax=ax)

        df_color = basic.getadataColor(refAd, refLabel)
        refAd = basic.setadataColor(refAd, f"labelTransfer_scanvi_{refLabel}", df_color)
        ax = sc.pl.umap(refAd, color=f"labelTransfer_scanvi_{refLabel}", show=False)
        sc.pl.umap(
            refAd, color=f"labelTransfer_scanvi_{refLabel}", legend_loc="on data", ax=ax
        )

        # online learning
        lvae_online = scvi.model.SCANVI.load_query_data(
            queryAd,
            lvae,
        )
        lvae_online._unlabeled_indices = np.arange(queryAd.n_obs)
        lvae_online._labeled_indices = []
        lvae_online.train(
            max_epochs=max_epochs_scanvi,
            plan_kwargs=dict(weight_decay=0.0),
            batch_size=batch_size_query,
        )
        lvae_online.history["elbo_train"].plot()
        plt.yscale("log")
        plt.show()
        ad_merge.obsm["X_scANVI"] = lvae_online.get_latent_representation(ad_merge)
        sc.pp.neighbors(ad_merge, use_rep="X_scANVI")
        sc.tl.umap(ad_merge, min_dist=0.2)
    elif mode == "merge":
        sc.pp.subsample(ad_merge, fraction=1)  # scvi do not shuffle adata
        scvi.model.SCVI.setup_anndata(
            ad_merge,
            layer=None,
            labels_key=refLabel,
            batch_key=ls_removeCateKey[0],
            categorical_covariate_keys=ls_removeCateKey[1:],
        )
        scvi_model = scvi.model.SCVI(ad_merge, **dt_params2SCVIModel)
        scvi_model.train(
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            batch_size=batch_size_ref,
        )
        scvi_model.history["elbo_train"].plot()
        plt.yscale("log")
        plt.show()

        lvae = scvi.model.SCANVI.from_scvi_model(
            scvi_model, "unknown", **dt_params2SCANVIModel
        )
        lvae.train(
            max_epochs=max_epochs_scanvi,
            early_stopping=early_stopping,
            batch_size=batch_size_ref,
        )
        lvae.history["elbo_train"].plot()
        plt.yscale("log")
        plt.show()

        ad_merge.obsm["X_scANVI"] = lvae.get_latent_representation(ad_merge)
        sc.pp.neighbors(ad_merge, use_rep="X_scANVI")
        sc.tl.umap(ad_merge, min_dist=0.2)

        ax = sc.pl.umap(ad_merge, color=refLabel, show=False)
        sc.pl.umap(ad_merge, color=refLabel, legend_loc="on data", ax=ax)

        lvae_online = lvae

    else:
        assert False, "Unknown `mode`"

    # plot result on both dataset
    ad_merge.obs[f"labelTransfer_scanvi_{refLabel}"] = lvae_online.predict(ad_merge)

    dt_color = basic.getadataColor(refAd, refLabel)
    ad_merge = basic.setadataColor(
        ad_merge, f"labelTransfer_scanvi_{refLabel}", dt_color
    )
    dt_color["unknown"] = "#000000"
    dt_color = basic.setadataColor(ad_merge, refLabel, dt_color)
    sc.pl.umap(ad_merge, color="_batch")

    ax = sc.pl.umap(
        ad_merge,
        color=refLabel,
        show=False,
        groups=[x for x in ad_merge.obs[refLabel].unique() if x != "unknown"],
    )
    sc.pl.umap(
        ad_merge,
        color=refLabel,
        legend_loc="on data",
        ax=ax,
        groups=[x for x in ad_merge.obs[refLabel].unique() if x != "unknown"],
    )

    ax = sc.pl.umap(ad_merge, color=f"labelTransfer_scanvi_{refLabel}", show=False)
    sc.pl.umap(
        ad_merge, color=f"labelTransfer_scanvi_{refLabel}", legend_loc="on data", ax=ax
    )

    ax = sc.pl.umap(ad_merge, show=False)
    _ad = ad_merge[ad_merge.obs.eval("_batch == 'query'")]
    sc.pl.umap(
        _ad, color=f"labelTransfer_scanvi_{refLabel}", size=12e4 / len(ad_merge), ax=ax
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


def labelTransferByCelltypist(
    ad_ref: sc.AnnData,
    refLayer: str,
    refLabel: str,
    ad_query: sc.AnnData,
    queryLayer: str,
    mode: Literal["best match", "prob match"] = "prob match",
    model = None,
    dt_kwargs2train=dict(
        feature_selection=True,
        top_genes=300,
        use_SGD=True,
        mini_batch=True,
        balance_cell_type=True,
    ),
    dt_kwargs2annotate=dict(majority_voting=True, over_clustering=None),
):
    """> This function takes in two AnnData objects, one for training and one for prediction, and returns a
    prediction object from celltypist

    Parameters
    ----------
    ad_ref : sc.AnnData
        the reference dataset
    refLayer : str
        the layer in ad_ref that you want to use to train the model. must be 'raw'
    refLabel : str
        the label you want to transfer
    ad_query : sc.AnnData
        the query data
    queryLayer : str
        the layer in ad_query that you want to annotate, must be 'raw'
    dt_kwargs2train
        parameters for training the model
    dt_kwargs2annotate
        parameters for the celltypist.annotate function

    Returns
    -------
        (celltypist.classifier.AnnotationResult, model)
    """
    import celltypist

    ad_queryOrg = ad_query
    ad_ref, ad_query = basic.getOverlap(ad_ref, ad_query)
    ad_ref = ad_ref.copy()
    ad_query = ad_query.copy()
    ad_ref.X = ad_ref.layers[refLayer]
    sc.pp.normalize_total(ad_ref, target_sum=1e4)
    sc.pp.log1p(ad_ref)
    ad_query.X = ad_query.layers[queryLayer]
    sc.pp.normalize_total(ad_query, target_sum=1e4)
    sc.pp.log1p(ad_query)

    if model is None:
        model = celltypist.train(ad_ref, labels=refLabel, **dt_kwargs2train)

    predictions = celltypist.annotate(
        ad_query, model=model, mode=mode, **dt_kwargs2annotate
    )
    df_predResults = predictions.predicted_labels.rename(
        columns=lambda x: f"celltypist_{refLabel}_" + x
    )
    ad_queryOrg.obs = ad_queryOrg.obs.drop(
        columns=[x for x in df_predResults.columns if x in ad_queryOrg.obs.columns]
    ).join(df_predResults)
    if mode == "prob match":
        ad_queryOrg.obsm[f"celltypist_{refLabel}"] = predictions.probability_matrix
    return predictions, model