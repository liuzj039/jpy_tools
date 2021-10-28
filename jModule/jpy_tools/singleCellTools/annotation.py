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


def cellTypeAnnoByICI(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    cutoff: float = 0.01,
    threads: int = 24,
    groups: List = [],
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
    threads : int, optional
        by default 24
    groups : List, optional
        by default []
    keyAdded : str, optional
        by default None
    """
    from rpy2.robjects.packages import importr
    from .rTools import py2r, r2py

    icitools = importr("ICITools")
    future = importr("future")
    tibble = importr("tibble")
    future.plan(strategy="multiprocess", workers=threads)

    if (not refLayer) | (refLayer == "X"):
        df_refGene = basic.mergeadata(refAd, refLabel, "mean").to_df()
    else:
        df_refGene = basic.mergeadata(refAd, refLabel, [refLayer], "mean").to_df(
            refLayer
        )
    df_refGene = np.exp(df_refGene) - 1

    df_refGene = df_refGene.stack().reset_index()
    df_refGene.columns = ["Cell_Type", "Locus", "Expression"]
    df_refGene["Sample_Name"] = df_refGene["Cell_Type"]

    df_refGene = df_refGene.reindex(
        columns=["Locus", "Cell_Type", "Sample_Name", "Expression"]
    )

    if not groups:
        groups = list(df_refGene["Cell_Type"].unique())

    df_refGene = df_refGene.query("Cell_Type in @groups")

    dfR_spec = icitools.compute_spec_table(expression_data=py2r(df_refGene))

    if not queryLayer:
        queryLayer = "X"

    df_queryGene = queryAd.to_df() if queryLayer == "X" else queryAd.to_df(queryLayer)
    df_queryGene = np.exp(df_queryGene) - 1
    df_queryGene = df_queryGene.rename_axis(columns="Locus").T
    dfR_queryGene = py2r(df_queryGene)
    dfR_queryGene = tibble.as_tibble(dfR_queryGene, rownames="Locus")
    dfR_iciScore = icitools.compute_ici_scores(dfR_queryGene, dfR_spec, sig=True)
    df_iciScore = r2py(dfR_iciScore)
    df_iciScore = df_iciScore.pivot_table(
        index="Cell", columns="Cell_Type", values="ici_score_norm"
    )
    if not keyAdded:
        keyAdded = f"ici_{refLabel}"

    df_iciScore = df_iciScore.reindex(queryAd.obs.index)
    queryAd.obsm[f"{keyAdded}_normScore"] = df_iciScore
    queryAd.obs[f"{keyAdded}"] = np.select(
        [df_iciScore.max(1) > cutoff], [df_iciScore.idxmax(1)], "unknown"
    )


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
        df_labelTransfered = pd.concat(lsDf_labelTransfered).reindex(queryAd_org.obs.index)

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


def labelTransferBySeurat(
    refAd: anndata.AnnData,
    refLabel: str,
    refLayer: str,
    queryAd: anndata.AnnData,
    queryLayer: str,
    features: Optional[None] = None,
    npcs: int = 30,
    cutoff: float = 0.5,
    copy: bool = False,
    n_top_genes: int = 5000,
    needLoc: bool = False,
) -> Optional[anndata.AnnData]:
    """
    annotate queryAd based on refAd annotation result.

    Parameters
    ----------
    refAd : anndata.AnnData
    refLabel : str
    refLayer : str
        raw
    queryAd : anndata.AnnData
    queryLayer : str
        raw
    features : Optional[None]
        list, used gene to DR
    npcs : int, optional
        by default 30
    cutoff : float, optional
        by default 0.5
    copy : bool, optional
        Precedence over `needLoc`. by default False.
    needLoc: bool, optional
        if True, and `copy` is False, intregated anndata will be returned

    Returns
    -------
    Optional[anndata.AnnData]
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rGet, rSet
    from ..otherTools import setSeed

    rBase = importr("base")
    rUtils = importr("utils")
    R = ro.r
    seurat = importr("Seurat")
    seuratObject = importr("SeuratObject")
    setSeed(0)

    queryAdOrg = queryAd.copy() if copy else queryAd
    refAd = basic.getPartialLayersAdata(refAd, refLayer, [refLabel])
    queryAd = basic.getPartialLayersAdata(queryAd, queryLayer)
    queryAd.obs["empty"] = 0  # seurat need
    refAd, queryAd = basic.getOverlap(refAd, queryAd, copy=True)
    refAd.obs['__batch'] = 'reference'
    refAd.obs.index = 'reference-' + refAd.obs.index
    queryAd.obs['__batch'] = 'query'
    queryAd.obs.index = 'query-' + queryAd.obs.index
    ad_concat = sc.concat(
        {"ref": refAd, "query": queryAd}, label="__batch", index_unique="-batch-"
    )

    if not features:
        sc.pp.highly_variable_genes(
            ad_concat,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            batch_key="__batch",
            subset=True,
        )
        features = ad_concat.var.index.to_list()

    sc.pp.normalize_total(refAd, 1e4)
    sc.pp.normalize_total(queryAd, 1e4)

    ar_features = np.array(features)
    arR_features = py2r(ar_features)

    adR_query = py2r(queryAd)
    adR_query = seurat.as_Seurat_SingleCellExperiment(
        adR_query, counts=R("NULL"), data="X"
    )
    adR_query = seuratObject.RenameAssays(object=adR_query, originalexp="RNA")

    adR_ref = py2r(refAd)
    adR_ref = seurat.as_Seurat_SingleCellExperiment(adR_ref, counts=R("NULL"), data="X")
    adR_ref = seuratObject.RenameAssays(object=adR_ref, originalexp="RNA")

    adR_ref = seurat.ScaleData(trl(adR_ref))
    adR_query = seurat.ScaleData(trl(adR_query))
    anchors = seurat.FindTransferAnchors(
        reference=trl(adR_ref),
        query=trl(adR_query),
        dims=py2r(np.arange(0, npcs) + 1),
        features=arR_features,
    )

    predictions = seurat.TransferData(
        anchorset=anchors,
        refdata=rGet(adR_ref, "@meta.data", f"${refLabel}"),
        dims=py2r(np.arange(0, npcs) + 1),
        k_weight=10,
    )

    df_predScore = r2py(predictions)

    df_predScore = df_predScore[
        [
            x
            for x in df_predScore.columns
            if (x.startswith("prediction.score")) & (x != "prediction.score.max")
        ]
    ]
    df_predScore = df_predScore.rename(columns=lambda x: x.split("prediction.score.")[-1])

    dt_name2Org = {
        y: x
        for x, y in zip(
            sorted(list(refAd.obs[refLabel].unique())),
            sorted(list(df_predScore.columns)),
        )
    }

    df_predScore = df_predScore.rename(
        columns=dt_name2Org, index=lambda x: x.split("query-", 1)[1]
    )

    queryAdOrg.obsm[f"labelTransfer_score_seurat_{refLabel}"] = df_predScore.reindex(
        queryAdOrg.obs.index
    )

    queryAdOrg.obs[f"labelTransfer_seurat_{refLabel}"] = queryAdOrg.obsm[
        f"labelTransfer_score_seurat_{refLabel}"
    ].pipe(lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown"))

    rSet(
        adR_ref,
        rBase.as_matrix(rGet(adR_ref, "@assays", "$RNA", "@data")),
        "@assays",
        "$RNA",
        "@data",
    )
    rSet(
        adR_query,
        rBase.as_matrix(rGet(adR_query, "@assays", "$RNA", "@data")),
        "@assays",
        "$RNA",
        "@data",
    )
    anchor = seurat.FindIntegrationAnchors(
        trl(R.list(adR_query, adR_ref)),
        anchor_features=trl(arR_features),
        dims=py2r(np.arange(0, npcs) + 1),
    )
    adR_integrated = seurat.IntegrateData(
        anchorset=trl(anchor), normalization_method="LogNormalize"
    )
    adR_integrated = seurat.ScaleData(trl(adR_integrated))
    adR_integrated = seurat.RunPCA(object=trl(adR_integrated), features=arR_features)
    adR_integrated = seurat.RunUMAP(
        object=trl(adR_integrated), dims=py2r(np.arange(0, npcs) + 1)
    )

    ad_integrated = r2py(seurat.as_SingleCellExperiment(trl(adR_integrated)))

    ad_integrated.obs["batch"] = ad_integrated.obs.index.str.split("-").str[0]

    ad_integrated.obs["batch"] = ad_integrated.obs.index.str.split("-").str[0]

    ad_integrated.obs[f"labelTransfer_seurat_{refLabel}"] = pd.concat(
        [queryAdOrg.obs[f"labelTransfer_seurat_{refLabel}"], refAd.obs[refLabel]]
    ).to_list()

    dt_color = basic.getadataColor(refAd, refLabel)
    dt_color["unknown"] = "#111111"
    dt_color["None"] = "#D3D3D3"
    dt_color["nan"] = "#D3D3D3"
    ad_integrated = basic.setadataColor(ad_integrated, f"labelTransfer_seurat_{refLabel}", dt_color)
    sc.pl.umap(ad_integrated, color="batch")
    sc.pl.umap(ad_integrated, color=f"labelTransfer_seurat_{refLabel}", legend_loc="on data")
    if copy:
        return queryAdOrg
    if needLoc:
        return ad_integrated
