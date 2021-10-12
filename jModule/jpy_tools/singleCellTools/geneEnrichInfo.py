"""
DE analysis tools
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
from . import basic, diffxpy


def getUcellScore(
    ad: sc.AnnData,
    dt_deGene: Mapping[str, List[str]],
    layer: Optional[str],
    label,
    cutoff=0.2,
):
    """
    use ucell calculate average expression info

    Parameters
    ----------
    ad : sc.AnnData
    dt_deGene : Mapping[str, List[str]]
        key is label, value is marker genes
    layer : Optional[str]
        must NOT be scaled data
    label : 
        label for result.
    cutoff : float, optional
        by default 0.2
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py

    R = ro.r
    ucell = importr("UCell")
    rBase = importr("base")
    
    layer = None if layer == "X" else layer
    dtR_deGene = {x: R.c(*y) for x, y in dt_deGene.items()}
    dtR_deGene = R.list(**dtR_deGene)

    ss_forUcell = ad.layers[layer] if layer else ad.X
    ssR_forUcell = py2r(ss_forUcell.T)
    ssR_forUcell = R("`dimnames<-`")(
        ssR_forUcell,
        R.list(R.c(*ad.var.index), R.c(*ad.obs.index)),
    )

    r_scores = ucell.ScoreSignatures_UCell(ssR_forUcell, features=dtR_deGene)

    ad.obsm[f"ucell_score_{label}"] = r2py(rBase.as_data_frame(r_scores))
    ad.obs[f"ucell_celltype_{label}"] = ad.obsm[f"ucell_score_{label}"].pipe(
        lambda df: np.where(df.max(1) > cutoff, df.idxmax(1), "Unknown")
    )


def getOverlapInfo(
    adata: anndata.AnnData,
    key: str,
    markerDt: Mapping[str, List[str]],
    nTopGenes: int = 100,
) -> Mapping[str, List[str]]:
    """
    get overlap between marker genes with detected cluster-enriched genes

    Parameters
    ----------
    adata : anndata.AnnData
        after rank_genes_groups
    key : str
        key of rank_genes_groups
    markerDt : Mapping[str, List[str]]
        key is cell type, and value is corresponding marker genes
    nTopGenes : int, optional
        Number of cluster-enriched genes used, by default 100

    Returns
    -------
    Mapping[str, List[str]]
        key is cell type, and value is corresponding overlap genes
    """
    clusterEnrichedSt = set(
        sc.get.rank_genes_groups_df(adata, None, key=key)
        .groupby("group")
        .apply(lambda x: x.iloc[:nTopGenes]["names"])
    )
    overlapDt = {x: list(set(y) & clusterEnrichedSt) for x, y in markerDt.items()}
    return overlapDt


def detectMarkerGene(
    adata: anndata.AnnData,
    groupby: str,
    key_added: str,
    groups: Union[Literal["all"], Sequence[str], Callable[[str], bool]] = "all",
    use_raw: bool = False,
    layer: Optional[str] = None,
    method: Literal[
        "logreg", "t-test", "wilcoxon", "t-test_overestim_var"
    ] = "wilcoxon",
    pts: bool = True,
    min_in_group_fraction: float = 0.5,
    max_out_group_fraction: float = 0.25,
    min_fold_change: float = 0.585,
    rawDt: dict = {},
    filterDt: dict = {},
):
    """
    Rank and filter genes for characterizing groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Expects logarithmized data.
    groupby : str
        The key of the observations grouping to consider.
    key_added : str
        The key in adata.uns information is saved to.
    groups : Union[Literal[, optional
        Subset of groups, e.g. ['g1', 'g2', 'g3'], to which comparison shall be restricted, or 'all' (default), for all groups.
        Function also is supported. e.g. lambda x: x!='g1'
        Defaults to "all".
    use_raw : bool, optional
        by default False.
    layer: Optional, str.
        use which matrix as the used expression matrix. it takes precedence of use_raw.
    method : Literal[, optional
        't-test', 't-test_overestim_var' overestimates variance of each group,
        'wilcoxon' uses Wilcoxon rank-sum,
        'logreg' uses logistic regression.
        Defaults to "wilcoxon".
    pts : bool, optional
        Compute the fraction of cells expressing the genes. Defaults to True.
    min_in_group_fraction : float, optional
        by default 0.5
    max_out_group_fraction : float, optional
        by default 0.25
    min_fold_change : float, optional
        by default 2
    rawDt : dict, optional
        Other parameters for sc.tl.rank_genes_groups. Defaults to {}.
    filterDt : dict, optional
        ther parameters for sc.tl.filter_rank_genes_groups. Defaults to {}.
    """
    if groups != "all":
        if isinstance(groups, Callable):
            allCategoriesSq = adata.obs[groupby].astype("category").cat.categories
            groups = allCategoriesSq[allCategoriesSq.map(groups)]
            groups = list(groups)

        _adata = adata[adata.obs.query(f"{groupby} in @groups").index]
    else:
        _adata = adata

    rawDt = dict(
        groupby=groupby,
        groups=groups,
        use_raw=use_raw,
        layer=layer,
        method=method,
        pts=pts,
        key_added=key_added,
        **rawDt,
    )
    filterDt = dict(
        key=key_added,
        key_added=f"{key_added}_filtered",
        min_in_group_fraction=min_in_group_fraction,
        max_out_group_fraction=max_out_group_fraction,
        min_fold_change=min_fold_change,
        **filterDt,
    )

    sc.tl.rank_genes_groups(_adata, **rawDt)
    sc.tl.filter_rank_genes_groups(_adata, **filterDt)
    if groups != "all":
        adata.uns[key_added] = _adata.uns[key_added]
        adata.uns[f"{key_added}_filtered"] = _adata.uns[f"{key_added}_filtered"]


def calculateExpressionRatio(adata, clusterby):
    """
    逐个计算adata中每个基因在每个cluster中的表达比例

    adata:
        需要含有raw
    clusterby:
        adata.obs中的某个列名
    """
    transformadataRawToAd = lambda adata: anndata.AnnData(
        X=adata.raw.X, obs=adata.obs, var=adata.raw.var
    )
    rawAd = transformadataRawToAd(adata)
    expressionOrNotdf = (rawAd.to_df() > 0).astype(int)
    expressionOrNotdf[clusterby] = rawAd.obs[clusterby]
    expressionRatioDf = expressionOrNotdf.groupby(clusterby).agg(
        "sum"
    ) / expressionOrNotdf.groupby(clusterby).agg("count")
    return expressionRatioDf


def calculateGeneAverageEx(expressionMtxDf, geneDt, method="mean"):
    """
    根据geneDt对expressionMtxDf计算平均值或中位数

    expressionMtxDf:
        形如adata.to_df()

    geneDt:
        形如:{
    "type1": [
        "AT5G42235",
        "AT4G00540",
        ],
    "type2": [
        "AT1G55650",
        "AT5G45980",
        ],
    }
    method:
        'mean|median'

    """
    averageExLs = []
    for typeName, geneLs in geneDt.items():
        typeAvgExpress = (
            expressionMtxDf.reindex(geneLs, axis=1).mean(1)
            if method == "mean"
            else expressionMtxDf.reindex(geneLs, axis=1).median(1)
        )
        typeAvgExpress.name = typeName
        averageExLs.append(typeAvgExpress)
    averageExDf = pd.concat(averageExLs, axis=1)

    return averageExDf


def getEnrichedScore(adata, label, geneLs, threads=12, times=100):
    """
    获得ES值。ES值是通过对adata.obs中的label进行重排times次，然后计算原始label的zscore获得

    adata:
        必须有raw且为log-transformed

    label:
        adata.obs中的列名

    geneLs:
        需要计算的基因

    threads:
        使用核心数

    times:
        重排的次数
    """

    def __shuffleLabel(adata, label, i):
        """
        used for getEnrichedScore
        """
        shuffleAd = adata.copy()
        shuffleAd.obs[label] = adata.obs[label].sample(frac=1, random_state=i).values
        shuffleClusterDf = (
            mergeadataExpress(shuffleAd, label).to_df().reset_index().assign(label=i)
        )

        return shuffleClusterDf

    geneLs = geneLs[:]
    geneLs[0:0] = [label]
    adata = adata.copy()

    allShuffleClusterExpressLs = []
    with Mtp(threads) as mtp:
        for time in range(1, times + 1):
            allShuffleClusterExpressLs.append(
                mtp.submit(__shuffleLabel, adata, label, time)
            )

    allShuffleClusterExpressLs = [x.result() for x in allShuffleClusterExpressLs]
    originalClusterDf = (
        mergeadataExpress(adata, label).to_df().reset_index().assign(label=0)
    )
    allShuffleClusterExpressLs.append(originalClusterDf)
    allShuffleClusterExpressDf = (
        pd.concat(allShuffleClusterExpressLs).set_index("label").reindex(geneLs, axis=1)
    )
    logger.info(f"start calculate z score")
    allShuffleClusterZscoreDf = (
        allShuffleClusterExpressDf.groupby(label)
        .apply(lambda x: x.set_index(label, append=True).apply(zscore))
        .reset_index(level=0, drop=True)
    )
    clusterZscoreDf = (
        allShuffleClusterZscoreDf.query(f"label == 0")
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    return clusterZscoreDf


def calculateEnrichScoreByCellex(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    clusterName: str = "leiden",
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    calculateEnrichScoreByCellex

    Parameters
    ----------
    adata : anndata.AnnData
    layer : Optional[str], optional
        Must be int, by default None
    clusterName : str, optional
        by default 'leiden'
    copy : bool, optional
        by default False

    Returns
    -------
    anndata if copy else None
    """
    import cellex

    if layer == "X":
        layer = None

    adata = adata.copy() if copy else adata
    basic.testAllCountIsInt(adata, layer)

    df_mtx = adata.to_df(layer).T if layer else adata.to_df().T
    df_meta = adata.obs[[clusterName]].rename({clusterName: "cell_type"}, axis=1)
    eso = cellex.ESObject(data=df_mtx, annotation=df_meta)
    eso.compute()
    adata.varm[f"{clusterName}_cellexES"] = (
        eso.results["esmu"].reindex(adata.var.index).fillna(0)
    )
    if copy:
        return adata


def getEnrichedGeneByCellId(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    clusterName: str = "leiden",
    n_features: int = 200,
    nmcs: int = 50,
    copy: bool = False,
    returnR: bool = False,
    layerScaled: bool = False,
) -> Optional[pd.DataFrame]:
    """
    use CelliD get enriched gene

    Parameters
    ----------
    adata : anndata.AnnData
    layer : Optional[str], optional
        must be log-transformed data, by default None
    clusterName : str, optional
        by default 'leiden'
    n_features : int, optional
        by default 200
    copy : bool, optional
        by default False
    returnR : bool, optional
        This parameter takes precedence over copy. by default False.

    Returns
    -------
    Optional[pd.DataFrame]
        if copy, dataframe will be returned, else the anndata will be updated by following rules:
            obsm/varm will be updated by mca.
            uns will be updated by cellid_marker
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py

    rBase = importr("base")
    cellId = importr("CelliD")
    R = ro.r
    _ad = basic.getPartialLayersAdata(adata, [layer], [clusterName])
    if not layerScaled:
        sc.pp.scale(_ad, layer=layer, max_value=10)
    adataR = py2r(_ad)
    adataR = cellId.RunMCA(adataR, slot=layer, nmcs=nmcs)

    VectorR_marker = cellId.GetGroupGeneSet(
        adataR,
        group_by=clusterName,
        n_features=n_features,
        dims=py2r(np.arange(1, 1 + nmcs)),
    )
    if returnR:
        return VectorR_marker

    df_marker = r2py(rBase.data_frame(VectorR_marker, check_names=False))
    if copy:
        return df_marker
    else:
        adata.obsm["mca"] = r2py(
            rBase.as_data_frame(
                R.reducedDim(adataR, "MCA"),
            )
        )
        adata.varm["mca"] = r2py(
            rBase.as_data_frame(R.attr(R.reducedDim(adataR, "MCA"), "genesCoordinates"))
        ).reindex(adata.var.index, fill_value=0)
        adata.uns[f"{clusterName}_cellid_marker"] = df_marker


def getMarkerByFcCellexCellidDiffxpy(
    adata: anndata.AnnData,
    normalize_layer: str,
    raw_layer: str,
    groupby: str,
    groups: List[str] = None,
    forceAllRun: bool = False,
    dt_ByFcParams={},
    dt_DiffxpyParams={},
    dt_DiffxpyGetMarkerParams={},
    cutoff_cellex: float = 0.9,
    markerCounts_CellId: int = 50,
):
    """
    use three method to identify markers

    Parameters
    ----------
    adata : anndata.AnnData
    normalize_layer : str
        must be log-transformed
    raw_layer : str
        must be integer
    groupby : str
        column name in adata.obs
    groups : List[str], optional
        Only use these clusters, by default None
    forceAllRun : bool, optional
        by default False
    dt_ByFcParams : dict, optional
        params transfered to `geneEnrichInfo.detectMarkerGene`, by default {}
    dt_DiffxpyParams : dict, optional
        params transfered to `diffxpy.vsRest`, by default {}
    dt_DiffxpyGetMarkerParams : dict, optional
        params transfered to `diffxpy.getMarker`, by default {"detectedCounts":-2)}
    cutoff_cellex : float, optional
        by default 0.9
    markerCounts_CellId : int, optional
        by default 50
    """
    from itertools import product
    import scipy.sparse as ss
    import upsetplot
    import matplotlib.pyplot as plt

    adata.uns[f"marker_multiMethod_{groupby}"] = {}
    if not groups:
        groups = list(adata.obs[groupby].unique())

    ad_sub = adata[adata.obs.eval(f"{groupby} in @groups")].copy()
    # ad_sub.layers[f"{layer}_raw"] = (
    #     np.around(np.exp(ad_sub.layers[layer].A) - 1)
    #     if ss.issparse(ad_sub.layers[layer])
    #     else np.around(np.exp(ad_sub.layers[layer]) - 1)
    # )

    ## fc method
    if forceAllRun | (f"{groupby}_fcMarker" not in ad_sub.uns):
        geneEnrichInfo.detectMarkerGene(
            ad_sub,
            groupby,
            f"{groupby}_fcMarker",
            layer=normalize_layer,
            **dt_ByFcParams,
        )
        adata.uns[f"{groupby}_fcMarker"] = ad_sub.uns[f"{groupby}_fcMarker"]
        adata.uns[f"{groupby}_fcMarker_filtered"] = ad_sub.uns[
            f"{groupby}_fcMarker_filtered"
        ]
    dt_markerByFc = (
        sc.get.rank_genes_groups_df(ad_sub, None, key=f"{groupby}_fcMarker_filtered")
        .groupby("group")["names"]
        .agg(list)
        .to_dict()
    )
    adata.uns[f"marker_multiMethod_{groupby}"]["fcMarker"] = dt_markerByFc

    ## cellex method
    if forceAllRun | (f"{groupby}_cellexES" not in ad_sub.varm):
        geneEnrichInfo.calculateEnrichScoreByCellex(ad_sub, f"{raw_layer}", groupby)
        adata.varm[f"{groupby}_cellexES"] = ad_sub.varm[f"{groupby}_cellexES"]
    dt_marker_cellex = (
        ad_sub.varm[f"{groupby}_cellexES"]
        .apply(lambda x: list(x[x > cutoff_cellex].sort_values(ascending=False).index))
        .to_dict()
    )
    adata.uns[f"marker_multiMethod_{groupby}"]["cellexMarker"] = dt_marker_cellex

    ## cellid method
    if forceAllRun | (f"{groupby}_cellid_marker" not in adata.uns):
        geneEnrichInfo.getEnrichedGeneByCellId(
            ad_sub, normalize_layer, groupby, markerCounts_CellId
        )
        adata.uns[f"{groupby}_cellid_marker"] = ad_sub.uns[f"{groupby}_cellid_marker"]
    dt_markerCellId = {
        x: list(y)
        for x, y in ad_sub.uns[f"{groupby}_cellid_marker"].to_dict("series").items()
    }

    adata.uns[f"marker_multiMethod_{groupby}"]["cellidMarker"] = dt_markerCellId

    ## diffxpy method
    if forceAllRun | (f"{groupby}_diffxpy_marker" not in adata.uns):
        diffxpy.vsRest(
            ad_sub,
            raw_layer,
            groupby,
            keyAdded=f"{groupby}_diffxpy_marker",
            inputIsLog=False,
            **dt_DiffxpyParams,
        )
        adata.uns[f"{groupby}_diffxpy_marker"] = ad_sub.uns[f"{groupby}_diffxpy_marker"]

    df_diffxpyMarker = diffxpy.getMarker(
        adata, key=f"{groupby}_diffxpy_marker", **dt_DiffxpyGetMarkerParams
    )
    adata.uns[f"marker_multiMethod_{groupby}"]["diffxpyMarker"] = (
        df_diffxpyMarker.groupby("clusterName")["gene"].agg(list).to_dict()
    )

    # concat
    for markerCat, cluster in product(
        ["fcMarker", "cellexMarker", "cellidMarker", "diffxpyMarker"], groups
    ):
        if cluster not in adata.uns[f"marker_multiMethod_{groupby}"][markerCat]:
            adata.uns[f"marker_multiMethod_{groupby}"][markerCat][cluster] = []

    ls_allClusterMarker = []
    for cluster in groups:
        ls_clusterMarker = [
            y
            for x in ["fcMarker", "cellexMarker", "cellidMarker", "diffxpyMarker"]
            for y in adata.uns[f"marker_multiMethod_{groupby}"][x][cluster]
        ]

        ls_clusterMarker = list(set(ls_clusterMarker))
        ls_clusterName = [cluster] * len(ls_clusterMarker)
        df_clusterMarker = pd.DataFrame(
            [ls_clusterName, ls_clusterMarker], index=["cluster", "marker"]
        ).T
        df_clusterMarker = df_clusterMarker.pipe(
            lambda df: df.assign(
                fcMarker=df["marker"].isin(
                    adata.uns[f"marker_multiMethod_{groupby}"]["fcMarker"][cluster]
                ),
                cellexMarker=df["marker"].isin(
                    adata.uns[f"marker_multiMethod_{groupby}"]["cellexMarker"][cluster]
                ),
                cellidMarker=df["marker"].isin(
                    adata.uns[f"marker_multiMethod_{groupby}"]["cellidMarker"][cluster]
                ),
                diffxpyMarker=df["marker"].isin(
                    adata.uns[f"marker_multiMethod_{groupby}"]["diffxpyMarker"][cluster]
                ),
            ).assign(
                detectedMethodCounts=lambda df: df[
                    ["fcMarker", "cellexMarker", "cellidMarker", "diffxpyMarker"]
                ].sum(1)
            )
        )
        ls_allClusterMarker.append(df_clusterMarker)
    df_allClusterMarker = pd.concat(ls_allClusterMarker)
    adata.uns[f"marker_multiMethod_{groupby}"]["unionInfo"] = df_allClusterMarker

    axs = upsetplot.plot(
        df_allClusterMarker[
            ["fcMarker", "cellexMarker", "cellidMarker", "diffxpyMarker"]
        ].value_counts(),
        sort_by="cardinality",
    )
    plt.sca(axs["intersections"])
    plt.yscale("log")