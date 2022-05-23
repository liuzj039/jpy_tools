"""
DE analysis tools
"""
from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from io import StringIO
from concurrent.futures import ProcessPoolExecutor as Mtp
from concurrent.futures import ThreadPoolExecutor as MtT
from joblib import Parallel, delayed
import sh
import h5py
from tqdm import tqdm
import scvi
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
from ..rTools import rcontext


def getBgGene(
    ad,
    ls_gene,
    layer="normalize_log",
    bins=50,
    seed=0,
    usePreBin: str = None,
    multi=1,
    replacement=True,
):
    "replacement: if True, ls_gene will be included in the result"
    if not usePreBin:
        ad.var["means_ForPickMock"] = ad.to_df(layer).mean()
        ad.var["bins_ForPickMock"] = pd.qcut(
            ad.var["means_ForPickMock"], bins, duplicates="drop"
        )
    else:
        ad.var["bins_ForPickMock"] = ad.var[usePreBin]
    dt_binGeneCounts = ad.var.loc[ls_gene]["bins_ForPickMock"].value_counts().to_dict()
    if replacement:
        ls_gene = []
    ls_randomGene = (
        ad.var.query("index not in @ls_gene")
        .groupby("bins_ForPickMock", group_keys=False)
        .apply(
            lambda df: df.sample(
                n=dt_binGeneCounts[df["bins_ForPickMock"].iloc[0]] * multi,
                random_state=seed,
            )
        )
        .index.to_list()
    )
    return ls_randomGene


def getGeneModuleEnrichScore(
    ad,
    layer,
    ls_gene,
    times=100,
    groupby=None,
    targetOnly=True,
    disableBar=False,
    multi=False,
    **dt_paramsGetBgGene,
):
    """
    get gene module enrich score

    Parameters
    ----------
    ad : sc.AnnData
    layer :
        normalize_log
    ls_gene :
        gene list:
            ['a', 'b']
    times : int, optional
        by default 100
    groupby : _type_, optional
        by default None
    targetOnly : bool, optional
        remove shuffled gene results or not
    """

    ls_adGene = ad.var.index.to_list()
    st_notInGene = set(ls_gene) - set(ls_adGene)
    if len(st_notInGene):
        logger.warning(f"{ls_gene} not found in adata")
    ls_gene = [x for x in ls_gene if x in ls_adGene]
    if not multi:
        getBgGene(ad, ls_gene, layer=layer, **dt_paramsGetBgGene)
    ad.layers[f"{layer}_scaled"] = ad.layers[layer].copy()
    sc.pp.scale(ad, layer=f"{layer}_scaled")
    ls_bgGene = [
        getBgGene(
            ad,
            ls_gene,
            layer=layer,
            usePreBin="bins_ForPickMock",
            seed=x,
            **dt_paramsGetBgGene,
        )
        for x in tqdm(range(times), "get shuffled genes", disable=disableBar)
    ]
    if not groupby:
        ad.obs["temp_forES"] = "all"
        groupby = "temp_forES"
    dt_scaledMeanExp = {}
    for name, _ad in basic.splitAdata(ad, groupby, needName=True, copy=False):
        df_scaledExp = _ad.to_df(f"{layer}_scaled")
        dt_groupMeanExp = {"target": df_scaledExp[ls_gene].mean().mean()}
        for i, ls_bgGeneSingle in enumerate(ls_bgGene):
            dt_groupMeanExp[f"shuffle_{i}"] = (
                df_scaledExp[ls_bgGeneSingle].mean().mean()
            )
        dt_scaledMeanExp[name] = dt_groupMeanExp
    df_scaledMeanExp = pd.DataFrame(dt_scaledMeanExp).apply(zscore)
    if targetOnly:
        return df_scaledMeanExp.loc["target"].to_dict()
    else:
        return df_scaledMeanExp


def getGeneModuleEnrichScore_multiList(
    ad, layer, dt_gene, times=100, groupby=None, bins=50, **dt_paramsGetBgGene
):
    """
    get gene module enrich score

    Parameters
    ----------
    ad : sc.AnnData
    layer :
        normalize_log
    ls_gene :
        gene list:
            ['a', 'b']
    times : int, optional
        by default 100
    groupby : _type_, optional
        by default None
    targetOnly : bool, optional
        remove shuffled gene results or not
    """
    ad.var["means_ForPickMock"] = ad.to_df(layer).mean()
    ad.var["bins_ForPickMock"] = pd.qcut(
        ad.var["means_ForPickMock"], bins, duplicates="drop"
    )

    dt_result = {}
    for name, ls_gene in tqdm(
        dt_gene.items(), "get gene module enrich score", len(dt_gene)
    ):
        dt_result[name] = getGeneModuleEnrichScore(
            ad,
            layer,
            ls_gene,
            times,
            groupby,
            disableBar=True,
            multi=True,
            **dt_paramsGetBgGene,
        )
    return pd.DataFrame(dt_result)


def getUcellScore(
    ad: sc.AnnData,
    dt_deGene: Mapping[str, List[str]],
    layer: Optional[str],
    label,
    cutoff=0.2,
    batch=None,
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
    if batch:
        dfLs_ucellResults = []
        for _, _ad in basic.splitAdata(ad, batch, needName=True, copy=False):
            ss_forUcell = _ad.layers[layer] if layer else _ad.X
            ssR_forUcell = py2r(ss_forUcell.T)
            ssR_forUcell = R("`dimnames<-`")(
                ssR_forUcell,
                R.list(R.c(*_ad.var.index), R.c(*_ad.obs.index)),
            )

            r_scores = ucell.ScoreSignatures_UCell(ssR_forUcell, features=dtR_deGene)
            dfLs_ucellResults.append(r2py(rBase.as_data_frame(r_scores)))
        ad.obsm[f"ucell_score_{label}"] = pd.concat(dfLs_ucellResults).reindex(
            ad.obs.index
        )

    else:
        ss_forUcell = ad.layers[layer] if layer else ad.X
        ssR_forUcell = py2r(ss_forUcell.T)
        ssR_forUcell = R("`dimnames<-`")(
            ssR_forUcell,
            R.list(R.c(*ad.var.index), R.c(*ad.obs.index)),
        )

        r_scores = ucell.ScoreSignatures_UCell(ssR_forUcell, features=dtR_deGene)
        ad.obsm[f"ucell_score_{label}"] = r2py(rBase.as_data_frame(r_scores))

    if len(dt_deGene) > 1:
        ad.obs[f"ucell_celltype_{label}"] = ad.obsm[f"ucell_score_{label}"].pipe(
            lambda df: np.where(df.max(1) > cutoff, df.idxmax(1), "Unknown")
        )


def getGeneScore(
    ad: sc.AnnData,
    dt_Gene: Dict[str, List[str]],
    layer: Optional[str],
    label: str,
    func: Callable,
):
    """
    use gene score calculate average expression info

    Parameters
    ----------
    ad : sc.AnnData
    layer : Optional[str]
    dt_Gene : Dict[str, List[str]]
        key is label, value is marker genes
    label :
        label for result.
    func : Callable
        calculate function
    """
    df_results = pd.DataFrame(index=ad.obs.index, columns=dt_Gene.keys())
    for name, ls_gene in dt_Gene.items():
        _sr = ad[:, ls_gene].to_df(layer).apply(func, axis=1)
        df_results[name] = _sr
    ad.obsm[label] = df_results


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


def calculateEnrichScoreByCellex(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    clusterName: str = "leiden",
    batchKey: Optional[str] = None,
    copy: bool = False,
    dt_kwargsForCellex: dict = {},
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

    def _singleBatch(adata, layer, clusterName, dt_kwargsForCellex):
        df_mtx = adata.to_df(layer).T if layer else adata.to_df().T
        df_meta = adata.obs[[clusterName]].rename({clusterName: "cell_type"}, axis=1)
        eso = cellex.ESObject(data=df_mtx, annotation=df_meta, **dt_kwargsForCellex)
        eso.compute()
        mtx_enrichScore = eso.results["esmu"].reindex(adata.var.index).fillna(0)
        adata.varm[f"{clusterName}_cellexES"] = mtx_enrichScore

        mtx_geneExpRatio = (
            adata.to_df(layer)
            .groupby(adata.obs[clusterName])
            .apply(lambda df: (df > 0).mean())
            .T
        )
        df_geneExpRatio = (
            mtx_geneExpRatio.rename_axis(index="gene", columns=clusterName)
            .melt(ignore_index=False, value_name="expressed_ratio")
            .reset_index()
        )

        ls_cluster = adata.obs[clusterName].unique()
        mtx_binary = (adata.to_df(layer) > 0).astype(int)

        dt_geneExpRatioOtherCluster = {}
        for cluster in ls_cluster:
            ls_index = adata.obs[clusterName].pipe(lambda sr: sr[sr != cluster]).index
            dt_geneExpRatioOtherCluster[cluster] = mtx_binary.loc[ls_index].mean(0)

        mtx_geneExpRatioOtherCluster = pd.DataFrame.from_dict(
            dt_geneExpRatioOtherCluster
        )
        df_geneExpRatioOtherCluster = (
            mtx_geneExpRatioOtherCluster.rename_axis(index="gene", columns=clusterName)
            .melt(ignore_index=False, value_name="expressed_ratio_others")
            .reset_index()
        )

        df_result = (
            mtx_enrichScore.rename_axis(index="gene", columns=clusterName)
            .melt(ignore_index=False, value_name="enrichScore")
            .reset_index()
            .merge(
                df_geneExpRatio,
                left_on=["gene", clusterName],
                right_on=["gene", clusterName],
            )
            .merge(
                df_geneExpRatioOtherCluster,
                left_on=["gene", clusterName],
                right_on=["gene", clusterName],
            )
        )
        adata.uns[f"{clusterName}_cellexES"] = df_result

    adata = adata.copy() if copy else adata
    basic.testAllCountIsInt(adata, layer)

    if layer == "X":
        layer = None
    if batchKey is None:
        _singleBatch(adata, layer, clusterName, dt_kwargsForCellex)
    else:
        ls_batchAd = basic.splitAdata(adata, batchKey, needName=True)
        adata.uns[f"{clusterName}_cellexES"] = {}
        for sample, ad_batch in ls_batchAd:
            _singleBatch(ad_batch, layer, clusterName, dt_kwargsForCellex)
            adata.uns[f"{clusterName}_cellexES"][sample] = ad_batch.uns[
                f"{clusterName}_cellexES"
            ]

    if copy:
        return adata


def getMarkerFromCellexResults(
    ad,
    clusterName,
    minCounts,
    filterExpr="enrichScore > 0.9 & expressed_ratio > 0.1 & expressed_ratio_others < 0.1",
):
    from functools import reduce

    _ls = []
    lsDf_Concat = []
    for sample, df_geneInfo in ad.uns[f"{clusterName}_cellexES"].items():
        _ls.append(df_geneInfo.query(filterExpr)[["gene", clusterName]])
        lsDf_Concat.append(
            df_geneInfo.rename(
                columns=lambda x: f"{sample}_{x}"
                if x in ["enrichScore", "expressed_ratio", "expressed_ratio_others"]
                else x
            )
        )
    sr_geneCounts = (
        pd.concat(_ls).value_counts().rename("counts").pipe(pd.DataFrame).reset_index()
    )
    df_results = reduce(
        lambda x, y: pd.merge(
            x,
            y,
            left_on=("gene", clusterName),
            right_on=("gene", clusterName),
            how="outer",
        ),
        [sr_geneCounts, *lsDf_Concat],
    ).query("counts >= @minCounts")
    df_results.insert(
        3,
        "mean_expressed_ratio_others",
        df_results.filter(like="expressed_ratio_others").mean(1),
    )
    df_results.insert(
        3, "mean_expressed_ratio", df_results.filter(like="expressed_ratio").mean(1)
    )
    df_results.insert(
        3, "mean_enrichScore", df_results.filter(like="enrichScore").mean(1)
    )
    df_results = df_results.sort_values(
        [clusterName, "mean_enrichScore"], ascending=[True, False]
    )
    return df_results


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
        detectMarkerGene(
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
        calculateEnrichScoreByCellex(ad_sub, f"{raw_layer}", groupby)
        adata.varm[f"{groupby}_cellexES"] = ad_sub.varm[f"{groupby}_cellexES"]
    dt_marker_cellex = (
        ad_sub.varm[f"{groupby}_cellexES"]
        .apply(lambda x: list(x[x > cutoff_cellex].sort_values(ascending=False).index))
        .to_dict()
    )
    adata.uns[f"marker_multiMethod_{groupby}"]["cellexMarker"] = dt_marker_cellex

    ## cellid method
    if forceAllRun | (f"{groupby}_cellid_marker" not in adata.uns):
        getEnrichedGeneByCellId(ad_sub, normalize_layer, groupby, markerCounts_CellId)
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


def getDEGByScvi(
    ad: sc.AnnData,
    path_model: Optional[Union[str, scvi.model.SCVI]],
    layer: Optional[str],
    groupby: str,
    batchKey: Optional[str] = None,
    correctBatch: bool = True,
    minCells: int = 10,
    threads: int = 36,
    keyAdded: Optional[str] = "marker_byScvi",
    only_train_model: bool = False,
    batch_size=512,
) -> Tuple[scvi.model.SCVI, pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    ad : sc.AnnData
    path_model : Optional[str]
        stored path of trained scvi.model.SCVI or scvi.model.SCVI
    layer : Optional[str]
        must be raw
    groupby : str
        cluster key
    batchKey : Optional[str]
        batch key
    correctBatch : bool, optional
        by default True
    minCells : int, optional
        by default 10
    threads : int, optional
        by default 36
    keyAdded : Optional[str], optional
        if None, adata will not be updated, by default "marker_byScvi"

    Returns
    -------
    Tuple[scvi.model.SCVI, pd.DataFrame]
    """
    scvi.settings.seed = 39
    scvi.settings.num_threads = threads

    layer = None if layer == "X" else layer
    ad.X = ad.layers[layer].copy()
    ad_forDE = sc.pp.filter_genes(ad, min_cells=minCells, copy=True)
    scvi.model.SCVI.setup_anndata(
        ad_forDE,
        layer=None,
        batch_key=batchKey,
    )

    if not path_model:
        scvi_model = scvi.model.SCVI(ad_forDE)
        scvi_model.train(early_stopping=True, batch_size=batch_size)
        scvi_model.history["elbo_train"].plot()
        plt.yscale("log")
        plt.show()

    else:
        if isinstance(path_model, str):
            scvi_model = scvi.model.SCVI.load(path_model, ad_forDE)
        elif isinstance(path_model, scvi.SCVI.model):
            scvi_model = path_model
        else:
            assert False, f"Unknown data type: {type(path_model)}"

    if only_train_model:
        return scvi_model, None

    df_deInfo = scvi_model.differential_expression(
        ad_forDE, groupby=groupby, batch_correction=correctBatch
    )
    if not keyAdded:
        pass
    else:
        keyAdded = keyAdded + "_" + groupby
        ad.uns[keyAdded] = df_deInfo

    return scvi_model, df_deInfo


def getDEGFromScviResult(
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


def getCosgResult(ad, key="cosg") -> pd.DataFrame:
    """
    maybe the SOTA algorithm for one batch datasets
    """
    df = (
        pd.concat(
            [
                pd.DataFrame(ad.uns[key]["names"]).stack().rename("name"),
                pd.DataFrame(ad.uns[key]["scores"]).stack().rename("score"),
            ],
            axis=1,
        )
        .reset_index(level=0, drop=True)
        .rename_axis("cluster")
        .reset_index()
        .sort_values(["cluster", "score"], ascending=[True, False])
    )
    return df


@rcontext
def getAUCellScore(
    ad,
    dt_genes,
    layer,
    threads=1,
    aucMaxRank=500,
    label="AUCell",
    thresholdsHistCol=3,
    show=False,
    forceDense=False,
    rEnv=None,
):
    """

    Parameters
    ----------
    ad :
    dt_genes :
    layer :
        rank based
    label : str, optional
        by default 'AUCell'
    rEnv: ro.Environment
        by default None
    """
    from math import ceil
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import scipy.sparse as ss
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rSet, rGet, ad2so, so2ad

    rBase = importr("base")
    rUtils = importr("utils")
    dplyr = importr("dplyr")
    reticulate = importr("reticulate")
    R = ro.r

    aucell = importr("AUCell")

    def getThreshold(objR_name, geneCate):
        # print(f"as.data.frame({objR_name}${geneCate}$aucThr$thresholds)")
        df = r2py(R(f"as.data.frame({objR_name}$`{geneCate}`$aucThr$thresholds)"))
        df = df.assign(geneCate=geneCate)
        return df

    thresholdsHistRow = ceil(len(dt_genes) / thresholdsHistCol)
    ls_varName = ad.var.index.to_list()
    ls_obsName = ad.obs.index.to_list()
    mtx = ad.layers[layer].T
    if ss.issparse(mtx) & forceDense:
        mtx = mtx.A
    mtxR = py2r(mtx)
    del mtx

    fc_list2R = lambda x: R.unlist(R.list(x))
    lsR_obsName = fc_list2R(ls_obsName)
    lsR_varName = fc_list2R(ls_varName)
    dtR_genes = R.list(**{x: fc_list2R(y) for x, y in dt_genes.items()})

    rEnv["mtxR"] = mtxR
    rEnv["lsR_obsName"] = lsR_obsName
    rEnv["lsR_varName"] = lsR_varName
    rEnv["dtR_genes"] = dtR_genes
    rEnv["threads"] = threads
    rEnv["aucMaxRank"] = aucMaxRank
    rEnv["thresholdsHistCol"] = thresholdsHistCol
    rEnv["thresholdsHistRow"] = thresholdsHistRow
    with r_inline_plot():
        R(
            """
        rownames(mtxR) <- lsR_varName
        colnames(mtxR) <- lsR_obsName
        cells_rankings <- AUCell_buildRankings(mtxR, nCores=threads, plotStats=TRUE)
        cells_AUC <- AUCell_calcAUC(dtR_genes, cells_rankings, aucMaxRank=aucMaxRank)
        """
        )
    if show:
        with r_inline_plot(width=512):
            R(
                """
            par(mfrow=c(thresholdsHistRow, thresholdsHistCol)) 
            cells_assignment <- AUCell_exploreThresholds(cells_AUC, plotHist=T) 
            """
            )
    else:
        R(
            """
        cells_assignment <- AUCell_exploreThresholds(cells_AUC, plotHist=F) 
        """
        )
    df_auc = r2py(R("as.data.frame(cells_AUC@assays@data$AUC)")).T
    df_aucThreshold = pd.concat(
        [getThreshold("cells_assignment", x) for x in dt_genes.keys()]
    )
    ad.obsm[label] = df_auc.copy()
    ad.uns[label] = df_aucThreshold.copy()


def _getMetaCells(
    ad, ls_obs, layer="raw", skipSmallGroup=True, target_metacell_size=5e4, **kwargs
):
    """
    get meta-cell from adata
    """
    import metacells as mc

    if isinstance(ls_obs, str):
        ls_obs = [ls_obs]
    dtAd_meta = {}
    ls_changeObs = []
    for ls_label, df in ad.obs.groupby(ls_obs):
        if isinstance(ls_label, str):
            ls_label = [ls_label]

        ad_sub = ad[df.index].copy()
        ad_sub.X = ad_sub.layers[layer].copy()
        logger.info(f"{ls_label} info:: {len(ad_sub)} cells, {ad_sub.X.sum()} UMIs")

        if ad_sub.X.sum() / 2 < target_metacell_size:
            ls_changeObs.append(ls_label)
            if skipSmallGroup:
                logger.warning(f"{ls_label} is too small, skip")
                continue
            else:
                _target_metacell_size = ad_sub.X.sum() // 2
                logger.warning(
                    f"{ls_label} is too small, set target_metacell_size to {_target_metacell_size}"
                )
                mc.pl.divide_and_conquer_pipeline(
                    ad_sub, target_metacell_size=_target_metacell_size, **kwargs
                )
        else:
            mc.pl.divide_and_conquer_pipeline(
                ad_sub, target_metacell_size=target_metacell_size, **kwargs
            )

        ad_subMeta = mc.pl.collect_metacells(ad_sub, name="metacells")
        dt_metaLinkWithOrg = (
            ad_sub.obs["metacell"]
            .groupby(ad_sub.obs["metacell"])
            .apply(lambda sr: sr.index.to_list())
            .to_dict()
        )
        ad_subMeta.obs["metacell_link"] = ad_subMeta.obs.index.map(
            lambda x: dt_metaLinkWithOrg[int(x)]
        )

        # ad_sub.obs.index = ad_sub.obs.index.astype(str) + '-' + '-'.join(ls_label)
        for obsName, label in zip(ls_obs, ls_label):
            ad_subMeta.obs[obsName] = label
        ad_subMeta.layers[layer] = ad_subMeta.X.copy()
        dtAd_meta["-".join(ls_label)] = ad_subMeta
    ad_meta = sc.concat(dtAd_meta, index_unique="-")
    for obsName in ls_obs:
        if isinstance(ad.obs[obsName].dtype, pd.CategoricalDtype):
            ad_meta.obs[obsName] = (
                ad_meta.obs[obsName]
                .astype("category")
                .cat.reorder_categories(ad.obs[obsName].cat.categories)
            )
    ad_meta.obs["metacell_link"] = ad_meta.obs["metacell_link"].str.join(",")
    print(ad_meta.obs[ls_obs].value_counts())
    print(f"These {ls_changeObs} groups are skipped or changed due to small size")
    sns.histplot(ad_meta.to_df(layer).sum(1))
    plt.xlabel("metacell UMI counts")
    plt.show()

    sns.histplot(ad_meta.obs["metacell_link"].str.split(",").map(len))
    plt.xlabel("metacell cell counts")
    plt.show()
    return ad_meta


def scWGCNA(
    ad: sc.AnnData,
    layer: str,
    ls_obs: List,
    skipSmallGroup: bool,
    target_metacell_size: int,
    dir_result: str,
    jobid: str,
    ls_hvgGene: List[str],
    minModuleSize: int = 50,
    deepSplit: float = 4,
    mergeCutHeight: float = 0.2,
    threads: int = 1,
    softPower: Optional[Union[int, Literal["auto"]]] = None,
    renv=None,
    dt_getMetaCellsKwargs: dict = {},
    fc_autoPickSoftPower=lambda x: x["SFT.R.sq"] >= 0.8,
    calcAucell=True,
) -> sc.AnnData:
    """`scWGCNA` is a function that takes in a single cell RNA-seq dataset, and returns a single cell
    RNA-seq dataset with metacells

    Parameters
    ----------
    ad : sc.AnnData
        sc.AnnData
    layer : str
        str， 'raw'
    ls_obs : List
    skipSmallGroup : bool
        if True, skip the small groups in metacell construction step
    target_metacell_size : int
        the target size of metacells.
    dir_result : str
        str, store `blockwiseConsensusModules` results
    jobid : str
        str
    ls_hvgGene : List[str]
        List[str]
    minModuleSize : int, optional
        minimum number of genes in a module
    deepSplit : float, optional
        the number of times to split the dendrogram.
    mergeCutHeight : float
        the height of the merge cut.
    threads : int, optional
        number of threads to use
    softPower : Optional[Union[int, Literal['auto']]]
        The soft power parameter for WGCNA. If None, it will be manually picked; if 'auto', it will be automatically picked.
    renv
        R environment to use. If None, will use the default R environment.
    dt_getMetaCellsKwargs : dict
        dict = {}
    fc_autoPickSoftPower
        function to pick the soft power. columns: ['SFT.R.sq', 'mean.k.']. By default, it will pick the soft power if the 'SFT.R.sq' is greater than 0.8 (lambda x:x['SFT.R.sq'] >= 0.8).

    ---
    sc.AnnData
        with WGCNA results
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from . import plotting
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rGet, rSet, ad2so, so2ad
    import os
    import warnings

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    rBase = importr("base")
    rUtils = importr("utils")
    tidyverse = importr("tidyverse")
    WGCNA = importr("WGCNA")
    seurat = importr("Seurat")
    gplots = importr("gplots")
    R = ro.r
    R(f"disableWGCNAThreads()")

    if renv is None:
        renv = ro.Environment()

    # rlc = ro.local_context(renv)
    # rlc.__enter__()
    with ro.local_context(renv) as rlc:
        rlc["minModuleSize"] = minModuleSize
        rlc["deepSplit"] = deepSplit
        rlc["mergeCutHeight"] = mergeCutHeight
        rlc["maxBlockSize"] = len(ls_hvgGene)
        rlc["jobid"] = jobid
        rlc["dir_result"] = dir_result
        ls_removeGene = [x for x in ad.var.index.to_list() if x not in ls_hvgGene]
        ad_meta = _getMetaCells(
            ad,
            ls_obs,
            layer=layer,
            skipSmallGroup=skipSmallGroup,
            target_metacell_size=target_metacell_size,
            forbidden_gene_names=ls_removeGene,
            **dt_getMetaCellsKwargs,
        )[:, ls_hvgGene]
        # ad_meta = ad[:, ls_hvgGene].copy()
        basic.initLayer(ad_meta, layer=layer, total=1e6)
        datExpr = py2r(ad_meta.to_df("normalize_log"))
        rlc["datExpr"] = datExpr
        # import pdb; pdb.set_trace()
        R(
            """
        datExpr <- datExpr[,goodGenes(datExpr)]
        lsR_useGene = colnames(datExpr)
        """
        )

        if (softPower is None) | (softPower == "auto"):
            with r_inline_plot(width=768):
                R(
                    """
                powers = c(seq(1,10,by=1), seq(12,20, by=2))
                powerTable = list(
                data = pickSoftThreshold(
                    datExpr,
                    powerVector=powers,
                    verbose = 100,
                    networkType="signed",
                    corFnc="bicor"
                )[[2]]
                )

                # Plot the results:

                colors = c("blue", "red","black")
                # Will plot these columns of the returned scale free analysis tables
                plotCols = c(2,5,6,7)
                colNames = c("Scale Free Topology Model Fit", "Mean connectivity", "Median connectivity",
                "Max connectivity");

                # Get the minima and maxima of the plotted points
                ylim = matrix(NA, nrow = 2, ncol = 4);
                for (col in 1:length(plotCols)){
                ylim[1, col] = min(ylim[1, col], powerTable$data[, plotCols[col]], na.rm = TRUE);
                ylim[2, col] = max(ylim[2, col], powerTable$data[, plotCols[col]], na.rm = TRUE);
                }
                """
                )
                # Plot the quantities in the chosen columns vs. the soft thresholding power
                R(
                    """
                par(mfcol = c(2,2));
                par(mar = c(4.2, 4.2 , 2.2, 0.5))
                cex1 = 0.7;

                for (col in 1:length(plotCols)){
                    plot(powerTable$data[,1], -sign(powerTable$data[,3])*powerTable$data[,2],
                    xlab="Soft Threshold (power)",ylab=colNames[col],type="n", ylim = ylim[, col],
                    main = colNames[col]);
                    addGrid();

                    if (col==1){
                        text(powerTable$data[,1], -sign(powerTable$data[,3])*powerTable$data[,2],
                        labels=powers,cex=cex1,col=colors[1]);
                    } else
                    text(powerTable$data[,1], powerTable$data[,plotCols[col]],
                    labels=powers,cex=cex1,col=colors[1]);
                    if (col==1){
                        legend("bottomright", legend = 'Metacells', col = colors, pch = 20) ;
                    } else
                    legend("topright", legend = 'Metacells', col = colors, pch = 20) ;
                }
                """
                )
            if softPower == "auto":
                df_powerTable = r2py(R("powerTable$data"))
                df_powerTable["Power"] = df_powerTable["Power"].astype(int)
                df_powerTable = df_powerTable.loc[fc_autoPickSoftPower]
                softPower = df_powerTable["Power"].min()
                logger.info(f"Soft Power: {softPower}")
            else:
                softPower = int(input("Soft Power"))
        rlc["softPower"] = softPower

        if threads > 1:
            R(f"enableWGCNAThreads({threads})")
        else:
            R(f"disableWGCNAThreads()")

        R(
            """
        nSets = 1
        setLabels <- 'ODC'
        shortLabels <- setLabels

        multiExpr <- list()
        multiExpr[['ODC']] <- list(data=datExpr)
        checkSets(multiExpr) 

        net <- blockwiseConsensusModules(multiExpr, blocks = NULL,
                                                maxBlockSize = maxBlockSize, ## This should be set to a smaller size if the user has limited RAM
                                                randomSeed = 39,
                                                corType = "pearson",
                                                power = softPower,
                                                consensusQuantile = 0.3,
                                                networkType = "signed",
                                                TOMType = "unsigned",
                                                TOMDenom = "min",
                                                scaleTOMs = TRUE, scaleQuantile = 0.8,
                                                sampleForScaling = TRUE, sampleForScalingFactor = 1000,
                                                useDiskCache = TRUE, chunkSize = NULL,
                                                deepSplit = deepSplit,
                                                pamStage=FALSE,
                                                detectCutHeight = 0.995, minModuleSize = minModuleSize,
                                                mergeCutHeight = mergeCutHeight,
                                                saveConsensusTOMs = TRUE,
                                                consensusTOMFilePattern = paste0(dir_result, "/", jobid, "_TOM_block.%b.rda"))

        consMEs = net$multiMEs;
        moduleLabels = net$colors;

        # Convert the numeric labels to color labels
        moduleColors = as.character(moduleLabels)
        consTree = net$dendrograms[[1]];

        # module eigengenes
        MEs=moduleEigengenes(multiExpr[[1]]$data, colors = moduleColors, nPC=1)$eigengenes
        MEs=orderMEs(MEs)
        meInfo<-data.frame(rownames(datExpr), MEs)
        colnames(meInfo)[1]= "SampleID"

        # intramodular connectivity
        KMEs<-signedKME(datExpr, MEs,outputColumnName = "kME",corFnc = "bicor")

        # compile into a module metadata table
        geneInfo=as.data.frame(cbind(colnames(datExpr),moduleColors, KMEs))

        # how many modules did we get?
        nmodules <- length(unique(moduleColors))

        # merged gene symbol column
        colnames(geneInfo)[1]= "GeneSymbol"
        colnames(geneInfo)[2]= "Initially.Assigned.Module.Color"
        PCvalues=MEs
        """
        )

        with r_inline_plot(width=768):
            R(
                """
            plotDendroAndColors(consTree, moduleColors, "Module colors", dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, guideHang = 0.05,
                                main = paste0("ODC lineage gene dendrogram and module colors"))"""
            )
        with r_inline_plot(width=768):
            R(
                """
            plotEigengeneNetworks(PCvalues, "Eigengene adjacency heatmap", 
                                marDendro = c(3,3,2,4),
                                marHeatmap = c(3,4,2,2), plotDendrograms = T, 
                                xLabelsAngle = 90)
            """
            )
        R(
            """
        load(paste0(dir_result, "/", jobid, "_TOM_block.1.rda"), verbose=T)

        probes = colnames(datExpr)
        TOM <- as.matrix(consTomDS)
        dimnames(TOM) <- list(probes, probes)

        # cyt = exportNetworkToCytoscape(TOM,
        #             weighted = TRUE, threshold = 0.1,
        #             nodeNames = probes, nodeAttr = moduleColors)
        """
        )

        ad_meta = ad_meta[:, list(rlc["lsR_useGene"])]
        ad_meta.obsm["eigengene"] = r2py(R("meInfo"))
        ad_meta.obsm["eigengene"].drop(columns="SampleID", inplace=True)
        ad_meta.varm["KME"] = r2py(R("geneInfo"))
        ad_meta.varp["TOM"] = r2py(R("TOM"))

        # cyt = R("cyt")
        # df_edge = r2py(rGet(cyt, "$edgeData"))
        # df_node = r2py(rGet(cyt, "$nodeData"))

        # dt_cyt = {"node": df_node, "edge": df_edge}
        # ad_meta.uns["cyt"] = dt_cyt
        ad_meta.uns[f"{jobid}_wgcna"] = {}
        ls_colorName = (
            ad_meta.varm["KME"]["Initially.Assigned.Module.Color"].unique().tolist()
        )
        dt_color = {x: gplots.col2hex(x)[0] for x in ls_colorName}
        ad_meta.uns[f"{jobid}_wgcna"]["colors"] = dt_color

        dt_moduleGene = (
            ad_meta.varm["KME"]
            .groupby("Initially.Assigned.Module.Color")
            .apply(lambda df: df.index.to_list())
            .to_dict()
        )
        ad_meta.uns[f"{jobid}_wgcna"]["genes"] = dt_moduleGene

        dt_moduleGene = {x: y for x, y in dt_moduleGene.items() if x != "grey"}
        axs = plotting.clustermap(
            ad_meta,
            dt_moduleGene,
            obsAnno=ls_obs,
            layer="normalize_log",
            #     standard_scale=1,
            space_obsAnnoLegend=0.3,
            cbarPos=None,
            dt_geneColor=dt_color,
        )
        plt.show()

        ad_metaEigengene = sc.AnnData(
            ad_meta.obsm["eigengene"], obs=ad_meta.obs, uns=ad_meta.uns
        )
        _dt = {x.split("ME")[-1]: [x] for x in ad_metaEigengene.var.index}
        axs = plotting.clustermap(
            ad_metaEigengene,
            _dt,
            obsAnno=ls_obs,
            layer=None,
            standard_scale=1,
            space_obsAnnoLegend=0.3,
            cbarPos=None,
            dt_geneColor=dt_color,
        )
        plt.show()

        ad_meta.varm["KME"] = ad_meta.varm["KME"].assign(
            kME=lambda df: df.apply(
                lambda x: x.at["kME" + x.at["Initially.Assigned.Module.Color"]], axis=1
            ),
            module=lambda df: df["Initially.Assigned.Module.Color"],
        )

        if calcAucell:
            getAUCellScore(
                ad,
                dt_moduleGene,
                layer,
                aucMaxRank=1000,
                label=f"{jobid}_AUCell",
                rEnv=renv,
            )
            if "X_umap" in ad.obsm:
                sc.pl.umap(
                    plotting.obsmToObs(ad, f"{jobid}_AUCell"),
                    color=ad.obsm[f"{jobid}_AUCell"].columns,
                    cmap="Reds",
                )
    # rlc.__exit__(None, None, None)
    ro.r.gc()
    return ad_meta


def _mergeData(ad, obsKey, layer="raw"):
    basic.testAllCountIsInt(ad, layer)
    df_oneHot = pd.DataFrame(
        index=ad.obs.index, columns=ad.obs[obsKey].cat.categories
    ).fillna(0)
    for col in df_oneHot.columns:
        df_oneHot.loc[ad.obs[obsKey] == col, col] = 1
    ad_merge = sc.AnnData(
        df_oneHot.values.T @ ad.layers[layer],
        obs=pd.DataFrame(index=df_oneHot.columns),
        var=pd.DataFrame(index=ad.var.index),
    )
    return ad_merge


@rcontext
def timeSeriesAnalysisByMfuzz(
    ad,
    *,
    timeObs,
    clusterObs=None,
    filterGeneThres=0.25,
    fillNaGeneMethod="mean",
    geneClusterCounts=range(5, 75, 5),
    rFigsize=256,
    rcol=4,
    showNLargest=0,
    keyAdded="mfuzz",
    minMembership=0.3,
    palette="terrain_r",
    forcePlotAll=False,
    repeats=3,
    threads=24,
    rEnv=None,
):
    """`timeSeriesAnalysisByMfuzz` is a function that takes in a single cell RNA-seq dataframe, performs
    time series analysis using the `mfuzz` R package

    Parameters
    ----------
    ad
        AnnData object
    timeObs
        the time points of the experiment
    filterGeneThres
        the threshold for filtering out genes with low expression.
    fillNaGeneMethod, optional
        How to fill NA values in the gene expression matrix.
    geneClusterCounts, optional
        number of gene clusters
    rFigsize
        the size of the figure to be generated by R
    rMfrow
        the number of rows and columns of plots to show in the R plot
    showNLargest, optional
        number of genes to show in the plot, if 0 plot all
    keyAdded, optional
        the key to be added to the adata object.
    rEnv
        R environment to use. If None, a new one will be created.

    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from math import ceil
    import pickle
    from . import plotting
    from ..rTools import (
        py2r,
        r2py,
        r_inline_plot,
        rHelp,
        trl,
        rGet,
        rSet,
        ad2so,
        so2ad,
        r_set_seed,
    )

    rBase = importr("base")
    rUtils = importr("utils")
    Mfuzz = importr("Mfuzz")
    importr("DescTools")
    R = ro.r
    r_set_seed(0)
    # print(1)
    def _calcClusterDistance(pkl_eset, pkl_m1, c, repeats):
        eset = pickle.loads(pkl_eset)
        m1 = pickle.loads(pkl_m1)
        ro.r("set.seed")(0)
        Mfuzz = importr("Mfuzz")
        result = Mfuzz.Dmin(eset, m=m1, crange=c, repeats=repeats, visu=False)[0]
        return c, result

    def _timeSeriesPlot(data, x, y, cmap="terrain_r", forcePlotAll=False, **kws):
        dt_membership = (
            data.loc[data["gene"].duplicated()]
            .set_index("gene")["membership"]
            .to_dict()
        )
        cmap = sns.palettes.color_palette(cmap, as_cmap=True)
        _df = data.pivot_table(y, index="gene", columns=x).reindex(
            sorted(dt_membership.keys(), key=lambda x: dt_membership[x])
        )
        if not forcePlotAll:
            _df = _df.iloc[-1000:]
        for gene, ar_exp in _df.iterrows():
            plt.plot(range(len(ar_exp)), ar_exp, c=cmap(int(dt_membership[gene] * 255)))
        plt.xticks(range(len(ar_exp)), data[x].cat.categories)

    # assert (rMfrow[0] * rMfrow[1]) >= geneClusterCounts, "rMfrow[0] * rMfrow[1] < geneClusterCounts"
    if clusterObs is None:
        ad_merged = _mergeData(ad, timeObs)
    else:
        dt_adMerged = {}
        for clusterName, ad_cluster in basic.splitAdata(
            ad, "leiden", needName=True, copy=False
        ):
            _ad = _mergeData(ad_cluster, "Sample")
            _ad.var.index = _ad.var.index + "||" + clusterName
            dt_adMerged[clusterName] = _ad
        ad_merged = sc.concat(dt_adMerged, axis=1)
    basic.initLayer(ad_merged, total=1e6)  # cpm
    esR_Merged = R.ExpressionSet(
        rBase.as_matrix(py2r(ad_merged.to_df("normalize_log").replace(0, np.nan).T))
    )  # esR: ExpressionSet # mfuzz only recognized NA as missing value

    rEnv["filterGeneThres"] = filterGeneThres
    rEnv["fillNaGeneMethod"] = fillNaGeneMethod
    rEnv["esR_Merged"] = esR_Merged
    rEnv["minMembership"] = minMembership

    R(
        """
    esR_Merged.r <- filter.NA(esR_Merged, thres=filterGeneThres)
    esR_Merged.f <- fill.NA(esR_Merged.r,mode=fillNaGeneMethod)
    esR_Merged.s <- standardise(esR_Merged.f)
    m1 <- mestimate(esR_Merged.s)
    """
    )

    if isinstance(geneClusterCounts, int):
        pass
    else:
        ls_dminRes = Parallel(threads)(
            delayed(_calcClusterDistance)(
                pickle.dumps(rEnv["esR_Merged.s"]), pickle.dumps(rEnv["m1"]), x, repeats
            )
            for x in geneClusterCounts
        )
        plt.scatter(
            [x[0] for x in ls_dminRes], [x[1] for x in ls_dminRes], color="black", s=10
        )
        plt.show()
        # arR_crange = R.c(*geneClusterCounts)
        # dt_kwargsToDmin['crange'] = arR_crange
        # dt_kwargsToDmin['visu'] = True
        # dtR_kwargsToDmin = R.list(**dt_kwargsToDmin)
        # rEnv['dtR_kwargsToDmin'] = dtR_kwargsToDmin
        # with r_inline_plot():
        #     R("""
        #     dtR_kwargsToDmin$eset <- esR_Merged.s
        #     dtR_kwargsToDmin$m <- m1
        #     DoCall(Dmin, dtR_kwargsToDmin)
        #     """)

        geneClusterCounts = int(input())
    rrow = ceil(geneClusterCounts / rcol)
    rMfrow = (rrow, rcol)
    rEnv["geneClusterCounts"] = geneClusterCounts

    r_set_seed(0)
    R("cl <- mfuzz(esR_Merged.s, c=geneClusterCounts, m=m1)")

    vtR_sampleLabel = R.c(*ad.obs[timeObs].cat.categories)
    rEnv["vtR_sampleLabel"] = vtR_sampleLabel
    rEnv["rMfrow"] = R.c(*rMfrow)

    with r_inline_plot(rFigsize * rcol, rFigsize * rrow):
        R(
            "mfuzz.plot(esR_Merged.s,cl=cl,mfrow=rMfrow,new.window=FALSE, time.labels=vtR_sampleLabel)"
        )

    vtR_cluster = R("cl$cluster")
    df_ms = r2py(R("as.data.frame(cl$membership)"))
    dt_cluster = {x: y for x, y in zip(vtR_cluster.names, vtR_cluster)}
    df_ms = df_ms.assign(cluster=lambda df: df.index.map(dt_cluster).astype(str))
    df_ms = df_ms.assign(
        membership=lambda df: df.apply(lambda x: x.loc[x.loc["cluster"]], axis=1)
    )
    df_ms = df_ms.sort_values("cluster", key=lambda x: list(map(int, x)))
    sns.boxplot(data=df_ms.query("membership > 0.1"), x="cluster", y="membership")
    plt.show()

    df_mmPassed = (
        (
            df_ms.groupby("cluster")
            .apply(lambda df: df.nlargest(showNLargest, "membership"))
            .reset_index(level=0, drop=True)
        )
        if showNLargest > 0
        else df_ms
    )
    df_mmPassed = df_mmPassed.query("membership > @minMembership")
    df_MergedScaledMtx = r2py(rBase.as_data_frame(R.exprs(R("esR_Merged.s")))).T
    # _ad = ad_merged[:, df_mmPassed.index]
    # _ad.layers["normalize_log_scaled"] = _ad.layers["normalize_log"].copy()
    # sc.pp.scale(_ad, layer="normalize_log_scaled")

    _df = (
        df_MergedScaledMtx.melt(ignore_index=False, var_name="gene", value_name="exp")
        .rename_axis(timeObs)
        .reset_index()
        .merge(df_mmPassed[["cluster", "membership"]], left_on="gene", right_index=True)
        .sort_values("cluster", key=lambda x: list(map(int, x)))
    )

    _df[timeObs] = (
        _df[timeObs]
        .astype("category")
        .cat.set_categories(ad.obs[timeObs].cat.categories)
    )
    _df["cluster"] = (
        _df["cluster"]
        .astype("category")
        .cat.set_categories(sorted(_df["cluster"].unique(), key=int))
    )
    axs = sns.FacetGrid(
        _df,
        col="cluster",
        col_wrap=rMfrow[-1],
    )
    axs.map_dataframe(
        _timeSeriesPlot, x=timeObs, y="exp", cmap=palette, forcePlotAll=forcePlotAll
    )
    dt_geneCategoryCounts = (
        _df.groupby("cluster")["gene"].agg(lambda x: len(x.unique())).to_dict()
    )
    for ax, (category, geneCounts) in zip(
        axs.axes.flatten(), dt_geneCategoryCounts.items()
    ):
        plt.sca(ax)
        plt.title(f"Gene category {category}\n(N = {geneCounts})")
    # add colorbar
    ax_cbar = axs.fig.add_axes([1.015, 0.4, 0.015, 0.2])
    plt.sca(ax_cbar)
    cmap = mpl.cm.get_cmap(palette)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm)
    cb.set_ticks([])
    plt.text(x=1, y=0.5, s="Membership value", ha="left", va="center", rotation=270)

    plt.tight_layout()
    plt.show()
    # axs = sns.FacetGrid(
    #     _df,
    #     col="cluster",
    #     col_wrap=rMfrow[-1],
    #     # hue="membership",
    #     # palette="viridis_r",
    #     #     hue_kws=dict(hue_norm=(0, 1)),
    # )
    # axs.map_dataframe(sns.lineplot, x=timeObs, y="exp", hue_norm=(0, 1), ci=None)
    # axs.set_titles(col_template="Gene category {col_name}")
    # plt.show()
    df_ms = df_ms.rename(columns={"cluster": "geneCategory"})
    ad_merged.varm[keyAdded] = df_ms.reindex(ad_merged.var.index)
    if clusterObs is None:
        ad.varm[keyAdded] = df_ms.reindex(ad.var.index)
    else:
        df_mfuzzMembership = (
            (
                df_ms.dropna().assign(
                    cellCluster=lambda df: df.index.str.split("\|\|").str[-1],
                    geneName=lambda df: df.index.str.split("\|\|").str[0],
                )
            )
            .set_index(["geneName", "cellCluster"])
            .unstack()
        )
        ad.varm[keyAdded] = df_mfuzzMembership.reindex(ad.var.index)

    return ad_merged
