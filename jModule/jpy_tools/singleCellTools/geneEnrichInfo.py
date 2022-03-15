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


def getBgGene(ad, ls_gene, layer="normalize_log", bins=50, seed=0, usePreBin:str = None, multi = 1, replacement = True):
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
                n=dt_binGeneCounts[df["bins_ForPickMock"].iloc[0]] * multi, random_state=seed
            )
        )
        .index.to_list()
    )
    return ls_randomGene

def getGeneModuleEnrichScore(ad, layer, ls_gene, times=100, groupby=None, targetOnly=True, disableBar=False, multi=False, **dt_paramsGetBgGene):
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
            dt_groupMeanExp[f"shuffle_{i}"] = df_scaledExp[ls_bgGeneSingle].mean().mean()
        dt_scaledMeanExp[name] = dt_groupMeanExp
    df_scaledMeanExp = pd.DataFrame(dt_scaledMeanExp).apply(zscore)
    if targetOnly:
        return df_scaledMeanExp.loc['target'].to_dict()
    else:
        return df_scaledMeanExp

def getGeneModuleEnrichScore_multiList(ad, layer, dt_gene, times=100, groupby=None, bins=50, **dt_paramsGetBgGene):
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
    for name, ls_gene in tqdm(dt_gene.items(), 'get gene module enrich score', len(dt_gene)):
        dt_result[name] = getGeneModuleEnrichScore(
            ad, layer, ls_gene, times, groupby, disableBar=True, multi=True, **dt_paramsGetBgGene
        )
    return pd.DataFrame(dt_result)

def getUcellScore(
    ad: sc.AnnData,
    dt_deGene: Mapping[str, List[str]],
    layer: Optional[str],
    label,
    cutoff=0.2,
    batch = None,
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
        # import pdb; pdb.set_trace()
        ad.obsm[f"ucell_score_{label}"] = pd.concat(dfLs_ucellResults).reindex(ad.obs.index)

    else:
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

def getGeneScore(ad: sc.AnnData, dt_Gene: Dict[str, List[str]], layer: Optional[str], label:str, func:Callable):
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
    df_results = pd.DataFrame(index = ad.obs.index, columns = dt_Gene.keys())
    for name, ls_gene in dt_Gene.items():
        _sr = ad[:, ls_gene].to_df(layer).apply(func, axis = 1)
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

    def _singleBatch(adata, layer, clusterName):
        df_mtx = adata.to_df(layer).T if layer else adata.to_df().T
        df_meta = adata.obs[[clusterName]].rename({clusterName: "cell_type"}, axis=1)
        eso = cellex.ESObject(data=df_mtx, annotation=df_meta)
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
        _singleBatch(adata, layer, clusterName)
    else:
        ls_batchAd = basic.splitAdata(adata, batchKey, needName=True)
        adata.uns[f"{clusterName}_cellexES"] = {}
        for sample, ad_batch in ls_batchAd:
            _singleBatch(ad_batch, layer, clusterName)
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
            x, y, left_on=("gene", clusterName), right_on=("gene", clusterName), how='outer'
        ),
        [sr_geneCounts, *lsDf_Concat],
    ).query("counts >= @minCounts")
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
        plt.yscale('log')
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