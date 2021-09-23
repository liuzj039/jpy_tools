"""
basic tools
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


def getOverlap(ad_a: anndata.AnnData, ad_b: anndata.AnnData, copy=False):
    ls_geneOverlap = list(ad_a.var.index & ad_b.var.index)
    logger.info(f"Used Gene Counts: {len(ls_geneOverlap)}")
    if copy:
        return ad_a[:, ls_geneOverlap].copy(), ad_b[:, ls_geneOverlap].copy()
    else:
        return ad_a[:, ls_geneOverlap], ad_b[:, ls_geneOverlap]


def splitAdata(
    adata: anndata.AnnData,
    batchKey: str,
    copy=True,
    axis: Literal[0, "cell", 1, "feature"] = 0,
    needName = False
) -> Iterator[anndata.AnnData]:
    if axis in [0, "cell"]:
        assert batchKey in adata.obs.columns, f"{batchKey} not detected in adata"
        indexName = "index" if (not adata.obs.index.name) else adata.obs.index.name
        adata.obs["__group"] = adata.obs[batchKey]
        batchObsLs = (
            adata.obs.filter(["__group"])
            .reset_index()
            .groupby("__group")[indexName]
            .agg(list)
        )
        for batchObs in tqdm(batchObsLs):
            if needName:
                if copy:
                    yield adata[batchObs].obs.iloc[0].loc["__group"], adata[batchObs].copy()
                else:
                    yield adata[batchObs].obs.iloc[0].loc["__group"], adata[batchObs]
            else:
                if copy:
                    yield adata[batchObs].copy()
                else:
                    yield adata[batchObs]
        del adata.obs["__group"]

    elif axis in [1, "feature"]:
        assert batchKey in adata.var.columns, f"{batchKey} not detected in adata"
        indexName = "index" if (not adata.var.index.name) else adata.var.index.name
        adata.var["__group"] = adata.var[batchKey]
        batchVarLs = (
            adata.var.filter(["__group"])
            .reset_index()
            .groupby("__group")[indexName]
            .agg(list)
        )
        del adata.var["__group"]
        for batchVar in tqdm(batchVarLs):
            if needName:
                if copy:
                    yield adata[batchVar].var.iloc[0].loc["__group"], adata[batchVar].copy()
                else:
                    yield adata[batchVar].var.iloc[0].loc["__group"], adata[batchVar]
            else:
                if copy:
                    yield adata[:, batchVar].copy()
                else:
                    yield adata[:, batchVar]
        del adata.var["__group"]
    else:
        assert False, "Unknown `axis` parameter"


def groupAdata(adata: sc.AnnData, batchKey: str, function, axis=0, **params):
    def __concatAd(ls_result, axis, ls_indexOrder, ls_batchContents):
        ad_final = sc.concat(ls_result, axis=axis)
        ad_final = (
            ad_final[ls_indexOrder]
            if axis in [0, "cell"]
            else ad_final[:, ls_indexOrder]
        )
        return ad_final

    def __concatDf(ls_result, axis, ls_indexOrder, ls_batchContents):
        df_final = pd.concat(ls_result, axis=1)
        df_final.columns = ls_batchContents
        return df_final

    ls_indexOrder = adata.obs.index if axis in [0, "cell"] else adata.var.index

    ls_batchContents = (
        list(adata.obs[batchKey].sort_values().unique())
        if axis in [0, "cell"]
        else list(adata.var[batchKey].sort_values().unique())
    )

    it_ad = basic.splitAdata(adata, batchKey, copy=False, axis=axis)
    ls_result = []
    for _ad in it_ad:
        ls_result.append(function(_ad, **params))

    if isinstance(ls_result[0], sc.AnnData):
        return __concatAd(ls_result, axis, ls_indexOrder, ls_batchContents)
    elif isinstance(ls_result[0], pd.DataFrame) | isinstance(ls_result[0], pd.Series):
        return __concatDf(ls_result, axis, ls_indexOrder, ls_batchContents)
    else:
        assert False, "Unsupported data type"


def getPartialLayersAdata(
    adata: anndata.AnnData,
    layers: Optional[Union[str, List[str]]] = None,
    obsInfoLs: Optional[Sequence[str]] = None,
    varInfoLs: Optional[Sequence[str]] = None,
) -> anndata.AnnData:
    """
    get a subset of adata. Only contains one layer expression matrix, and several obs information.

    Parameters
    ----------
    adata : anndata.AnnData
    layers : Optional[Union[str, List[str]]], optional
        None will be parsed as 'X', by default None
    obsInfoLs : Optional[Sequence[str]], optional
        by default None
    varInfoLs : Optional[Sequence[str]], optional
        by default None

    Returns
    -------
    anndata.AnnData
        if data type of `layers` is list, all element in 'X' of returned adata will be set as 0
    """
    import scipy.sparse as ss

    if not obsInfoLs:
        obsInfoLs = []
    else:
        assert sum([x in adata.obs.columns for x in obsInfoLs]) == len(
            obsInfoLs
        ), "Requested feature not located in adata.obs"
    if not varInfoLs:
        varInfoLs = []
    else:
        assert sum([x in adata.var.columns for x in varInfoLs]) == len(
            varInfoLs
        ), "Requested feature not located in adata.var"

    if not layers:
        layers = "X"
    if isinstance(layers, list):
        dt_layerMtx = {}
        for layer in layers:
            ar_mtx = adata.X if layer == "X" else adata.layers[layer]
            dt_layerMtx[layer] = ar_mtx
        subAd = anndata.AnnData(
            ss.csr_matrix(np.zeros(adata.shape)),
            adata.obs[obsInfoLs],
            adata.var[varInfoLs],
            layers=dt_layerMtx,
        )

    elif isinstance(layers, str):
        layer = layers
        mtxAr = adata.X if layer == "X" else adata.layers[layer]
        subAd = anndata.AnnData(mtxAr, adata.obs[obsInfoLs], adata.var[varInfoLs])

    else:
        assert False, f"unsupported layers data type: {type(layers)}"

    return subAd.copy()


def testAllCountIsInt(adata: anndata.AnnData, layer: Optional[str]) -> None:
    """
    Test whether all counts is int
    """
    import scipy.sparse as sp

    if layer == "X":
        layer = None

    testColCounts = min([10, adata.shape[0]])
    if not layer:
        X_subset = adata.X[:testColCounts]
    else:
        X_subset = adata.layers[layer][:testColCounts]

    err = (
        "Make sure that adata.layer contains unnormalized count data"
        + f"\tLayer:{layer}"
    )
    if sp.issparse(X_subset):
        assert (X_subset.astype(int) != X_subset).nnz == 0, err
    else:
        assert np.all(X_subset.astype(int) == X_subset), err


def getadataColor(adata, label):
    if f"{label}_colors" not in adata.uns:
        basic.setadataColor(adata, label)
    return {
        x: y
        for x, y in zip(adata.obs[label].cat.categories, adata.uns[f"{label}_colors"])
    }


def setadataColor(adata, label, colorDt=None, hex=True):
    adata.obs[label] = adata.obs[label].astype("category")
    if colorDt:
        if not hex:
            from matplotlib.colors import to_hex

            colorDt = {x: to_hex(y) for x, y in colorDt.items()}
        adata.uns[f"{label}_colors"] = [
            colorDt[x] for x in adata.obs[label].cat.categories
        ]
    else:
        if f"{label}_colors" not in adata.uns:
            sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)

    return adata


def creatAnndataFromDf(df, **layerInfoDt):
    """
    df,
    layerInfoDt:
        key: layer name
        value: mtx
    column is barcode raw is feature
    """
    transformedAd = anndata.AnnData(
        X=df.values,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    for layerName, layerMtx in layerInfoDt.items():

        transformedAd.layers[layerName] = layerMtx

    return transformedAd


def mergeadata(adata, groupby, mergeLayer=[], method="sum"):
    """
    通过adata.obs中的<groupby>合并X和layer
    """
    adataXDf = adata.to_df()
    groupbyXDf = adataXDf.join(adata.obs[groupby]).groupby(groupby).agg(method)

    adataLayerDfDt = {}
    for singleLayer in mergeLayer:
        adataLayerDfDt[singleLayer] = (
            adata.to_df(singleLayer)
            .join(adata.obs[groupby])
            .groupby(groupby)
            .agg(method)
        )
    return basic.creatAnndataFromDf(groupbyXDf, **adataLayerDfDt)


def clusterBySC3(
    adata: anndata.AnnData,
    layer: str,
    clusterNum: Union[int, Sequence[int]],
    layerIsLogScaled: bool = True,
    biologyInfo: bool = False,
    threads: int = 24,
    needSCE: bool = False,
    copy: bool = False,
) -> Tuple[Optional[anndata.AnnData], Optional[Any]]:
    """
    Cluster by SC3

    Parameters
    ----------
    adata : anndata.AnnData
        anndata
    layer : str
        use this layer as input for SC3. WARNING: By default, this layer is log-scaled.
    clusterNum : Union[int, list]
        cluster counts.
    layerIsLogScaled: bool
        To Indicate whether layer is log-scaled or not. by default True
    biologyInfo : bool, optional
        need biology info or not. It is means that the DEG, marker, and others. This information will stored in var. by default False
    threads : int, optional
        by default 24
    needSCE : bool, optional
        need sce object or not. this object could be used for plot. by default False
    copy : bool, optional
        by default False

    Returns
    -------
    Tuple[Optional[anndata.AnnData], Optional[Any]]
        anndata and singleCellExperiment. DEPOND ON copy and needSCE
    """
    import scipy.sparse as ss
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from .rTools import r2py, py2r

    R = ro.r

    adata = adata.copy() if copy else adata

    importr("SC3")
    importr("SingleCellExperiment")
    useMtx = adata.layers[layer] if layer != "X" else adata.X
    if layerIsLogScaled:
        useMtx = (
            np.exp(useMtx) - 1 if isinstance(useMtx, np.ndarray) else useMtx.expm1()
        )

    _adata = anndata.AnnData(
        None, obs=adata.obs[[]], var=adata.var[[]], layers=dict(counts=useMtx)
    )
    _adata.var["feature_symbol"] = _adata.var.index
    _adata.layers["logcounts"] = sc.pp.log1p(_adata.layers["counts"], copy=True)

    logger.info("transform data to R")
    sceObj = py2r(_adata)
    logger.info("transform end")

    setAssay = R("`assay<-`")
    sceObj = setAssay(sceObj, "counts", value=R("as.matrix")(R.assay(sceObj, "counts")))
    sceObj = setAssay(
        sceObj, "logcounts", value=R("as.matrix")(R.assay(sceObj, "logcounts"))
    )

    if isinstance(clusterNum, Sequence):
        clusterNum = np.array(clusterNum)
    else:
        clusterNum = np.array([clusterNum])  # if is int, transform it to list.

    sceObj = R.sc3(sceObj, ks=py2r(clusterNum), biology=biologyInfo, n_cores=threads)
    adata.uns[f"SC3_consensus"] = {}
    # trainSvmObsIndexSr = r2py(
    #     R.metadata(sceObj).rx2["sc3"].rx2["svm_train_inds"]
    # ).copy()  # To record obs which used for calculate consensus matrix
    # adata.uns[f"SC3_consensus"]["useObs"] = adata.obs.index[
    #     trainSvmObsIndexSr
    # ].values

    if _adata.shape[0] > 5000:
        logger.info("To start predicts cell labels by SVM")
        sceObj = R.sc3_run_svm(sceObj, ks=py2r(clusterNum))
        if biologyInfo:
            logger.info("To start calculates biology information")
            ro.globalenv["sceObj"] = sceObj
            R("metadata(sceObj)$sc3$svm_train_inds <- NULL")
            sceObj = R.sc3_calc_biology(sceObj, ks=clusterNum)

    adata.obs = adata.obs.combine_first(r2py(R.colData(sceObj))).copy()
    adata.var = adata.var.combine_first(r2py(R.rowData(sceObj))).copy()

    # for singleClusterNum in clusterNum:
    #     singleClusterNum = str(singleClusterNum)
    #     adata.uns["SC3_consensus"][singleClusterNum] = r2py(
    #         sceObj.slots["metadata"]
    #         .rx2["sc3"]
    #         .rx2["consensus"]
    #         .rx2[singleClusterNum]
    #         .rx2["consensus"]
    #     ).copy()

    returnAd = adata if copy else None
    returnSe = sceObj if needSCE else None
    # with r_inline_plot():
    #     R.sc3_plot_consensus(sceObj, k=3, show_pdata=py2r(np.array(["sc3_3_clusters", "sc3_4_clusters"])))
    return returnAd, returnSe


@staticmethod
def constclustWriteResult(path, params, clusterings, adata):
    with h5py.File(path, "w") as f:
        cluster_group = f.create_group("clusterings")
        cluster_group.create_dataset(
            "clusterings", data=clusterings.values, compression="lzf"
        )
        cluster_group.create_dataset(
            "obs_names", data=adata.obs_names.values, compression="lzf"
        )

        params_group = f.create_group("params")
        for k, v in params.items():
            params_group.create_dataset(k, data=v.values, compression="lzf")


@staticmethod
def constclustReadResult(path) -> "Tuple[pd.DataFrame, pd.DataFrame]":
    """Read params and clusterings which have been stored to disk."""
    with h5py.File(path, "r") as f:
        params_group = f["params"]
        params = pd.DataFrame(
            {
                col: params_group[col]
                for col in ["n_neighbors", "resolution", "random_state"]
            }
        )
        cluster_group = f["clusterings"]
        clusterings = pd.DataFrame(
            cluster_group["clusterings"][()],
            index=cluster_group["obs_names"].asstr()[:],
        )
    return params, clusterings


def constclustFlatLabelling(
    complist,
    obs_names: pd.Index,
    cutoff: float = 0.75,
    compNameLs: Optional[Union[Sequence[str], Sequence[int], Mapping[int, str]]] = None,
    figsizePerComponent: float = 0.4,
    start_num: int = 1,
) -> pd.Series:
    """
    Convenience function for creating a flat labelling from some components.


    Parameters
    ----------
    complist : [type]
        rec.get_components results
    obs_names : pd.Index
        adata.obs_names
    cutoff : float, optional
        used to determine which cell is included in component, by default 0.75
    compNameLs: Optional[Union[Sequence[str], Sequence[int], Mapping[int,str]]], optional
        if Sequence and content is string, the length must same as the complist;
        if Sequence and content is int, the sum must same as the complist;
        if Mapping, only these components located in the Mapping's key will be used, and values are corresponding names.

    Returns
    -------
    pd.Series
        This series should be stored in adata's obs attribute
    """
    from itertools import product

    def __getObsName(comp, obs_names, cutoff):
        cell_value = pd.Series(0, obs_names, dtype=float)
        for cluster in comp.cluster_ids:
            cell_value[comp._parent._mapping.iloc[cluster]] += 1
        cell_value = cell_value / cell_value.max()
        cell_value = cell_value[cell_value >= cutoff].index
        return cell_value

    compLength = len(complist)
    if not compNameLs:
        compNameLs = [str(x) for x in range(compLength)]
    else:
        if isinstance(compNameLs, Sequence):
            if isinstance(compNameLs[0], str):
                assert (
                    len(compNameLs) == compLength
                ), "compNameLs length is not equals to complist length"

            elif isinstance(compNameLs[0], int):
                logger.warning("compNameLs will be re-parsed")
                assert (
                    sum(compNameLs) == compLength
                ), "compNameLs length is not equals to complist length"
                compNameLs_ = []
                for mainNum, subCounts in enumerate(compNameLs, start=start_num):
                    if subCounts == 1:
                        compNameLs_.append(f"{mainNum}")
                    else:
                        for subNum in range(start_num, subCounts + start_num):
                            compNameLs_.append(f"{mainNum}-{subNum}")
                compNameLs = compNameLs_
                del compNameLs_

        elif isinstance(compNameLs, Mapping):
            useCompLs = list(compNameLs.keys())
            complist = complist[useCompLs]
            compNameLs = [compNameLs[x] for x in useCompLs]
            compLength = len(complist)
        else:
            assert False, "unsupported data type"

    logger.info("Start parsing components contents")
    flatObsNameWithCompNameLs = []
    for i, comp in enumerate(complist):
        compObsLs = __getObsName(comp, obs_names, cutoff)
        compName = compNameLs[i]
        for singleObs in compObsLs:
            flatObsNameWithCompNameLs.append([singleObs, compName])

    logger.info("Start get overlap among components")
    flatObsNameWithCompNameDf = pd.DataFrame(
        flatObsNameWithCompNameLs, columns=["obsName", "compName"]
    )
    flatObsNameWithCompNameDt = (
        flatObsNameWithCompNameDf.groupby("compName")["obsName"].agg(set).to_dict()
    )
    overlapCountInfo = np.zeros([compLength, compLength], dtype=int)
    for i, j in product(range(compLength), range(compLength)):
        if i >= j:
            overlapCountInfo[i, j] = len(
                flatObsNameWithCompNameDt[compNameLs[i]]
                & flatObsNameWithCompNameDt[compNameLs[j]]
            )

    overlapCountInfo = np.triu(overlapCountInfo.T, 1) + overlapCountInfo
    overlapCountInfoDf = pd.DataFrame(
        overlapCountInfo, index=compNameLs, columns=compNameLs
    )

    figsize = (
        figsizePerComponent * compLength + 1,
        figsizePerComponent * compLength,
    )
    compAddOriNumLs = [f"{x} ({y})" for x, y in zip(compNameLs, complist._comps.index)]
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        overlapCountInfoDf / overlapCountInfoDf.max(0),
        cmap="Reds",
        ax=ax,
        annot=overlapCountInfoDf,
        fmt=".4g",
    )
    plt.title("overlap info among all components")
    plt.xticks([x + 0.5 for x in range(compLength)], compAddOriNumLs, rotation=90)
    plt.yticks([x + 0.5 for x in range(compLength)], compAddOriNumLs, rotation=0)
    plt.show()

    logger.info("remove overlap of components")
    flatObsNameWithDropSq = flatObsNameWithCompNameDf.drop_duplicates("obsName")[
        "obsName"
    ]
    flatObsNameWithCompNameSr = (
        flatObsNameWithCompNameDf.drop_duplicates("obsName", keep=False)
        .set_index("obsName")
        .reindex(flatObsNameWithDropSq)
        .fillna("overlap")
        .reindex(obs_names)
        .fillna("unstable")["compName"]
    )
    return flatObsNameWithCompNameSr.astype("category")


def scIB_hvg_batch(
    adata,
    batch_key=None,
    target_genes=2000,
    flavor="cell_ranger",
    n_bins=20,
    adataOut=False,
):
    """
    forked from scib

    Method to select HVGs based on mean dispersions of genes that are highly
    variable genes in all batches. Using a the top target_genes per batch by
    average normalize dispersion. If target genes still hasn't been reached,
    then HVGs in all but one batches are used to fill up. This is continued
    until HVGs in a single batch are considered.
    """

    def checkadata(adata):
        if type(adata) is not anndata.AnnData:
            raise TypeError("Input is not a valid AnnData object")

    def checkBatch(batch, obs, verbose=False):
        if batch not in obs:
            raise ValueError(f"column {batch} is not in obs")
        elif verbose:
            print(f"Object contains {obs[batch].nunique()} batches.")

    checkadata(adata)
    if batch_key is not None:
        checkBatch(batch_key, adata.obs)

    adata_hvg = adata if adataOut else adata.copy()

    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    # Calculate double target genes per dataset
    sc.pp.highly_variable_genes(
        adata_hvg,
        flavor=flavor,
        n_top_genes=target_genes,
        n_bins=n_bins,
        batch_key=batch_key,
    )

    nbatch1_dispersions = adata_hvg.var["dispersions_norm"][
        adata_hvg.var.highly_variable_nbatches
        > len(adata_hvg.obs[batch_key].cat.categories) - 1
    ]

    nbatch1_dispersions.sort_values(ascending=False, inplace=True)

    if len(nbatch1_dispersions) > target_genes:
        hvg = nbatch1_dispersions.index[:target_genes]

    else:
        enough = False
        print(f"Using {len(nbatch1_dispersions)} HVGs from full intersect set")
        hvg = nbatch1_dispersions.index[:]
        not_n_batches = 1

        while not enough:
            target_genes_diff = target_genes - len(hvg)

            tmp_dispersions = adata_hvg.var["dispersions_norm"][
                adata_hvg.var.highly_variable_nbatches == (n_batches - not_n_batches)
            ]

            if len(tmp_dispersions) < target_genes_diff:
                print(
                    f"Using {len(tmp_dispersions)} HVGs from n_batch-{not_n_batches} set"
                )
                hvg = hvg.append(tmp_dispersions.index)
                not_n_batches += 1

            else:
                print(
                    f"Using {target_genes_diff} HVGs from n_batch-{not_n_batches} set"
                )
                tmp_dispersions.sort_values(ascending=False, inplace=True)
                hvg = hvg.append(tmp_dispersions.index[:target_genes_diff])
                enough = True

    print(f"Using {len(hvg)} HVGs")

    if not adataOut:
        del adata_hvg
        return hvg.tolist()
    else:
        return adata_hvg[:, hvg].copy()


def selectCellFromObsm(
    adata: anndata.AnnData,
    xlim: Sequence[float],
    ylim: Sequence[float],
    obsmBasis: str = "X_umap",
    returnName: bool = True,
) -> pd.Series:
    "select cells based on location"
    locationAr = adata.obsm[obsmBasis]
    useCellBoolLs = ((locationAr[:, 0] > xlim[0]) & (locationAr[:, 0] < xlim[1])) & (
        (locationAr[:, 1] > ylim[0]) & (locationAr[:, 1] < ylim[1])
    )
    if returnName:
        return adata[useCellBoolLs].obs.index
    else:
        return useCellBoolLs


def scIB_scale_batch(adata, batch) -> anndata.AnnData:
    """
    Function to scale the gene expression values of each batch separately.
    """

    def checkadata(adata):
        if type(adata) is not anndata.AnnData:
            raise TypeError("Input is not a valid AnnData object")

    def checkBatch(batch, obs, verbose=False):
        if batch not in obs:
            raise ValueError(f"column {batch} is not in obs")
        elif verbose:
            print(f"Object contains {obs[batch].nunique()} batches.")

    def splitBatches(adata, batch, hvg=None, return_categories=False):
        split = []
        batch_categories = adata.obs[batch].unique()
        if hvg is not None:
            adata = adata[:, hvg]
        for i in batch_categories:
            split.append(adata[adata.obs[batch] == i].copy())
        if return_categories:
            return split, batch_categories
        return split

    def merge_adata(adata_list, sep="-"):
        """
        merge adatas from list and remove duplicated obs and var columns
        """

        if len(adata_list) == 1:
            return adata_list[0]

        adata = adata_list[0].concatenate(
            *adata_list[1:], index_unique=None, batch_key="tmp"
        )
        del adata.obs["tmp"]

        if len(adata.obs.columns) > 0:
            # if there is a column with separator
            if sum(adata.obs.columns.str.contains(sep)) > 0:
                columns_to_keep = [
                    name.split(sep)[1] == "0" for name in adata.var.columns.values
                ]
                clean_var = adata.var.loc[:, columns_to_keep]
            else:
                clean_var = adata.var

        if len(adata.var.columns) > 0:
            if sum(adata.var.columns.str.contains(sep)) > 0:
                adata.var = clean_var.rename(
                    columns={
                        name: name.split("-")[0] for name in clean_var.columns.values
                    }
                )

        return adata

    checkadata(adata)
    checkBatch(batch, adata.obs)

    # Store layers for after merge (avoids vstack error in merge)
    adata_copy = adata.copy()
    tmp = dict()
    for lay in list(adata_copy.layers):
        tmp[lay] = adata_copy.layers[lay]
        del adata_copy.layers[lay]

    split = splitBatches(adata_copy, batch)

    for i in split:
        sc.pp.scale(i, max_value=10)

    adata_scaled = merge_adata(split)

    # Reorder to original obs_name ordering
    adata_scaled = adata_scaled[adata.obs_names]

    # Add layers again
    for key in tmp:
        adata_scaled.layers[key] = tmp[key]

    del tmp
    del adata_copy

    return adata_scaled


def hvgBatch(
    adata: anndata.AnnData,
    batchKey: str,
    layer: Optional[str] = None,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "cell_ranger",
    singleBatchHvgCounts: int = 1000,
    keyAdded: str = "highly_variable",
    copy: bool = False,
    **highly_variable_genes_params,
) -> anndata.AnnData:
    from functools import reduce

    adata_org = adata
    adata = adata.copy()
    batchAdLs = list(basic.splitAdata(adata, batchKey))
    [
        sc.pp.highly_variable_genes(
            x,
            layer=layer,
            n_top_genes=singleBatchHvgCounts,
            flavor=flavor,
            **highly_variable_genes_params,
        )
        for x in batchAdLs
    ]
    finalHvgBoolLs = reduce(
        lambda a, b: a | b, [x.var.highly_variable for x in batchAdLs]
    )
    if copy:
        adata = sc.concat(batchAdLs)
        adata.var[keyAdded] = finalHvgBoolLs
        return adata
    else:
        adata_org.var[keyAdded] = finalHvgBoolLs


def saveMarkerGeneToPdf(
    adata: anndata.AnnData,
    outputDirPath: str,
    group: Optional[str] = None,
    key: str = "rank_genes_groups_filtered",
    layer: Optional[str] = None,
    pval_cutoff: float = 0.05,
    geneDt: Optional[Dict[str, List[str]]] = None,
    allGeneStoreDir: Optional[str] = None,
):
    """save all marker gene as pdf format"""
    import os
    import shutil
    from PyPDF2 import PdfFileMerger

    if allGeneStoreDir:
        allGeneStoreDir = allGeneStoreDir.rstrip("/") + "/"

    outputDirPath = outputDirPath.rstrip("/") + "/"
    if geneDt:
        markerDt = geneDt
    else:
        markerDf = sc.get.rank_genes_groups_df(
            adata, group=group, key=key, pval_cutoff=pval_cutoff
        )
        markerDt = markerDf.groupby("group")["names"].agg(list).to_dict()
    for groupName, groupMarkerGeneLs in markerDt.items():
        pdfMerger = PdfFileMerger()
        groupMarkerPathLs = []
        for gene in groupMarkerGeneLs:
            if allGeneStoreDir:
                shutil.copyfile(
                    f"{allGeneStoreDir}{gene}.pdf",
                    f"{outputDirPath}{groupName}_{gene}.pdf",
                )
            else:
                sc.pl.umap(adata, layer=layer, color=gene, cmap="Reds", show=False)
                plt.savefig(f"{outputDirPath}{groupName}_{gene}.pdf", format="pdf")
            pdfMerger.append(f"{outputDirPath}{groupName}_{gene}.pdf")
            groupMarkerPathLs.append(f"{outputDirPath}{groupName}_{gene}.pdf")
        pdfMerger.write(f"{outputDirPath}{groupName}_all.pdf")
        pdfMerger.close()
        [os.remove(x) for x in groupMarkerPathLs]
        logger.info(f"{groupName} finished")
    logger.info("All finished")


def saveAllGeneEmbedding(
    adata: anndata.AnnData,
    outputDirPath: str,
    layer: Optional[str] = None,
    useRaw: Optional[bool] = None,
    batch: Optional[str] = None,
):
    # def __saveSingleGene(gene):
    #     nonlocal adata
    #     nonlocal layer
    #     nonlocal outputDirPath
    #     sc.pl.umap(adata, layer=layer, color=gene, cmap="Reds", show=False)
    #     plt.savefig(f"{outputDirPath}{gene}.pdf", format="pdf")

    # from concurrent.futures import ThreadPoolExecutor

    outputDirPath = outputDirPath.rstrip("/") + "/"
    if layer:
        useRaw = False
    if useRaw is None:
        if adata.raw:
            useRaw = True
        else:
            useRaw = False

    if useRaw:
        allGeneLs = adata.raw.var.index
    else:
        allGeneLs = adata.var.index
    geneCounts = len(allGeneLs)

    if not batch:
        for i, gene in tqdm(enumerate(allGeneLs), "Processed Gene", geneCounts):
            sc.pl.umap(adata, layer=layer, color=gene, cmap="Reds", show=False)
            plt.savefig(f"{outputDirPath}{gene}.pdf", format="pdf")
            plt.close()
    else:
        ls_batch = adata.obs[batch].unique()
        for i, gene in tqdm(enumerate(allGeneLs), "Processed Gene", geneCounts):
            for batchName in ls_batch:
                ax = sc.pl.umap(adata, show=False)
                sc.pl.umap(
                    adata[adata.obs[batch] == batchName],
                    layer=layer,
                    color=gene,
                    cmap="Reds",
                    ax=ax,
                    size=120000 / len(adata),
                    show=False,
                )
                plt.savefig(f"{outputDirPath}{gene}_{batchName}.pdf", format="pdf")
                plt.close()

    logger.info("All finished")


def setLayerInfo(adata: anndata.AnnData, **dt_layerInfo):
    if "layerInfo" not in adata.uns:
        adata.uns["layerInfo"] = {}
    for key, value in dt_layerInfo.items():
        adata.uns["layerInfo"][key] = value