##########################################################################################
#                                     Author: Liu ZJ                                     #
#                          Last Updated: May 26, 2021 08:05 PM                           #
#                                Source Language: python                                 #
#               Repository: https://github.com/liuzj039/myPythonTools.git                #
##########################################################################################

##########################################################################################
#                                     Author: Liu ZJ                                     #
#                          Creation Date: May 26, 2021 03:29 PM                          #
#                          Last Updated: May 26, 2021 03:55 PM                           #
#                                Source Language: python                                 #
#               Repository: https://github.com/liuzj039/myPythonTools.git                #
#                                                                                        #
#                                --- Code Description ---                                #
#                                      add plotting                                      #
##########################################################################################

##########################################################################################
#                                     Author: Liu ZJ                                     #
#                         Creation Date: April 14, 2021 04:08 PM                         #
#                         Last Updated: April 14, 2021 04:08 PM                          #
#                                Source Language: python                                 #
#               Repository: https://github.com/liuzj039/myPythonTools.git                #
#                                                                                        #
#                                --- Code Description ---                                #
#                                    classify functions                                  #
##########################################################################################

##########################################################################################
#                                     Author: Liu ZJ                                     #
#                         Creation Date: March 23, 2021 11:09 AM                         #
#                         Last Updated: March 23, 2021 11:10 AM                          #
#                                Source Language: python                                 #
#               Repository: https://github.com/liuzj039/myPythonTools.git                #
#                                                                                        #
#                                --- Code Description ---                                #
#                             Tools For Single Cell Analysis                             #
##########################################################################################


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


class basic(object):
    @staticmethod
    def splitAdata(
        adata: anndata.AnnData, batchKey: str, copy=True
    ) -> Iterator[anndata.AnnData]:
        assert batchKey in adata.obs.columns, f"{batchKey} not detected in adata"
        indexName = "index" if (not adata.obs.index.name) else adata.obs.index.name
        batchObsLs = (
            adata.obs.filter([batchKey])
            .reset_index()
            .groupby(batchKey)[indexName]
            .agg(list)
        )
        for batchObs in batchObsLs:
            if copy:
                yield adata[batchObs].copy()
            else:
                yield adata[batchObs]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def getadataColor(adata, label):
        if f"{label}_colors" not in adata.uns:
            basic.setadataColor(adata, label)
        return {
            x: y
            for x, y in zip(
                adata.obs[label].cat.categories, adata.uns[f"{label}_colors"]
            )
        }

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        sceObj = setAssay(
            sceObj, "counts", value=R("as.matrix")(R.assay(sceObj, "counts"))
        )
        sceObj = setAssay(
            sceObj, "logcounts", value=R("as.matrix")(R.assay(sceObj, "logcounts"))
        )

        if isinstance(clusterNum, Sequence):
            clusterNum = np.array(clusterNum)
        else:
            clusterNum = np.array([clusterNum])  # if is int, transform it to list.

        sceObj = R.sc3(
            sceObj, ks=py2r(clusterNum), biology=biologyInfo, n_cores=threads
        )
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

    @staticmethod
    def constclustFlatLabelling(
        complist,
        obs_names: pd.Index,
        cutoff: float = 0.75,
        compNameLs: Optional[
            Union[Sequence[str], Sequence[int], Mapping[int, str]]
        ] = None,
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
        compAddOriNumLs = [
            f"{x} ({y})" for x, y in zip(compNameLs, complist._comps.index)
        ]
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

    @staticmethod
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
                    adata_hvg.var.highly_variable_nbatches
                    == (n_batches - not_n_batches)
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

    @staticmethod
    def selectCellFromObsm(
        adata: anndata.AnnData,
        xlim: Sequence[float],
        ylim: Sequence[float],
        obsmBasis: str = "X_umap",
        returnName: bool = True,
    ) -> pd.Series:
        "select cells based on location"
        locationAr = adata.obsm[obsmBasis]
        useCellBoolLs = (
            (locationAr[:, 0] > xlim[0]) & (locationAr[:, 0] < xlim[1])
        ) & ((locationAr[:, 1] > ylim[0]) & (locationAr[:, 1] < ylim[1]))
        if returnName:
            return adata[useCellBoolLs].obs.index
        else:
            return useCellBoolLs

    @staticmethod
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
                            name: name.split("-")[0]
                            for name in clean_var.columns.values
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

    @staticmethod
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

    @staticmethod
    def plotCellScatter(**plotDt):
        return plotting.plotCellScatter(plotDt)

    @staticmethod
    def plotLabelPercentageInCluster(**plotDt):
        return plotting.plotLabelPercentageInCluster(plotDt)

    @staticmethod
    def plotClusterSankey(**plotDt):
        return plotting.plotClusterSankey(plotDt)

    @staticmethod
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

    @staticmethod
    def saveAllGeneEmbedding(
        adata: anndata.AnnData,
        outputDirPath: str,
        layer: Optional[str] = None,
        useRaw: Optional[bool] = None,
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

        for i, gene in enumerate(allGeneLs):
            sc.pl.umap(adata, layer=layer, color=gene, cmap="Reds", show=False)
            plt.savefig(f"{outputDirPath}{gene}.pdf", format="pdf")
            print("\r" + f"Progress: {i} / {geneCounts}", end="", flush=True)

        logger.info("All finished")


class plotting(object):
    @staticmethod
    def plotCellScatter(
        adata, plotFeature: Sequence[str] = ["n_counts", "n_genes", "percent_ct"]
    ):
        adata.obs = adata.obs.assign(
            n_genes=(adata.X > 0).sum(1), n_counts=adata.X.sum(1)
        )
        # adata.var = adata.var.assign(n_cells=(adata.X > 0).sum(0))
        ctGene = (adata.var_names.str.startswith("ATCG")) | (
            adata.var_names.str.startswith("ATMG")
        )
        adata.obs["percent_ct"] = np.sum(adata[:, ctGene].X, axis=1) / np.sum(
            adata.X, axis=1
        )
        sc.pl.violin(adata, plotFeature, multi_panel=True, jitter=0.4)

    @staticmethod
    def plotLabelPercentageInCluster(
        adata, groupby, label, labelColor: Optional[dict] = None
    ):
        """
        根据label在adata.obs中groupby的占比绘图

        groupby:
            表明cluster。需要存在于adata.obs
        label:
            展示的占比。需要存在于adata.obs
        labelColor:
            label的颜色
        """
        if not labelColor:
            labelColor = basic.getadataColor(adata, label)

        groupbyWithLabelCountsDf = (
            adata.obs.groupby(groupby)[label]
            .apply(lambda x: x.value_counts())
            .unstack()
        )
        groupbyWithLabelCounts_CumsumPercDf = groupbyWithLabelCountsDf.pipe(
            lambda x: x.cumsum(1).div(x.sum(1), 0) * 100
        )
        legendHandleLs = []
        legendLabelLs = []
        for singleLabel in groupbyWithLabelCounts_CumsumPercDf.columns[::-1]:
            ax = sns.barplot(
                x=groupbyWithLabelCounts_CumsumPercDf.index,
                y=groupbyWithLabelCounts_CumsumPercDf[singleLabel],
                color=labelColor[singleLabel],
            )
            legendHandleLs.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc=labelColor[singleLabel], edgecolor="none"
                )
            )
            legendLabelLs.append(singleLabel)
        legendHandleLs, legendLabelLs = legendHandleLs[::-1], legendLabelLs[::-1]
        plt.legend(legendHandleLs, legendLabelLs, bbox_to_anchor=[1, 1], frameon=False)
        plt.xlabel(groupby.capitalize())
        plt.ylabel(f"Percentage")
        return ax

    @staticmethod
    def plotClusterSankey(
        adata: anndata.AnnData, clusterNameLs: Sequence[str], figsize=[5, 5]
    ):
        """
        Returns
        -------
        pyecharts.charts.basic_charts.sankey.Sankey
            Utilize Function render_notebook can get the final figure
        """
        from .otherTools import sankeyPlotByPyechart

        df = adata.obs.filter(clusterNameLs).astype(str)

        [basic.setadataColor(adata, x) for x in clusterNameLs]
        colorDictLs = [basic.getadataColor(adata, x) for x in clusterNameLs]

        sankey = sankeyPlotByPyechart(df, clusterNameLs, figsize, colorDictLs)
        return sankey

    @staticmethod
    def plotSC3sConsensusMatrix(
        adata: anndata.AnnData,
        matrixLabel: str,
        clusterResultLs: Sequence[str],
        cmap="Reds",
        metrix="cosine",
        row_cluster=True,
        **clustermapParamsDt: Dict,
    ):
        import sys
        from .otherTools import addColorLegendToAx

        sys.setrecursionlimit(100000)

        matrixLabel = matrixLabel.rstrip("_consensus") + "_consensus"

        colorDt = adata.obs.filter(clusterResultLs)
        for clusterName in clusterResultLs:
            basic.setadataColor(adata, clusterName)
            clusterColorMapDt = basic.getadataColor(adata, clusterName)
            colorDt[clusterName] = colorDt[clusterName].map(clusterColorMapDt)

        cellIndexOrderSq = adata.obs.sort_values(matrixLabel.rstrip("_consensus")).index
        consensusMatrix = pd.DataFrame(
            adata.obsm[matrixLabel], index=adata.obs.index
        ).reindex(cellIndexOrderSq)

        g = sns.clustermap(
            consensusMatrix,
            cmap=cmap,
            metric=metrix,
            row_colors=colorDt,
            row_cluster=row_cluster,
            cbar_pos=None,
            **clustermapParamsDt,
        )

        currentYPos = 1
        currentXPos = 1.05
        interval = 0.25
        for clusterName in clusterResultLs:

            clusterColorMapDt = basic.getadataColor(adata, clusterName)
            length = 0.04 * (len(clusterColorMapDt) + 1)
            if (currentYPos == 1) or (currentYPos - length > 0):
                bbox_to_anchor = [currentXPos, currentYPos]

            else:
                currentXPos = currentXPos + interval
                currentYPos = 1
                bbox_to_anchor = [currentXPos, currentYPos]

            currentYPos = currentYPos - length
            addColorLegendToAx(
                g.ax_heatmap,
                clusterName,
                clusterColorMapDt,
                1,
                bbox_to_anchor=bbox_to_anchor,
                loc="upper left",
                # bbox_transform=plt.gcf().transFigure,
            )

        plt.xticks([])
        plt.yticks([])

        sys.setrecursionlimit(20000)

        return g

    @staticmethod
    def maskGeneExpNotInSpecialCluster(
        adata: anndata.AnnData,
        obsKey: str,
        clusterNameLs: Sequence[str],
        layer: Optional[str] = None,
        embedding: str = "X_umap",
    ) -> anndata.AnnData:
        """
        all expression value of cell which not belongs to <clusterNameLs> is equal to 0
        """
        import scipy.sparse as ss

        tempAd = basic.getPartialLayersAdata(adata, layer, [obsKey]).copy()
        tempAd.obsm[embedding] = adata.obsm[embedding]

        inClusterBoolLs = tempAd.obs[obsKey].isin(clusterNameLs)

        tempAd.layers[layer] = tempAd.X.A if ss.issparse(tempAd.X) else tempAd.X
        tempAd.layers[layer][~inClusterBoolLs, :] = 0
        return tempAd


class multiModle(object):
    @staticmethod
    def addDfToObsm(adata, copy=False, **dataDt):
        """addDfToObsm, add data to adata.obsm

        Args:
            adata ([anndata])
            copy (bool, optional)
            dataDt: {label: dataframe}, dataframe must have the same dimension

        Returns:
            adata if copy=True, otherwise None
        """
        adata = adata.copy() if copy else adata
        for label, df in dataDt.items():
            if (adata.obs.index != df.index).all():
                logger.error(f"dataset {label} have a wrong shape/index")
                0 / 0
            if label in adata.obsm:
                logger.warning(f"dataset {label} existed! Overwrite")
            adata.uns[f"{label}_label"] = df.columns.values
            adata.obsm[label] = df.values
        if copy:
            return adata

    @staticmethod
    def getMatFromObsm(
        adata: anndata.AnnData,
        keyword: str,
        minCell: int = 5,
        useGeneLs: Union[list, pd.Series, np.ndarray] = [],
        normalize=True,
        logScale=True,
        ignoreN=False,
        clear=False,
        raw=False,
        strCommand=None,
    ) -> anndata.AnnData:
        """
        use MAT deposited in obsm replace the X MAT

        params:
            adata:
                version 1.0 multiAd
            keyword:
                stored in obsm
            minCell:
                filter feature which expressed not more than <minCell> cells
            useGeneLs:
                if not specified useGeneLs, all features will be output, otherwise only features associated with those gene will be output
            normalize:
                normalize the obtained Mtx or not
            logScale:
                log-transformed or not
            ignoreN:
                ignore ambiguous APA/Splice info
            clear:
                data not stored in obs or var will be removed
            raw:
                return the raw dataset stored in the obsm. This parameter is prior to all others
            strCommand:
                use str instead of specified params:
                "n": set normalize True
                "s": set logScale True
                'N': set ignoreN True
                'c' set clear True
                '': means all is False
                This parameter is prior to all others except raw
        return:
            anndata
        """
        if clear:
            transformedAd = anndata.AnnData(
                X=adata.obsm[keyword].copy(),
                obs=adata.obs,
                var=pd.DataFrame(index=adata.uns[f"{keyword}_label"]),
            )
        else:
            transformedAd = anndata.AnnData(
                X=adata.obsm[keyword].copy(),
                obs=adata.obs,
                var=pd.DataFrame(index=adata.uns[f"{keyword}_label"]),
                obsp=adata.obsp,
                obsm=adata.obsm,
                uns=adata.uns,
            )

        if raw:
            return transformedAd

        if strCommand != None:
            normalize = True if "n" in strCommand else False
            logScale = True if "s" in strCommand else False
            ignoreN = True if "N" in strCommand else False
            clear = True if "c" in strCommand else False

        logger.info(
            f"""
        final mode: 
            normalize: {normalize}, 
            logScale: {logScale}, 
            ignoreN: {ignoreN}, 
            clear: {clear}
        """
        )

        sc.pp.filter_genes(transformedAd, min_cells=minCell)

        if normalize:
            sc.pp.normalize_total(transformedAd, target_sum=1e4)
        if logScale:
            sc.pp.log1p(transformedAd)

        useGeneLs = list(useGeneLs)
        if not useGeneLs:
            transformedAd = transformedAd
        else:
            transformedAdFeatureSr = transformedAd.var.index
            transformedAdFeatureFilterBl = (
                transformedAdFeatureSr.str.split("_").str[0].isin(useGeneLs)
            )
            transformedAd = transformedAd[:, transformedAdFeatureFilterBl]

        if ignoreN:
            transformedAdFeatureSr = transformedAd.var.index
            transformedAdFeatureFilterBl = (
                ~transformedAdFeatureSr.str.split("_").str[1].isin(["N", "Ambiguous"])
            )

            transformedAd = transformedAd[:, transformedAdFeatureFilterBl]

        return transformedAd

    @staticmethod
    def transformEntToAd(ent) -> anndata.AnnData:
        """
        parse mofa
        transformEntToAd parse trained ent object to anndata

        Args:
            ent ([entry_point]): only one group

        Returns:
            anndata: the X represents the sample-factor weights,
                    the layer represents the feature-factor weight and variance-factor matrix,
                    the uns['mofaR2_total] stored the total variance of factors could be explained
        """
        factorOrderLs = np.argsort(
            np.array(ent.model.calculate_variance_explained()).sum(axis=(0, 1))
        )[::-1]

        sampleWeightDf = pd.DataFrame(ent.model.getExpectations()["Z"]["E"]).T
        sampleWeightDf = sampleWeightDf.reindex(factorOrderLs).reset_index(drop=True)
        sampleWeightDf.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
        sampleWeightDf.columns = ent.data_opts["samples_names"][0]
        mofaAd = basic.creatAnndataFromDf(sampleWeightDf)

        for label, featureSr, data in zip(
            ent.data_opts["views_names"],
            ent.data_opts["features_names"],
            ent.model.getExpectations()["W"],
        ):
            df = pd.DataFrame(data["E"]).T
            featureSr = pd.Series(featureSr)
            featureSr = featureSr.str.rstrip(label)
            if label in ["APA", "fullySpliced"]:
                featureSr = featureSr + label
            df.columns = featureSr
            df = df.reindex(factorOrderLs).reset_index(drop=True)
            df.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
            addDfToObsm(mofaAd, **{label: df})

        r2Df = pd.DataFrame(ent.model.calculate_variance_explained()[0]).T
        r2Df = r2Df.reindex(factorOrderLs).reset_index(drop=True)
        r2Df.index = [f"factor_{x}" for x in range(1, len(factorOrderLs) + 1)]
        r2Df.columns = ent.data_opts["views_names"]
        addDfToObsm(mofaAd, mofaR2=r2Df)

        mofaAd.uns["mofaR2_total"] = {
            x: y
            for x, y in zip(
                ent.data_opts["views_names"],
                ent.model.calculate_variance_explained(True)[0],
            )
        }
        return mofaAd


class normalize(object):
    @staticmethod
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

    @staticmethod
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
        from .rTools import py2r, r2py, r_inline_plot
        from scipy.sparse import csr_matrix, isspmatrix

        R = ro.r
        importr("scran")
        logger.info("Initialization")
        adata = adata.copy() if copy else adata

        if not clusterInfo:
            adataPP = basic.getPartialLayersAdata(adata, layer)

            if needNormalizePre:
                basic.testAllCountIsInt(adataPP, layer)
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

        rawMtx = adata.X if not layer else adata.layers[layer]
        dataMat_r = py2r(rawMtx.A.T) if isspmatrix(rawMtx) else py2r(rawMtx.T)

        logger.info("calculate size factor")
        sizeFactorSr_r = R.sizeFactors(
            R.computeSumFactors(
                R.SingleCellExperiment(R.list(counts=dataMat_r)),
                clusters=inputGroupDf_r,
                **{"min.mean": 0.1},
            )
        )
        sizeFactorSr = r2py(sizeFactorSr_r).copy()

        logger.info("process result")
        adata.obs["sizeFactors"] = sizeFactorSr
        adata.layers["scran"] = rawMtx / adata.obs["sizeFactors"].values.reshape(
            [-1, 1]
        )
        if logScaleOut:
            logger.warning("output is logScaled")
            sc.pp.log1p(adata, layer="scran")

        return adata if copy else None

    @staticmethod
    def normalizeBySCT(
        adata: anndata.AnnData,
        layer: Union[Literal["X"], str] = "X",
        regress_out: Sequence = ("log_umi",),
        method="poisson",
        batch_key: Optional[str] = None,
        n_top_genes: int = 3000,
        regress_out_nonreg: Optional[Sequence] = None,
        min_cells: int = 5,
        store_residuals: bool = True,
        correct_counts: bool = True,
        log_scale_correct: bool = False,
        verbose: bool = True,
        inplace: bool = True,
        seed: int = 0,
    ) -> Optional[anndata.AnnData]:
        """\
        Forked from gokceneraslan
        Normalization and variance stabilization of scRNA-seq data using regularized
        negative binomial regression [Hafemeister19]_.
        sctransform uses Pearson residuals from regularized negative binomial regression to
        correct for the sequencing depth. After regressing out total number of UMIs (and other
        variables if given) it ranks the genes based on their residual variances and therefore
        also acts as a HVG selection method.
        This function replaces `sc.pp.normalize_total` and `sc.pp.highly_variable_genes` and requires
        raw counts in `adata.X`.
        .. note::
            More information and bug reports `here <https://github.com/ChristophH/sctransform>`__.
        Parameters
        ----------
        adata
            An anndata file with `X` attribute of unnormalized count data
        layer
            which layer is used as input matrix for SCT
        regress_out
            Variables to regress out. Default is logarithmized total UMIs which is implicitly
            calculated by sctransform. Other obs keys can also be used.
        batch_key
            If specified, HVGs are ranked after batch_key is regressed out. This avoids the
            selection of batch-specific genes and acts as a lightweight batch correction method.
            Note that corrected counts are not batch-corrected but only depth-normalized.
        n_top_genes
            Total number of highly variable genes selected.
        min_cells
            Only use genes that have been detected in at least this many cells; default is 5.
        store_residuals
            Store Pearson residuals in adata.layers['sct_residuals']. These values represent
            batch corrected and depth-normalized gene expression values. Due to potential
            high memory use for big matrices, they are not stored by default.
        correct_counts
            Store corrected counts in adata.layers['sct_corrected']. Default is True.
        log_scale_correct
            Default is False
        verbose
            Show progress bar during normalization.
        inplace
            Save HVGs and corrected UMIs inplace. Default is True.
        seed
            Random seed for R RNG. Default is 0.
        Returns
        -------
        If `inplace` is False, anndata is returned.
        If `store_residuals` is True, residuals are stored in adata.layers['sct_residuals'].
        `adata.layers['sct_corrected']` stores normalized representation of gene expression.
        `adata.var['highly_variable']` stores highly variable genes.
        `adata.var['highly_variable_sct_residual_var']` stores the residual variances that
        are also used for ranking genes by variability.
        """

        import rpy2
        from rpy2.robjects import r
        from rpy2.robjects.packages import importr
        from scanpy.preprocessing import filter_genes
        import scipy.sparse as sp
        from .rTools import (
            py2r,
            r2py,
            r_is_installed,
            r_set_seed,
        )

        r_is_installed("sctransform")
        r_set_seed(seed)

        # check if observations are unnormalized using first 10
        testColCounts = min([10, adata.shape[0]])
        if layer == "X":
            X_subset = adata.X[:testColCounts]
        else:
            X_subset = adata.layers[layer][:testColCounts]
        err = "Make sure that adata.X contains unnormalized count data"
        if sp.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, err
        else:
            assert np.all(X_subset.astype(int) == X_subset), err

        assert regress_out, "regress_out cannot be emtpy"

        if not inplace:
            adata = adata.copy()

        filter_genes(adata, min_cells=min_cells)

        mat = adata.X.T if layer == "X" else adata.layers[layer].T
        if sp.issparse(mat):
            mat.sort_indices()
        mat = py2r(mat)

        set_colnames = r("`colnames<-`")
        set_rownames = r("`rownames<-`")

        mat = set_colnames(mat, adata.obs_names.values.tolist())
        mat = set_rownames(mat, adata.var_names.values.tolist())

        assert isinstance(
            regress_out, collections.abc.Sequence
        ), "regress_out is not a sequence"

        obs_keys = [x for x in regress_out if x != "log_umi"]
        regress_out = py2r(np.array(regress_out))
        if regress_out_nonreg is not None:
            assert isinstance(
                regress_out_nonreg, collections.abc.Sequence
            ), "regress_out_nonreg is not a sequence"

            obs_keys += list(regress_out_nonreg)
            regress_out_nonreg = py2r(np.array(regress_out_nonreg))
        else:
            regress_out_nonreg = rpy2.rinterface.NULL

        if batch_key is not None:
            obs_keys += [batch_key]
        else:
            batch_key = rpy2.rinterface.NULL

        if obs_keys:
            assert np.all(
                np.isin(obs_keys, adata.obs.columns)
            ), "Some regress_out or batch_key values are not found in adata.obs"
            cell_attr = adata.obs[obs_keys]
            cell_attr = py2r(cell_attr)
        else:
            cell_attr = rpy2.rinterface.NULL

        sct = importr("sctransform")
        residual_type = "pearson" if store_residuals else "none"

        vst_out = sct.vst(
            mat,
            cell_attr=cell_attr,
            batch_var=batch_key,
            latent_var=regress_out,
            latent_var_nonreg=regress_out_nonreg,
            residual_type=residual_type,
            return_cell_attr=True,
            min_cells=min_cells,
            method=method,
            n_genes=n_top_genes,
            show_progress=verbose,
        )

        res_var = r2py(sct.get_residual_var(vst_out, mat))

        if correct_counts:
            corrected = r2py(sct.correct_counts(vst_out, mat)).T
            adata.layers["sct_corrected"] = corrected.copy()
            if log_scale_correct:
                sc.pp.log1p(adata, layer="sct_corrected")
                logger.warning("sct_corrected layer IS log-scaled")
            else:
                logger.warning("sct_corrected layer is NOT log-scaled")

        adata.var["highly_variable_sct_residual_var"] = res_var.copy()

        if store_residuals:
            adata.layers["sct_residuals"] = r2py(vst_out.rx2("y")).T.copy()

        top_genes = (
            adata.var["highly_variable_sct_residual_var"]
            .sort_values(ascending=False)[:n_top_genes]
            .index.tolist()
        )
        adata.var["highly_variable"] = adata.var_names.isin(top_genes)

        if not inplace:
            return adata


class geneEnrichInfo(object):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            shuffleAd.obs[label] = (
                adata.obs[label].sample(frac=1, random_state=i).values
            )
            shuffleClusterDf = (
                mergeadataExpress(shuffleAd, label)
                .to_df()
                .reset_index()
                .assign(label=i)
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
            pd.concat(allShuffleClusterExpressLs)
            .set_index("label")
            .reindex(geneLs, axis=1)
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

    @staticmethod
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

    @staticmethod
    def getEnrichedGeneByCellId(
        adata: anndata.AnnData,
        layer: Optional[str] = None,
        clusterName: str = "leiden",
        n_features: int = 200,
        nmcs: int = 50,
        copy: bool = False,
        returnR: bool = False,
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
        from .rTools import py2r, r2py

        rBase = importr("base")
        cellId = importr("CelliD")
        R = ro.r
        adataR = py2r(basic.getPartialLayersAdata(adata, [layer], [clusterName]))
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
                rBase.as_data_frame(
                    R.attr(R.reducedDim(adataR, "MCA"), "genesCoordinates")
                )
            ).reindex(adata.var.index, fill_value=0)
            adata.uns[f"{clusterName}_cellid_marker"] = df_marker

    @staticmethod
    def getMarkerByFcCellexCellidDiffxpy(
        adata: anndata.AnnData,
        layer: str,
        groupby: str,
        groups: List[str] = None,
        forceAllRun: bool = False,
        dt_ByFcParams={},
        dt_DiffxpyParams={},
        dt_DiffxpyGetMarkerParams={"detectedCounts": -2},
        cutoff_cellex: float = 0.9,
        markerCounts_CellId: int = 50,
    ):
        """
        use three method to identify markers

        Parameters
        ----------
        adata : anndata.AnnData
        layer : str
            must be log-transformed
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
        ad_sub.layers[f"{layer}_raw"] = (
            np.around(np.exp(ad_sub.layers[layer].A) - 1)
            if ss.issparse(ad_sub.layers[layer])
            else np.around(np.exp(ad_sub.layers[layer]) - 1)
        )
        if forceAllRun | (f"{groupby}_fcMarker" not in ad_sub.uns):
            geneEnrichInfo.detectMarkerGene(
                ad_sub,
                groupby,
                f"{groupby}_fcMarker",
                layer=layer,
                **dt_ByFcParams,
            )
            adata.uns[f"{groupby}_fcMarker"] = ad_sub.uns[f"{groupby}_fcMarker"]
            adata.uns[f"{groupby}_fcMarker_filtered"] = ad_sub.uns[
                f"{groupby}_fcMarker_filtered"
            ]
        dt_markerByFc = (
            sc.get.rank_genes_groups_df(
                ad_sub, None, key=f"{groupby}_fcMarker_filtered"
            )
            .groupby("group")["names"]
            .agg(list)
            .to_dict()
        )
        adata.uns[f"marker_multiMethod_{groupby}"]["fcMarker"] = dt_markerByFc

        if forceAllRun | (f"{groupby}_cellexES" not in ad_sub.varm):
            geneEnrichInfo.calculateEnrichScoreByCellex(ad_sub, f"{layer}_raw", groupby)
            adata.varm[f"{groupby}_cellexES"] = ad_sub.varm[f"{groupby}_cellexES"]
        dt_marker_cellex = (
            ad_sub.varm[f"{groupby}_cellexES"]
            .apply(
                lambda x: list(x[x > cutoff_cellex].sort_values(ascending=False).index)
            )
            .to_dict()
        )
        adata.uns[f"marker_multiMethod_{groupby}"]["cellexMarker"] = dt_marker_cellex

        if forceAllRun | (f"{groupby}_cellid_marker" not in adata.uns):
            geneEnrichInfo.getEnrichedGeneByCellId(
                ad_sub, layer, groupby, markerCounts_CellId
            )
            adata.uns[f"{groupby}_cellid_marker"] = ad_sub.uns[
                f"{groupby}_cellid_marker"
            ]
        dt_markerCellId = {
            x: list(y)
            for x, y in ad_sub.uns[f"{groupby}_cellid_marker"].to_dict("series").items()
        }

        adata.uns[f"marker_multiMethod_{groupby}"]["cellidMarker"] = dt_markerCellId

        if forceAllRun | (f"{groupby}_diffxpy_marker" not in adata.uns):
            diffxpy.pairWise(
                ad_sub,
                layer,
                groupby,
                keyAdded=f"{groupby}_diffxpy_marker",
                **dt_DiffxpyParams,
            )
            adata.uns[f"{groupby}_diffxpy_marker"] = ad_sub.uns[
                f"{groupby}_diffxpy_marker"
            ]

        df_diffxpyMarker = diffxpy.getMarker(
            adata, key=f"{groupby}_diffxpy_marker", **dt_DiffxpyGetMarkerParams
        )
        adata.uns[f"marker_multiMethod_{groupby}"]["diffxpyMarker"] = (
            df_diffxpyMarker.groupby("testedCluster")["gene"].agg(list).to_dict()
        )

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
                        adata.uns[f"marker_multiMethod_{groupby}"]["cellexMarker"][
                            cluster
                        ]
                    ),
                    cellidMarker=df["marker"].isin(
                        adata.uns[f"marker_multiMethod_{groupby}"]["cellidMarker"][
                            cluster
                        ]
                    ),
                    diffxpyMarker=df["marker"].isin(
                        adata.uns[f"marker_multiMethod_{groupby}"]["diffxpyMarker"][
                            cluster
                        ]
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


class annotation(object):
    @staticmethod
    def cellTypeAnnoByICI(
        adata,
        specDfPath,
        cutoff_adjPvalue,
        layer=None,
        copy=False,
        threads=1,
        logTransformed=True,
        addName="ICI",
    ):
        """
        use ICI method annotate cell type

        adata:
            anndata, normalized but not log-transformed.
        specDfPath:
            ICITools::compute_spec_table result
        """
        from rpy2.robjects import pandas2ri
        import rpy2.robjects as ro

        pandas2ri.activate()

        adata = adata.copy() if copy else adata

        adataDf = adata.to_df(layer).T.rename_axis("Locus")
        if logTransformed:
            adataDf = np.exp(adataDf) - 1
        logger.info("Start transfer dataframe to R")
        ro.globalenv["adataDf"] = adataDf
        if threads > 1:
            ro.r(
                f"""
            future::plan(strategy = "multiprocess", workers = {threads})
            1
            """
            )

        logger.info("Start calculate ICI score")
        ro.r(
            f"""
        specDf <- readRDS('{specDfPath}')
        adataDf <- tibble::as_tibble(adataDf, rownames = 'Locus')
        adataAnnoDf_melted <- ICITools::compute_ici_scores(expression_data = adataDf,
                                            spec_table = specDf,
                                            min_spec_score = 0.15,
                                            information_level = 50, sig = TRUE)
        1
        """
        )
        logger.info("Start transfer ICI score result to python")
        adataAnnoDf_wide = ro.globalenv["adataAnnoDf_melted"]

        logger.info("Start parse ICI score result")
        adata.obsm[f"{addName}-adjPvalue"] = adataAnnoDf_wide.pivot_table(
            "p_adj", "Cell", "Cell_Type"
        ).reindex(adata.obs_names)
        adata.obsm[f"{addName}-score"] = adataAnnoDf_wide.pivot_table(
            "ici_score", "Cell", "Cell_Type"
        ).reindex(adata.obs_names)

        adata.obsm[f"{addName}-score_masked"] = pd.DataFrame(
            np.select(
                [adata.obsm[f"{addName}-adjPvalue"] < cutoff_adjPvalue],
                [adata.obsm[f"{addName}-score"]],
                0,
            ),
            index=adata.obsm[f"{addName}-score"].index,
            columns=adata.obsm[f"{addName}-score"].columns,
        )

        adata.obs[f"{addName}-anno"] = np.select(
            [adata.obsm[f"{addName}-score_masked"].max(1) > 0],
            [adata.obsm[f"{addName}-score_masked"].idxmax(1)],
            "Unknown",
        )

        if copy:
            return adata

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
                logger.warning(
                    f"Specific genes dont have any overlap with cell type <{x}>"
                )
                delKeyLs.append(x)
        [markerDt.pop(x) for x in delKeyLs]

        return markerDt

    def labelTransferByCellId(
        refAd: anndata.AnnData,
        refLabel: str,
        refLayer: str,
        queryAd: anndata.AnnData,
        queryLayer: str,
        markerCount: int = 200,
        cutoff: float = 2.0,
        nmcs: int = 50,
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
            must be log-transformed
        queryAd : anndata.AnnData
        queryLayer : str
            must be log-transformed
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
        from .rTools import py2r, r2py, r_inline_plot, rHelp

        rBase = importr("base")
        rUtils = importr("utils")
        cellId = importr("CelliD")
        R = ro.r
        VectorR_Refmarker = geneEnrichInfo.getEnrichedGeneByCellId(
            refAd, refLayer, refLabel, markerCount, copy=True, returnR=True, nmcs=nmcs
        )

        queryAd = queryAd.copy() if copy else queryAd
        adR_query = py2r(basic.getPartialLayersAdata(queryAd, [queryLayer]))
        adR_query = cellId.RunMCA(adR_query, slot=queryLayer, nmcs=nmcs)
        df_labelTransfered = r2py(
            rBase.data_frame(
                cellId.RunCellHGT(
                    adR_query, VectorR_Refmarker, dims=py2r(np.arange(1, 1 + nmcs))
                ),
                check_names=False,
            )
        ).T
        queryAd.obsm[f"cellid_{refLabel}_labelTranferScore"] = df_labelTransfered
        queryAd.obs[f"cellid_{refLabel}_labelTranfer"] = queryAd.obsm[
            f"cellid_{refLabel}_labelTranferScore"
        ].pipe(lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown"))
        if copy:
            return queryAd


class deprecated(object):
    @staticmethod
    def cellTypeAnnoByCorr(
        adata,
        bulkExpressionDf,
        threads=1,
        method="pearsonr",
        reportFinalUseGeneCounts=False,
        geneCountsCutoff=0,
        logTransformed=True,
        returnR=False,
        keepZero=True,
        useRaw=True,
        reportCounts=50,
    ):
        """
        通过bulk数据的相关性鉴定细胞类型

        adata: log-transformed adata

        bulkExpressionDf:
                                                    AT1G01010  AT1G01030  AT1G01040
            bending cotyledon : chalazal endosperm   3.018853   2.430005   8.284994
            bending cotyledon : chalazal seed coat   2.385562   2.364294   8.674318
            bending cotyledon : embryo proper        2.258559   2.249158   7.577717

        returnR:
            返回最大值还是所有结果的r

        method:
            'pearsonr' | 'spearmanr'

        geneCountsCutoff:
            CPM
        """

        def __getSpearmanRForCell(cellExpressionSr):
            nonlocal i, bulkExpressionDf, keepZero, reportCounts, threads, method, geneCountsCutoff, reportFinalUseGeneCounts
            if not keepZero:
                cellExpressionSr = cellExpressionSr.pipe(lambda x: x[x != 0])
            cellExpressionSr = cellExpressionSr.pipe(lambda x: x[x >= geneCountsCutoff])
            #         print(cellExpressionSr)
            useGeneLs = cellExpressionSr.index
            bulkExpressionDf = bulkExpressionDf.reindex(useGeneLs, axis=1).dropna(
                axis=1
            )
            useGeneLs = bulkExpressionDf.columns
            cellExpressionSr = cellExpressionSr.reindex(useGeneLs)

            if reportFinalUseGeneCounts:
                logger.info(len(cellExpressionSr))

            if len(cellExpressionSr) <= 1:
                return None

            if method == "spearmanr":
                bulkExpressionCorrDf = bulkExpressionDf.apply(
                    lambda x: spearmanr(cellExpressionSr, x)[0], axis=1
                )
            elif method == "pearsonr":
                bulkExpressionCorrDf = bulkExpressionDf.apply(
                    lambda x: pearsonr(cellExpressionSr, x)[0], axis=1
                )
            else:
                logger.error("Unrecognized method")
                1 / 0

            i += 1
            if i % reportCounts == 0:
                logger.info(f"{i * threads} / {cellCounts} processed")

            if not returnR:
                mostSimilarBulk = bulkExpressionCorrDf.idxmax()
                return mostSimilarBulk
            else:
                return bulkExpressionCorrDf

        i = 0
        adata = adata.copy()
        cellCounts = len(adata)
        geneCountsCutoff = np.log(geneCountsCutoff + 1)

        adataExpressDf = (
            pd.DataFrame(
                adata.raw.X.A, columns=adata.raw.var.index, index=adata.obs.index
            )
            if useRaw
            else adata.to_df()
        )
        adataExpressDf = (
            np.exp(adataExpressDf) - 1 if logTransformed else adataExpressDf
        )
        adataExpressDf = adataExpressDf.div(adataExpressDf.sum(1), axis=0) * 1000000
        adataExpressDf = (
            np.log(adataExpressDf + 1) if logTransformed else adataExpressDf
        )
        #     print(adataExpressDf)

        if threads == 1:
            cellAnnotatedType = adataExpressDf.apply(__getSpearmanRForCell, axis=1)
        else:
            from pandarallel import pandarallel

            pandarallel.initialize(nb_workers=threads)
            cellAnnotatedType = adataExpressDf.parallel_apply(
                __getSpearmanRForCell, axis=1
            )
        return cellAnnotatedType

    @staticmethod
    def cellTypeAnnoByMarker(adata, allMarkerUse, label="louvain", method="mean"):
        """
        通过marker基因表达量鉴定细胞类型

        adata:
            adata.obs中有louvain
            通过adata.raw.var来判断哪些基因表达
            存在raw, log-transformed

        allMarkerUse:
            {
                'Columella root cap':
                    ['AT4G27400','AT3G18250', 'AT5G20045']
            }

        method = mean|median

        return:
            df: markerExpressCount(not logtransformed), expressRatio
        """
        #     import pdb; pdb.set_trace()
        markerRevDt = {z: x for x, y in allMarkerUse.items() for z in y}
        rawCountMtx = np.exp(adata.raw.to_adata().to_df()) - 1
        rawCountMtxWithLabel = rawCountMtx.join(adata.obs[label])

        clusterExMtx = rawCountMtxWithLabel.groupby(label).agg(method)
        #     return clusterExMtx.T
        clusterExMtxTr = clusterExMtx.T
        clusterExMtxTr.columns = clusterExMtxTr.columns.astype("str")
        clusterExMtxTr = clusterExMtxTr.assign(
            cellType=lambda df: df.index.map(lambda x: markerRevDt.get(x, "Others"))
        )
        clusterTypeExpMtx = clusterExMtxTr.groupby("cellType").agg(method).T

        cellExRatioMtxTr = rawCountMtx.applymap(lambda x: 1 if x > 0 else 0).T
        cellExRatioMtxTr.columns = cellExRatioMtxTr.columns.astype("str")
        cellExRatioMtxTr = cellExRatioMtxTr.assign(
            cellType=lambda df: df.index.map(lambda x: markerRevDt.get(x, "Others"))
        )
        cellExRatioMtx = (
            cellExRatioMtxTr.groupby("cellType").apply(lambda x: x.mean(0)).T
        )
        cellExRatioMtxWithLabel = cellExRatioMtx.join(adata.obs[label])
        clusterExRatioMtx = cellExRatioMtxWithLabel.groupby(label).agg(method)

        finalMtx = (
            pd.concat(
                [clusterTypeExpMtx.unstack(), clusterExRatioMtx.unstack()], axis=1
            )
            .reset_index()
            .rename({0: "express", 1: "ratio", "level_0": "cellType"}, axis=1)
        ).query("cellType != 'Others'")

        return finalMtx

    @staticmethod
    def cellTypeAnnoByMarkerOld(
        adata, allMarkerUse, expressionMtx, zscoreby="cluster", method="mean"
    ):
        """
        通过marker基因表达量鉴定细胞类型

        adata:
            adata.obs中有louvain
            通过adata.raw.var来判断哪些基因表达

        allMarkerUse:
        {
        'Zhang et al.':
            {
                'Columella root cap':
                    ['AT4G27400','AT3G18250', 'AT5G20045']
            }
        }
        expressionMtx:
            由adata.to_df()获得:
                没有log-transformed
                没有筛选基因
                经过normalize_sum

        zscoreby = cluster|cell

        method = mean|median
        """

        def _getMarkerExpressionGene(adata, allMarkerUse):
            """
            去除marker中不表达的基因
            adata 存在raw
            allMarkerUse
                {'Zhang et al.': {'Columella root cap': ['AT4G27400','AT3G18250', 'AT5G20045']}}
            """
            expressionGene = set(adata.raw.var.index)

            integrateMarkerGene = {}
            for x, y in allMarkerUse.items():
                singleMarkerGeneUse = {}
                for j, k in y.items():
                    k = list(set(k) & expressionGene)
                    singleMarkerGeneUse[j] = k
                integrateMarkerGene[x] = singleMarkerGeneUse
            return integrateMarkerGene

        expressionMtx = expressionMtx.copy(True)
        #     expressionMtx = np.log2(expressionMtx + 1)

        allMarkers = _getMarkerExpressionGene(adata, allMarkerUse)

        expressionMtx = expressionMtx.join(adata.obs["louvain"], how="inner")
        allLouvain = expressionMtx["louvain"].unique()
        expressionCounts = (
            expressionMtx.groupby("louvain")
            .apply(lambda x: x.drop("louvain", axis=1).pipe(lambda y: y.sum() / len(y)))
            .fillna(0)
        )
        expressionCounts = np.log2(expressionCounts + 1)
        expressionSizes = (
            expressionMtx.groupby("louvain")
            .apply(
                lambda x: x.drop("louvain", axis=1).pipe(
                    lambda y: (y > 0).sum() / len(y)
                )
            )
            .fillna(0)
        )
        if zscoreby == "cluster":
            expressionZscore = expressionCounts.apply(zscore)
        elif zscoreby == "cell":
            expressionMtx = np.log2(expressionMtx.drop("louvain", axis=1) + 1)
            expressionMtx = expressionMtx.apply(zscore)
            expressionZscore = (
                expressionMtx.join(adata.obs["louvain"], how="inner")
                .groupby("louvain")
                .apply(
                    lambda x: x.drop("louvain", axis=1).pipe(lambda y: y.sum() / len(y))
                )
                .fillna(0)
            )
        #     expressionCounts = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: y.sum() / len(y))).fillna(0)
        #     expressionCounts = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: y.sum() / (y > 0).sum())).fillna(0)

        groupAllClustersExpressionCounts = []
        groupAllClustersExpressionZscore = []
        groupAllClustersExpressionSizes = []
        groupNames = []
        for stage, tissueGenes in allMarkers.items():
            for tissue, genes in tissueGenes.items():
                if method == "mean":
                    groupGeneCountsDf = expressionCounts.loc[:, genes].mean(1)
                    groupGeneZscoreDf = expressionZscore.loc[:, genes].mean(1)
                    groupGeneSizesDf = expressionSizes.loc[:, genes].mean(1)
                elif method == "median":
                    groupGeneCountsDf = expressionCounts.loc[:, genes].median(1)
                    groupGeneZscoreDf = expressionZscore.loc[:, genes].median(1)
                    groupGeneSizesDf = expressionSizes.loc[:, genes].median(1)
                groupGeneCountsDf.name = f"{stage} {tissue}"
                groupGeneZscoreDf.name = f"{stage} {tissue}"
                groupGeneSizesDf.name = f"{stage} {tissue}"
                groupAllClustersExpressionCounts.append(groupGeneCountsDf)
                groupAllClustersExpressionZscore.append(groupGeneZscoreDf)
                groupAllClustersExpressionSizes.append(groupGeneSizesDf)

        groupAllClustersExpressionCounts = pd.concat(
            groupAllClustersExpressionCounts, 1
        )
        groupAllClustersExpressionZscore = pd.concat(
            groupAllClustersExpressionZscore, 1
        )
        groupAllClustersExpressionSizes = pd.concat(groupAllClustersExpressionSizes, 1)
        groupAllClustersExpression = pd.concat(
            [
                groupAllClustersExpressionSizes.stack(),
                groupAllClustersExpressionZscore.stack(),
                groupAllClustersExpressionCounts.stack(),
            ],
            axis=1,
        )
        groupAllClustersExpression.reset_index(inplace=True)
        groupAllClustersExpression.columns = [
            "louvain",
            "tissues",
            "Percentage of expressed nuclei",
            "Z-score of Expression",
            "Average expression",
        ]
        #     groupAllClustersExpression = groupAllClustersExpression.reset_index()
        #     groupAllClustersExpression.columns = ['louvain','tissues','Percentage of expressed nuclei', 'Average expression']
        return groupAllClustersExpression

    @staticmethod
    def cellTypeAnnoByClusterEnriched(
        arrayExpressDf_StageTissue,
        clusterEnrichedGeneDf,
        useCluster="all",
        useGeneCounts=10,
    ):
        """
        使用cluster enriched基因在bulk数据中的表达情况对cluster进行注释
        暂时只能用在胚乳数据上 待进一步优化

        arrayExpressDf_StageTissue: dataframe, 形如
                                                AT1G01010  AT1G01030  AT1G01040
        stage             correctedTissue
        bending cotyledon chalazal endosperm     3.018853   2.430005   8.284994
                        chalazal seed coat     2.385562   2.364294   8.674318
                        embryo proper          2.258559   2.249158   7.577717
                        general seed coat      2.000998   2.168115   7.721052
                        peripheral endosperm   2.503232   2.154924   8.002944

        clusterEnrichedGeneDf: getClusterEnrichedGene输出
        useCluster：'all'|['1', '2']
        useGeneCounts: 每个cluster使用的基因数
        """
        stageOrderLs = [
            "pre-globular",
            "globular",
            "heart",
            "linear-cotyledon",
            "bending cotyledon",
            "mature green",
        ]
        tissueOrderLs = [
            "chalazal endosperm",
            "micropylar endosperm",
            "peripheral endosperm",
            "chalazal seed coat",
            "general seed coat",
            "embryo proper",
            "suspensor",
        ]
        expressList = list(arrayExpressDf_StageTissue.columns)
        clusterEnrichedGene_FilteredDf = (
            clusterEnrichedGeneDf.sort_values(
                ["clusters", "logfoldchanges"], ascending=[True, False]
            )
            .groupby("clusters")
            .apply(lambda x: x.loc[x["names"].isin(expressList)].iloc[:useGeneCounts])
            .reset_index(drop=True)
        )

        clusterEnrichedGeneName_FilteredDf = clusterEnrichedGene_FilteredDf.groupby(
            "clusters"
        )["names"].agg(lambda x: list(x))

        clusterEnrichedGeneFc_FilteredDf = clusterEnrichedGene_FilteredDf.groupby(
            "clusters"
        )["logfoldchanges"].agg(lambda x: np.exp2(x).mean())

        print(clusterEnrichedGeneName_FilteredDf.map(lambda x: len(x)))

        print(clusterEnrichedGeneFc_FilteredDf)

        if useCluster == "all":
            useClusterLs = list(clusterEnrichedGeneName_FilteredDf.index)
        else:
            useClusterLs = useCluster

        #     return arrayExpressDf_StageTissue

        # clusterName = useClusterLs[0]
        #     import pdb;pdb.set_trace()
        for clusterName in useClusterLs:
            fig, ax = plt.subplots(figsize=[5, 3])
            clusterEnrichedGeneExpressPatternInBulkDf = (
                arrayExpressDf_StageTissue.loc[
                    :, clusterEnrichedGeneName_FilteredDf[clusterName]
                ]
                .median(1)
                .unstack()
                .reindex(stageOrderLs)
                .reindex(tissueOrderLs, axis=1)
            )
            sns.heatmap(clusterEnrichedGeneExpressPatternInBulkDf, cmap="Reds", ax=ax)
            ax.set_title(f"Cluster {clusterName}")
            ax.set_xlabel("Tissue")
            ax.set_ylabel("Stage")

        EnrichedGeneExpressPatternInBulkDf = clusterEnrichedGeneName_FilteredDf.map(
            lambda x: arrayExpressDf_StageTissue.loc[:, x].median(1).idxmax()
        )
        return EnrichedGeneExpressPatternInBulkDf

    @staticmethod
    def cellTypeAnnoByEnrichedScore(adata, label, markerGeneDt, threads=12, times=100):
        """
        通过enriched score对cluster进行注释

        adata:
            必须有raw且为log-transformed

        label:
            adata.obs中的列名

        markerGeneDt:
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

        threads:
            使用核心数

        times:
            重排的次数
        """
        adata = adata.copy()
        adata = adata[:, ~adata.var.index.str.contains("_")]
        adataRaw = adata.raw.to_adata()
        adataRaw = adataRaw[:, ~adataRaw.var.index.str.contains("_")]
        adata.raw = adataRaw

        markerGeneLs = list(set([y for x in markerGeneDt.values() for y in x]))
        clusterEnrichedScoreDf = getEnrichedScore(
            adata, label, markerGeneLs, threads, times
        )
        clusterTypeAvgEnrichedScoreDf = calculateGeneAverageEx(
            clusterEnrichedScoreDf, markerGeneDt
        )
        return clusterTypeAvgEnrichedScoreDf

    @staticmethod
    def cellTypeAnnoByCellScore(adata, markerGenesDt, clusterLabel):
        """
        利用cellscore计算每个细胞的type

        adata:
            anndata
        markerGenesDt:
            {type:[genes]}
        clusterLabel:
            cluster label

        return:
            cellScoreByGenesDf:
                每个细胞的cellScore
            clusterTypeRatio:
                每个cluster的type比例
        """
        adata = adata.copy()
        adata = adata[:, ~adata.var.index.str.contains("_")]
        adataRaw = adata.raw.to_adata()
        adataRaw = adataRaw[:, ~adataRaw.var.index.str.contains("_")]
        adata.raw = adataRaw

        for name, genes in markerGenesDt.items():
            sc.tl.score_genes(adata, genes, score_name=name, use_raw=True)

        cellScoreByGenesDf = adata.obs[markerGenesDt.keys()]
        cellScoreByGenesDf["maxType"], cellScoreByGenesDf["maxScore"] = (
            cellScoreByGenesDf.idxmax(1),
            cellScoreByGenesDf.max(1),
        )
        cellScoreByGenesDf["typeName"] = cellScoreByGenesDf["maxType"]
        cellScoreByGenesDf.loc[
            cellScoreByGenesDf.loc[:, "maxScore"] < 0, "typeName"
        ] = "Unknown"

        adata.obs["typeName"] = cellScoreByGenesDf["typeName"]

        clusterTypeRatio = (
            adata.obs.groupby(clusterLabel)["typeName"]
            .apply(lambda x: x.value_counts() / len(x))
            .unstack()
        )
        return cellScoreByGenesDf, clusterTypeRatio


#######
# 绘图
#######


##########
##bustools
##########
class bustools(object):
    @staticmethod
    def parseBustoolsIndex(t2gPath, t2cPath=False):
        """
        解析bustools的索引文件

        t2gPath:
            kbpython index t2g

        return:
            t2gDt: key为转录本名 value为基因名
            trLs: 所有的转录本 sorted
            geneLs: 所有的基因 sorted
        """
        t2cTrList = []
        if not t2cPath:
            pass
        else:
            with open(t2cPath) as t2cFh:
                for line in t2cFh:
                    t2cTrList.append(line.strip())
        t2cTrSet = set(t2cTrList)

        t2gDt = {}
        trLs = []
        with open(t2gPath) as t2gFh:
            for line in t2gFh:
                lineLs = line.split()
                if (not t2cTrSet) | (lineLs[0] in t2cTrSet):
                    t2gDt[lineLs[0]] = lineLs[1]
                    trLs.append(lineLs[0])
                else:
                    pass
        geneLs = list(set(t2gDt.values()))
        return t2gDt, trLs, geneLs

    @staticmethod
    def parseMatEc(ecPath, t2gDt, trLs):
        """
        解析表达矩阵ec

        ecPath:
            kb-python 产生的bus文件
        t2gDt:
            parseBustoolsIndex 产生的字典 key为转录本名 value为基因名
        trLs:
            parseBustoolsIndex 产生的list 所有的转录本 sorted

        return:
            函数 输入ec 输出对应的geneLs
        """
        ecsDt = {}
        with open(ecPath) as ecFh:
            for line in ecFh:
                l = line.split()
                ec = int(l[0])
                trs = [int(x) for x in l[1].split(",")]
                ecsDt[ec] = trs

        def __ec2g(ec):
            """
            函数 输入ec 输出对应的geneLs
            """
            if ec in ecsDt:
                return set(t2gDt[trLs[t]] for t in ecsDt[ec])
            else:
                return set([])

        return __ec2g

    @staticmethod
    def getBustoolsMappingResult(
        t2gPath, ecPath, busPath, method="inner", filterUmi=False
    ):
        """
        解析kbpython结果 并获得表达情况

        t2gPath:
            kbpython index t2g
        ecPath:
            kbpython matrix ec
        busPath:
            kbpython bus
        method:
            inner|outer  inner指与多个基因比对上取交集 outer指取并集
        filterUmi:
            是否只保留仅比对上一个基因的umi
        return:
            df:
                columns = ["barcode", "umi", "geneSet"]
        """
        logger.info("start parse index")
        t2gDt, trLs, geneLs = bustools.parseBustoolsIndex(t2gPath)

        logger.info("start parse mat")
        ec2GeneFun = bustools.parseMatEc(ecPath, t2gDt, trLs)

        busFh = StringIO()

        logger.info("start parse bus")
        sh.bustools.text("-p", busPath, _out=busFh)

        busFh = StringIO(busFh.getvalue())

        busDf = pd.read_csv(
            busFh, sep="\t", header=None, names=["barcode", "umi", "ec", "count"]
        )

        logger.info("start get mapped gene")
        busDf = busDf.assign(geneLs=lambda x: x["ec"].map(ec2GeneFun))

        def __getSetIntersect(*setList):
            if len(setList) == 1:
                return setList[0]
            else:
                return setList[0].intersection(*setList[1:])

        def __getSetOutersect(*setList):
            if len(setList) == 1:
                return setList[0]
            else:
                return setList[0].union(*setList[1:])

        setFc = {"outer": __getSetOutersect, "inner": __getSetIntersect}[method]

        logger.info("start get finnal results")

        busDf = (
            busDf.groupby(["barcode", "umi"])["geneLs"]
            .agg(lambda x: setFc(*x))
            .reset_index()
        ).assign(barcodeUmi=lambda df: df["barcode"] + "_" + df["umi"])
        logger.info(f"before filter {len(busDf)} Umis found")
        if filterUmi:
            logger.info("Umi filter: True, start filter")
            busDf = (
                busDf.assign(geneCounts=lambda df: df["geneLs"].map(len))
                .query("geneCounts == 1")
                .assign(geneLs=lambda df: df["geneLs"].map(lambda x: list(x)[0]))
            ).drop("geneCounts", axis=1)

            logger.info(f"after filter {len(busDf)} Umis found")
        else:
            logger.info("Umi filter: False")
        return busDf

    @staticmethod
    def getadataFromKbNucleiResult(
        t2gPath,
        ecPath,
        splicePath,
        unsplicePath,
        needUmiMappingInfo=False,
        adataPath=False,
    ):
        """
        用于从kb的nuclei策略中获得adata
        t2gPath:
            kbpython index t2g
        ecPath:
            kbpython matrix ec
        splicePath:
            kbpython splice bus
        unsplicePath:
            kbpython unsplice bus
        needUmiMappingInfo:.
            need umi mapping info or not
        adataPath:
            adata store path

        return:
            adata
            (umiMappingDf)
        """
        logger.info("start parse splice bus")
        spliceBusDf = bustools.getBustoolsMappingResult(
            t2gPath, ecPath, splicePath, "inner", True
        )
        logger.info("start parse unsplice bus")
        unspliceBusDf = bustools.getBustoolsMappingResult(
            t2gPath, ecPath, unsplicePath, "inner", True
        )
        kbDf = pd.concat([spliceBusDf, unspliceBusDf])

        logger.info("start get overlap umi")
        kbUmiGeneCountsSr = kbDf.groupby("barcodeUmi")["geneLs"].agg("count")
        unoverlapUmiLs = list(kbUmiGeneCountsSr.pipe(lambda x: x[x == 1]).index)
        overlapUmiSt = set(kbUmiGeneCountsSr.pipe(lambda x: x[x != 1]).index)
        overlapUseSr = (
            kbDf.query("barcodeUmi in @overlapUmiSt")
            .groupby("barcodeUmi")["geneLs"]
            .apply(lambda df: True if df.iat[0] == df.iat[1] else False)
        )
        overlapUmiUseLs = list(overlapUseSr.pipe(lambda x: x[x]).index)
        useUmiLs = sorted([*unoverlapUmiLs, *overlapUmiUseLs])

        logger.info("start filter overlap umi and creat anndata")
        kbDf = kbDf.drop_duplicates("barcodeUmi").query("barcodeUmi in @useUmiLs")
        kbMtxDf = (
            kbDf.groupby("barcode")["geneLs"]
            .apply(lambda df: df.value_counts())
            .unstack()
            .fillna(0)
        )
        kbSplicedMtxDf = (
            spliceBusDf.query("barcodeUmi in @unoverlapUmiLs")
            .groupby("barcode")["geneLs"]
            .apply(lambda df: df.value_counts())
            .unstack()
            .reindex(kbMtxDf.index)
            .reindex(kbMtxDf.columns, axis=1)
            .fillna(0)
        )
        kbUnsplicedMtxDf = (
            unspliceBusDf.query("barcodeUmi in @unoverlapUmiLs")
            .groupby("barcode")["geneLs"]
            .apply(lambda df: df.value_counts())
            .unstack()
            .reindex(kbMtxDf.index)
            .reindex(kbMtxDf.columns, axis=1)
            .fillna(0)
        )

        kbAd = basic.creatAnndataFromDf(
            kbMtxDf, spliced=kbSplicedMtxDf, unspliced=kbUnsplicedMtxDf
        )

        if adataPath:
            logger.info("start write anndata")
            kbAd.write(adataPath)

        if needUmiMappingInfo:
            return (kbAd, kbDf)
        else:
            return kbAd


######
# cellranger
#####
class parseCellranger(object):
    @staticmethod
    def extractReadCountsByUmiFromTenX(
        molInfoPath: str, kitVersion: Literal["v2", "v3"] = "v2", filtered: bool = True
    ) -> pd.DataFrame:
        """
        parse molInfo_info.h5

        Parameters
        ----------
        molInfoPath : strkitVersion, optional
            molecule_info.h5, by default "v2"
        Kitversion : 'v2' or 'v3'
            v2: umi 10bp; v3: umi 12bp
        filtered : bool, optional
            only use these cells which pass the filter, by default True

        Returns
        -------
        pd.DataFrame
            columns : ["barcodeUmi", "featureName", "readCount"]
        """
        umiLength = {"v2": 10, "v3": 12}[kitVersion]

        def NumToSeq():
            nonlocal umiLength

            numToBase = {"00": "A", "01": "C", "10": "G", "11": "T"}

            def _numToSeq(num):
                num = int(num)
                numStr = f"{num:032b}"[-umiLength * 2 :]
                return "".join(
                    [numToBase[numStr[2 * x : 2 * x + 2]] for x in range(umiLength)]
                )

            return _numToSeq

        logger.warning(
            "It consumes a lot of memory when processing files exceeding 1Gb."
        )
        numToSeq = NumToSeq()
        molInfo = h5py.File(molInfoPath, "r")
        allBarcodeAr = molInfo["barcodes"][()].astype(str)
        useBarcodeAr = allBarcodeAr[molInfo["barcode_idx"][()]]
        useFeatureAr = molInfo["features/id"][()][molInfo["feature_idx"][()]]
        umiAr = molInfo["umi"][()]
        countAr = molInfo["count"][()]

        if filtered:
            useCellLs = allBarcodeAr[molInfo["barcode_info/pass_filter"][()][:, 0]]
            useCellBoolLs = np.isin(useBarcodeAr, useCellLs)
            useBarcodeAr = useBarcodeAr[useCellBoolLs]
            useFeatureAr = useFeatureAr[useCellBoolLs]
            umiAr = umiAr[useCellBoolLs]
            countAr = countAr[useCellBoolLs]

        allUmiCount = pd.DataFrame(
            np.c_[
                umiAr,
                useBarcodeAr,
                countAr,
                useFeatureAr,
            ]
        )
        allUmiCount[0] = allUmiCount[0].map(numToSeq)
        allUmiCount["barcodeUmi"] = allUmiCount[1] + "_" + allUmiCount[0]
        allUmiCount = allUmiCount.reindex(["barcodeUmi", 3, 2], axis=1, copy=False)
        allUmiCount.columns = ["barcodeUmi", "featureName", "readCount"]
        return allUmiCount


### multilayer use
#####解析IR结果####


class parseSnuupy(object):
    @staticmethod
    def createMdFromSnuupy(
        ad: anndata.AnnData, removeAmbiguous: bool = True
    ) -> "mu.MuData":
        import muon as mu
        import scipy.sparse as ss

        ad = parseSnuupy.updateOldMultiAd(ad)
        md = mu.MuData(
            dict(
                apa=basic.getPartialLayersAdata(
                    multiModle.getMatFromObsm(ad, "APA", raw=True)
                ),
                abundance=basic.getPartialLayersAdata(
                    multiModle.getMatFromObsm(ad, "Abundance", raw=True)
                ),
                spliced=basic.getPartialLayersAdata(
                    multiModle.getMatFromObsm(ad, "Spliced", raw=True)
                ),
            )
        )
        # md['apa'].X = ss.csr_matrix(md['apa'].X.A)
        # md['abundance'].X = ss.csr_matrix(md['abundance'].X.A)
        # md['spliced'].X = ss.csr_matrix(md['spliced'].X.A)

        if removeAmbiguous:
            md = md[:, ~md.var.index.str.contains("_N_APA|_Ambiguous_fullySpliced")]

        md = md.copy()
        md.update()
        return md

    @staticmethod
    def updateOldMultiAd(adata: anndata.AnnData) -> anndata.AnnData:
        """
        update MultiAd from old version (all data deposit in X) to the 1.0 version (data deposit in obsm)
        """
        adata = adata.copy()

        def __addMatToObsm(adata, keyword):
            """
            read var name of adata, and add data matched the keyword to uns of adata
            """
            if keyword == "Abundance":
                subIndex = ~adata.var.index.str.contains("APA|Spliced")
            else:
                subIndex = adata.var.index.str.contains(keyword)
            subAd = adata[:, subIndex]
            adata.obsm[keyword] = subAd.X
            adata.uns[f"{keyword}_label"] = subAd.var.index.values

        __addMatToObsm(adata, "APA")
        __addMatToObsm(adata, "Spliced")
        __addMatToObsm(adata, "Abundance")
        adata = adata[:, ~adata.var.index.str.contains("APA|Spliced")]
        return adata

    @staticmethod
    def getSpliceInfoOnIntronLevel(irInfoPath, useIntronPath=None):
        """
        从intron水平获得剪接情况
        irInfoPath:
            snuupy getSpliceInfo的结果
        useIntronPath:
            使用的intron列表，需要表头'intron_id'

        return:
            adata:
                X: unsplice + splice
                layer[unspliced, spliced]
        """
        irInfoDf = pd.read_table(irInfoPath)
        intronCountMtxDt = {}
        intronRetenMtxDt = {}
        # 输入 0base
        # 输出 1base
        allLinesCounts = len(irInfoDf)
        for i, line in enumerate(irInfoDf.itertuples()):
            barcode = line.Name.split("_")[0]
            lineCountMtxDt = intronCountMtxDt.get(barcode, {})
            lineRetenMtxDt = intronRetenMtxDt.get(barcode, {})

            exonOverlapInfo = [int(x) for x in line.ExonOverlapInfo.split(",")]
            minIntron = min(exonOverlapInfo)
            maxIntron = max(exonOverlapInfo)
            intronCov = list(range(minIntron, maxIntron))

            if pd.isna(line.IntronOverlapInfo):
                intronOverlapInfo = []
            else:
                intronOverlapInfo = [int(x) for x in line.IntronOverlapInfo.split(",")]

            intronCov.extend(intronOverlapInfo)
            intronCov = set(intronCov)

            for intronCovNum in intronCov:
                lineCountMtxDt[f"{line.geneId}_intron_{intronCovNum+1}"] = (
                    lineCountMtxDt.get(f"{line.geneId}_intron_{intronCovNum+1}", 0) + 1
                )
            for intronRentNum in intronOverlapInfo:
                lineRetenMtxDt[f"{line.geneId}_intron_{intronRentNum+1}"] = (
                    lineRetenMtxDt.get(f"{line.geneId}_intron_{intronRentNum+1}", 0) + 1
                )

            intronCountMtxDt[barcode] = lineCountMtxDt
            intronRetenMtxDt[barcode] = lineRetenMtxDt
            if i % 1e5 == 0:
                logger.info(f"{i}/{allLinesCounts}")
        intronCountMtxDf = pd.DataFrame.from_dict(intronCountMtxDt, "index")
        intronRetenMtxDf = pd.DataFrame.from_dict(intronRetenMtxDt, "index")
        if useIntronPath:
            useIntronDf = pd.read_table(useIntronPath)
            useIntronLs = list(
                useIntronDf["intron_id"].str.split(".").str[0]
                + "_intron_"
                + useIntronDf["intron_id"].str.split("intron").str[1]
            )
            intronRetenMtxDf = intronRetenMtxDf.loc[
                :, intronRetenMtxDf.columns.isin(useIntronLs)
            ]
            intronCountMtxDf = intronCountMtxDf.loc[
                :, intronCountMtxDf.columns.isin(useIntronLs)
            ]
        intronCountMtxDf.index = intronCountMtxDf.index + "-1"
        intronRetenMtxDf.index = intronRetenMtxDf.index + "-1"
        intronRetenMtxDf = intronRetenMtxDf.fillna(0)
        intronCountMtxDf = intronCountMtxDf.fillna(0)
        intronCountMtxAd = basic.creatAnndataFromDf(intronCountMtxDf)
        intronRetenMtxAd = basic.creatAnndataFromDf(intronRetenMtxDf)

        useIntronLs = list(intronRetenMtxAd.var.index | intronCountMtxAd.var.index)
        useCellLs = list(intronRetenMtxAd.obs.index | intronCountMtxAd.obs.index)

        intronRetenMtxDf = (
            intronRetenMtxAd.to_df()
            .reindex(useIntronLs, axis=1)
            .reindex(useCellLs)
            .fillna(0)
        )
        intronCountMtxDf = (
            intronCountMtxAd.to_df()
            .reindex(useIntronLs, axis=1)
            .reindex(useCellLs)
            .fillna(0)
        )

        return basic.creatAnndataFromDf(
            intronCountMtxDf,
            unspliced=intronRetenMtxDf,
            spliced=intronCountMtxDf - intronRetenMtxDf,
        )

    @staticmethod
    def getSpliceInfoFromSnuupyAd(nanoporeAd):
        """
        用于从snuupy crMode产生的NanoporeMtx中提取产生splice和unsplice的read

        return:
            adata:
                X: unsplice + splice
                layer[unspliced, spliced]
        """
        nanoporeCountAd = nanoporeAd[:, ~nanoporeAd.var.index.str.contains("_")]
        unsplicedAd = nanoporeAd[
            :, nanoporeAd.var.index.str.contains("False_fullySpliced")
        ]
        unsplicedAd.var.index = unsplicedAd.var.index.str.split("_").str[0]
        splicedAd = nanoporeAd[
            :, nanoporeAd.var.index.str.contains("True_fullySpliced")
        ]
        splicedAd.var.index = splicedAd.var.index.str.split("_").str[0]
        useGeneLs = sorted(list(set(splicedAd.var.index) | set(unsplicedAd.var.index)))
        unsplicedDf = unsplicedAd.to_df().reindex(useGeneLs, axis=1).fillna(0)
        splicedDf = splicedAd.to_df().reindex(useGeneLs, axis=1).fillna(0)
        allSpliceDf = splicedDf + unsplicedDf
        return basic.creatAnndataFromDf(
            allSpliceDf, spliced=splicedDf, unspliced=unsplicedDf
        )

    @staticmethod
    def getDiffSplicedIntron(
        snSpliceIntronInfoAd,
        groupby,
        minCount,
        minDiff=0.0,
        threads=24,
        useMethod="winflat",
        fdrMethod="indep",
        winflatPath="/public/home/jiajb/soft/IRFinder/IRFinder-1.2.5/bin/util/winflat",
        fisherMethod="two-sided",
    ):
        """
        snSpliceIntronInfoAd:
            adata: layer['spliced', 'unspliced']
        groupby:
            data will be groupbyed by this label
        minCount:
            read total counts lower than this cutoff will be filtered
        minDiff:
            unspliced read ratio lower than this cutoff will be filtered
        useMethod:
            winflat|fisher
        fdrMethod:
            indep|negcorr
            FH or FY
        fisherMethod:
            two-sided|less|greater
            less: used to calculate these intron enriched in this group
            greater: used to calculate these intron excluded in this group
        """
        from pandarallel import pandarallel
        from statsmodels.stats import multitest
        from scipy.stats import fisher_exact
        import os

        pandarallel.initialize(nb_workers=threads)

        def calcuPvalueByWinflat(line):
            nonlocal winflatPath
            xUnsplice = line.iloc[0]
            yUnsplice = line.iloc[1]
            xTotal = line.iloc[2]
            yTotal = line.iloc[3]
            resultStr = (
                os.popen(
                    f"{winflatPath} -xvalue {xUnsplice} -yvalue {yUnsplice} -diff {xTotal} {yTotal}"
                )
                .read()
                .strip()
            )
            if not resultStr:
                return 1.0
            resultFloat = [
                float(x)
                for x in [
                    x.strip().split("=")[-1].strip() for x in resultStr.split("\n")
                ]
            ][1]

            return resultFloat

        def calcuPvalueByFisher(line):
            nonlocal fisherMethod
            xUnsplice = line.iloc[0]
            yUnsplice = line.iloc[1]
            xTotal = line.iloc[2]
            yTotal = line.iloc[3]
            xSplice = xTotal - xUnsplice
            ySplice = yTotal - yUnsplice
            return fisher_exact(
                [[xUnsplice, xSplice], [yUnsplice, ySplice]], fisherMethod
            )[1]

        allClusterDiffDt = {}
        calcuPvalue = {"winflat": calcuPvalueByWinflat, "fisher": calcuPvalueByFisher}[
            useMethod
        ]

        for singleCluster in snSpliceIntronInfoAd.obs[groupby].unique():
            snSpliceIntronInfoAd.obs = snSpliceIntronInfoAd.obs.assign(
                cate=lambda df: np.select(
                    [df[groupby].isin([singleCluster])],
                    [singleCluster],
                    f"non-{singleCluster}",
                )
            )
            clusterSpliceIntronInfoAd = mergeadata(
                snSpliceIntronInfoAd, "cate", ["unspliced", "spliced"]
            )
            clusterSpliceIntronInfoAd = clusterSpliceIntronInfoAd[
                :, clusterSpliceIntronInfoAd.to_df().min(0) >= minCount
            ]

            clusterSpliceIntronInfoDf = pd.concat(
                [
                    clusterSpliceIntronInfoAd.to_df("unspliced").T,
                    clusterSpliceIntronInfoAd.to_df().T,
                ],
                axis=1,
            )
            #         import pdb; pdb.set_trace()
            clusterSpliceIntronInfoDf.columns = [
                "unspliced",
                "non-unspliced",
                "total",
                "non-total",
            ]

            clusterSpliceIntronInfoDf[
                "pvalue"
            ] = clusterSpliceIntronInfoDf.parallel_apply(calcuPvalue, axis=1)
            clusterSpliceIntronInfoDf["fdr"] = multitest.fdrcorrection(
                clusterSpliceIntronInfoDf["pvalue"], 0.05, fdrMethod
            )[1]

            clusterSpliceIntronInfoDf = clusterSpliceIntronInfoDf.assign(
                diffRatio=lambda df: df["unspliced"] / df["total"]
                - df["non-unspliced"] / df["non-total"]
            )

            clusterSpliceIntronInfoDf = clusterSpliceIntronInfoDf.eval(
                f"significantDiff = (fdr <= 0.05) & (diffRatio >= {minDiff})"
            )
            allClusterDiffDt[singleCluster] = clusterSpliceIntronInfoDf
            logger.info(
                f"group {singleCluster} processed; {len(clusterSpliceIntronInfoDf)} input; {clusterSpliceIntronInfoDf['significantDiff'].sum()} output"
            )
        return allClusterDiffDt


############################
###parseMOFA################
############################


class detectDoublet(object):
    @staticmethod
    def byDoubletFinder(
        adata: anndata.AnnData, copy: bool = False, doubletRatio: float = 0.075
    ) -> Optional[anndata.AnnData]:
        """
        use doubletFinder detect doublets.


        Args:
            adata (anndata.AnnData): X must is raw counts
            copy (bool, optional): copy adata or not. Defaults to False.
            doubletRatio (float, optional): expected doublet ratio. Defaults to 0.075.

        Returns:
            Optional[anndata.AnnData]: anndata if copy
        """
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
        import anndata2ri
        import rpy2.robjects as ro

        adata = adata.copy() if copy else adata
        logger.info("start to transfer adata to R")

        with localconverter(anndata2ri.converter):
            ro.globalenv["adata"] = adata
        logger.info("start to preprocess adata")
        ro.r(
            f"""
        library(Seurat)
        library(DoubletFinder)
        seuratObj <- as.Seurat(adata, counts="X", data = NULL)
        seuratObj <- SCTransform(seuratObj, )
        seuratObj <- RunPCA(seuratObj)
        seuratObj <- RunUMAP(seuratObj, dims = 1:10)
        seuratObj <- FindNeighbors(seuratObj, dims = 1:10)
        seuratObj <- FindClusters(seuratObj, resolution = 0.6)
        1
        """
        )
        logger.info("start to calculate DF parameters")
        ro.r(
            f"""
        sweep.res.seuratObj <- paramSweep_v3(seuratObj, PCs = 1:10, sct = TRUE)
        sweep.stats.seuratObj <- summarizeSweep(sweep.res.seuratObj, GT = FALSE)
        annotationsDf <- seuratObj@meta.data$seurat_clusters
        homotypic.prop <- modelHomotypic(annotationsDf)
        nExp_poi <- round({doubletRatio}*nrow(seuratObj@meta.data)) 
        nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
        1
        """
        )
        logger.info("start to calculate doublets")
        ro.r(
            f"""
        seuratObj <- doubletFinder_v3(seuratObj, PCs = 1:10, pN = 0.25, pK = 0.09, nExp = nExp_poi, reuse.pANN = FALSE, sct = TRUE)
        seuratObj <- doubletFinder_v3(seuratObj, PCs = 1:10, pN = 0.25, pK = 0.09, nExp = nExp_poi.adj, reuse.pANN = paste('pANN_0.25_0.09_', nExp_poi, sep=''), sct = TRUE)
        1
        """
        )
        logger.info("start to intergrate result with adata")
        with localconverter(ro.default_converter + pandas2ri.converter):
            finalDf = ro.r("seuratObj@meta.data")
        colNameSr = list(
            ro.r(
                "c(paste('DF.classifications_0.25_0.09_', nExp_poi, sep=''), paste('DF.classifications_0.25_0.09_', nExp_poi.adj, sep=''))"
            )
        )
        finalDf = finalDf.filter(colNameSr).rename(
            {
                x: y
                for x, y in zip(
                    colNameSr, ["doubletFinder_raw", "doubletFinder_adjusted"]
                )
            },
            axis=1,
        )
        adata.obs = adata.obs.join(finalDf.copy(deep=True))

        if copy:
            return adata

    @staticmethod
    def byScDblFinder(
        adata: anndata.AnnData,
        layer: str = "X",
        copy: bool = False,
        doubletRatio: float = 0.1,
        skipCheck: bool = False,
    ) -> Optional[anndata.AnnData]:
        """
        use ScDblFinder detect doublets.

        Parameters
        ----------
        adata : anndata.AnnData
            anndata
        layer : str, optional
            use this layer. must is raw counts. Defaults to X
        copy : bool, optional
            copy adata or not. Defaults to False.
        doubletRatio : float, optional
            expected doublet ratio. Defaults to 0.1

        Returns
        -------
        Optional[anndata.AnnData]
            anndata if copy
        """
        import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from jpy_tools.rTools import py2r, r2py

        scDblFinder = importr("scDblFinder")

        if not skipCheck:
            basic.testAllCountIsInt(adata, layer)

        tempAd = basic.getPartialLayersAdata(adata, layer)
        tempAd.layers["counts"] = tempAd.X

        logger.info("start to transfer adata to R")
        tempAdr = py2r(tempAd)
        del tempAd

        logger.info("start to calculate doublets")
        tempAdr = scDblFinder.scDblFinder(tempAdr, dbr=doubletRatio)

        logger.info("start to intergrate result with adata")
        scDblFinderResultDf = r2py(tempAdr.slots["colData"])

        adata.obs = adata.obs.join(
            scDblFinderResultDf.filter(regex=r"^scDblFinder[\w\W]*").copy(deep=True)
        )
        adata.obs.columns = adata.obs.columns.astype(str)
        if copy:
            return adata


class parseStarsolo(object):
    @staticmethod
    def transferMtxToH5ad(starsoloMtxDir):
        import scipy.sparse as ss
        import glob

        starsoloMtxDir = starsoloMtxDir.rstrip("/") + "/"
        ls_allMtx = glob.glob(f"{starsoloMtxDir}*.mtx")
        ls_mtxName = [x.split("/")[-1].split(".")[0] for x in ls_allMtx]
        path_barcode = f"{starsoloMtxDir}barcodes.tsv"
        path_feature = f"{starsoloMtxDir}features.tsv"
        df_barcode = pd.read_table(path_barcode, names=["barcodes"]).set_index(
            "barcodes"
        )
        df_feature = pd.read_table(
            path_feature, names=["geneid", "symbol", "category"]
        ).set_index("geneid")
        path_out = f"{starsoloMtxDir}adata.h5ad"
        adata = sc.AnnData(
            X=ss.csr_matrix((len(df_barcode), len(df_feature))),
            obs=df_barcode,
            var=df_feature,
        )
        for mtxName, mtxPath in zip(ls_mtxName, ls_allMtx):
            adata.layers[mtxName] = sc.read_mtx(mtxPath).X.T
            logger.info(f"read {mtxName} done")
        adata.write_h5ad(path_out)


class diffxpy(object):
    @staticmethod
    def parseAdToDiffxpyFormat(
        adata: anndata.AnnData,
        testLabel: str,
        testComp: str,
        otherComp: Optional[Union[str, List[str]]] = None,
        batchLabel: Optional[str] = None,
        minCellCounts: int = 5,
        keyAdded: str = "temp",
    ):
        if not otherComp:
            otherComp = list(adata.obs[testLabel].unique())
            otherComp = [x for x in otherComp if x != testComp]
        if isinstance(otherComp, str):
            otherComp = [otherComp]
        adata = adata[adata.obs[testLabel].isin([testComp, *otherComp])].copy()
        sc.pp.filter_genes(adata, min_cells=minCellCounts)
        adata.obs = adata.obs.assign(
            **{keyAdded: np.select([adata.obs[testLabel] == testComp], ["1"], "0")}
        )
        if batchLabel:
            adata.obs = adata.obs.assign(
                **{
                    f"{batchLabel}_{keyAdded}": adata.obs[batchLabel].astype(str)
                    + "_"
                    + adata.obs[keyAdded].astype(str)
                }
            )
        return adata

    def testTwoSample(
        adata: anndata.AnnData,
        keyTest: str = "temp",
        batchLabel: Optional[str] = None,
        quickScale: bool = False,
        sizeFactor: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Use wald test between two sample.
        This function is always following `parseAdToDiffxpyFormat`

        Parameters
        ----------
        adata : anndata.AnnData
            generated by `parseAdToDiffxpyFormat`
        keyTest : str, optional
            `keyAdded` parameter of `parseAdToDiffxpyFormat`, by default 'temp'
        batchLabel : Optional[str], optional
            by default None
        quickScale : bool, optional
            by default False
        sizeFactor : Optional[str], optional
            by default None

        Returns
        -------
        pd.DataFrame
        """
        import diffxpy.api as de

        assert len(adata.obs[keyTest].unique()) == 2, "More Than Two Samples found"
        if batchLabel:
            test = de.test.wald(
                data=adata,
                formula_loc=f"~ 1 + {keyTest} + {batchLabel}_{keyTest}",
                factor_loc_totest=keyTest,
                constraints_loc={f"{batchLabel}_{keyTest}": keyTest},
                quick_scale=quickScale,
                size_factors=sizeFactor,
            )
        else:
            test = de.test.wald(
                data=adata,
                formula_loc=f"~ 1 + {keyTest}",
                factor_loc_totest=keyTest,
                quick_scale=quickScale,
                size_factors=sizeFactor,
            )
        return test.summary()

    @staticmethod
    def vsRest(
        adata: anndata.AnnData,
        layer: Optional[str],
        testLabel: str,
        groups: Optional[List[str]] = None,
        batchLabel: Optional[str] = None,
        minCellCounts: int = 5,
        sizeFactor: Optional[str] = None,
        inputIsLog: bool = True,
        keyAdded: str = None,
        quickScale: bool = True,
        copy: bool = False,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        use wald to find DEG

        Parameters
        ----------
        adata : anndata.AnnData
        layer : Optional[str]
            if is not log-transformed, the `inpuIsLog` must be assigned by `False`
        testLabel : str
            column name in adata.obs. used as grouping Infomation
        groups : Optional[List[str]], optional
            only use these groups, by default None
        batchLabel : Optional[str], optional
            column name in adata.obs. used as batch Infomation. by default None
        minCellCounts : int, optional
            used to filter genes. by default 5
        sizeFactor : Optional[str], optional
            column name in adata.obs. used as size factor Infomation. by default None
        inputIsLog : bool, optional
            is determined by `layer`. by default True
        keyAdded : str, optional
            key used to update adata.uns, by default None
        quickScale : bool, optional
            by default True
        copy : bool, optional
            by default False

        Returns
        -------
        Optional[Dict[str, pd.DataFrame]]
            return pd.DataFrame if copy
        """
        import scipy.sparse as ss

        layer = "X" if not layer else layer
        if not groups:
            groups = list(adata.obs[testLabel].unique())
        if not keyAdded:
            keyAdded = f"diffxpyVsRest_{testLabel}"
        ls_useCol = [testLabel]
        if batchLabel:
            ls_useCol.append(batchLabel)
        if sizeFactor:
            ls_useCol.append(batchLabel)
        adataOrg = adata.copy() if copy else adata
        adata = basic.getPartialLayersAdata(adataOrg, layer, ls_useCol)
        adata.X = adata.X.A if ss.issparse(adata.X) else adata.X
        if inputIsLog:
            adata.X = np.exp(adata.X) - 1
        adata = adata[adata.obs[testLabel].isin(groups)].copy()

        logger.info("start performing test")
        adataOrg.uns[keyAdded] = {"__cat": "vsRest"}
        for singleGroup in groups:
            ad_test = diffxpy.parseAdToDiffxpyFormat(
                adata,
                testLabel,
                singleGroup,
                batchLabel=batchLabel,
                minCellCounts=minCellCounts,
                keyAdded="temp",
            )
            test = diffxpy.testTwoSample(
                ad_test, "temp", batchLabel, quickScale, sizeFactor
            )
            adataOrg.uns[keyAdded][singleGroup] = test
            logger.info(f"{singleGroup} done")
        if copy:
            return adataOrg.uns[keyAdded]

    def pairWise(
        adata: anndata.AnnData,
        layer: Optional[str],
        testLabel: str,
        groups: Optional[List[str]] = None,
        batchLabel: Optional[str] = None,
        minCellCounts: int = 5,
        sizeFactor: Optional[str] = None,
        inputIsLog: bool = True,
        keyAdded: str = None,
        quickScale: bool = True,
        copy: bool = False,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        use wald to find DEG

        Parameters
        ----------
        adata : anndata.AnnData
        layer : Optional[str]
            if is not log-transformed, the `inpuIsLog` must be assigned by `False`
        testLabel : str
            column name in adata.obs. used as grouping Infomation
        groups : Optional[List[str]], optional
            only use these groups, by default None
        batchLabel : Optional[str], optional
            column name in adata.obs. used as batch Infomation. by default None
        minCellCounts : int, optional
            used to filter genes. by default 5
        sizeFactor : Optional[str], optional
            column name in adata.obs. used as size factor Infomation. by default None
        inputIsLog : bool, optional
            is determined by `layer`. by default True
        keyAdded : str, optional
            key used to update adata.uns, by default None
        quickScale : bool, optional
            by default True
        copy : bool, optional
            by default False

        Returns
        -------
        Optional[Dict[str, pd.DataFrame]]
            return pd.DataFrame if copy
        """
        from itertools import product
        import scipy.sparse as ss

        layer = "X" if not layer else layer
        if not groups:
            groups = list(adata.obs[testLabel].unique())
        if not keyAdded:
            keyAdded = f"diffxpyPairWise_{testLabel}"
        ls_useCol = [testLabel]
        if batchLabel:
            ls_useCol.append(batchLabel)
        if sizeFactor:
            ls_useCol.append(batchLabel)
        adataOrg = adata.copy() if copy else adata
        adata = basic.getPartialLayersAdata(adataOrg, layer, ls_useCol)
        adata.X = adata.X.A if ss.issparse(adata.X) else adata.X
        if inputIsLog:
            adata.X = np.exp(adata.X) - 1
        adata = adata[adata.obs[testLabel].isin(groups)].copy()

        logger.info("start performing test")
        adataOrg.uns[keyAdded] = {"__cat": "pairWise"}
        for x, y in product(range(len(groups)), range(len(groups))):
            if (
                x >= y
            ):  # only calculate half combination of groups, then use these result to fullfill another half
                continue
            testGroup = groups[x]
            backgroundGroup = groups[y]
            ad_test = diffxpy.parseAdToDiffxpyFormat(
                adata,
                testLabel,
                testGroup,
                backgroundGroup,
                batchLabel=batchLabel,
                minCellCounts=minCellCounts,
                keyAdded="temp",
            )
            test = diffxpy.testTwoSample(
                ad_test, "temp", batchLabel, quickScale, sizeFactor
            )
            adataOrg.uns[keyAdded][f"test_{testGroup}_bg_{backgroundGroup}"] = test
            logger.info(f"{testGroup} vs {backgroundGroup} done")
        for x, y in product(range(len(groups)), range(len(groups))):
            if x <= y:  # use these result to fullfill another half
                continue
            testGroup = groups[x]
            backgroundGroup = groups[y]
            adataOrg.uns[keyAdded][
                f"test_{testGroup}_bg_{backgroundGroup}"
            ] = adataOrg.uns[keyAdded][f"test_{backgroundGroup}_bg_{testGroup}"].copy()
            adataOrg.uns[keyAdded][f"test_{testGroup}_bg_{backgroundGroup}"][
                "log2fc"
            ] = -adataOrg.uns[keyAdded][f"test_{testGroup}_bg_{backgroundGroup}"][
                "log2fc"
            ]

        if copy:
            return adataOrg.uns[keyAdded]

    @staticmethod
    def getMarker(
        adata: anndata.AnnData,
        key: str,
        qvalue=0.05,
        log2fc=np.log2(1.5),
        mean=0.5,
        detectedCounts=-1,
    ) -> pd.DataFrame:
        """
        parse `vsRest` and `pairWise` results

        Parameters
        ----------
        adata : anndata.AnnData
            after appy `vsRest` or `pairWise`
        key : str
            `keyAdded` parameter of `vsRest` or `pairWise`
        qvalue : float, optional
            cutoff, by default 0.05
        log2fc : [type], optional
            cutoff, by default np.log2(1.5)
        mean : float, optional
            cutoff, by default 0.5
        detectedCounts : int, optional
            cutoff, only usefull for `pairWise`, by default -1

        Returns
        -------
        pd.DataFrame
            [description]
        """

        def __twoSample(df_marker, qvalue=0.05, log2fc=np.log2(1.5), mean=0.5):
            df_marker = df_marker.query(
                f"qval < {qvalue} & log2fc > {log2fc} & mean > {mean}"
            ).sort_values("coef_mle", ascending=False)
            return df_marker

        def __vsRest(
            dt_marker: Dict[str, pd.DataFrame],
            qvalue,
            log2fc,
            mean,
            detectedCounts,
        ) -> pd.DataFrame:
            ls_markerParsed = []
            for comp, df_marker in dt_marker.items():
                if comp == "__cat":
                    continue
                if "clusterName" not in df_marker.columns:
                    df_marker.insert(0, "clusterName", comp)
                df_marker = __twoSample(df_marker, qvalue, log2fc, mean)
                ls_markerParsed.append(df_marker)
            return pd.concat(ls_markerParsed)

        def __pairWise(
            dt_marker: Dict[str, pd.DataFrame],
            qvalue,
            log2fc,
            mean,
            detectedCounts=-1,
        ) -> pd.DataFrame:
            import re

            ls_markerParsed = []
            ls_compName = []
            for comp, df_marker in dt_marker.items():
                if comp == "__cat":
                    continue
                testedCluster = re.findall(r"test_([\w\W]+)_bg", comp)[0]
                bgCluster = re.findall(r"_bg_([\w\W]+)", comp)[0]
                ls_compName.append(bgCluster)
                if "testedCluster" not in df_marker.columns:
                    df_marker.insert(0, "testedCluster", testedCluster)
                if "bgCluster" not in df_marker.columns:
                    df_marker.insert(1, "bgCluster", bgCluster)

                df_marker = __twoSample(df_marker, qvalue, log2fc, mean)
                ls_markerParsed.append(df_marker)
            df_markerMerged = pd.concat(ls_markerParsed)
            df_markerMerged = (
                df_markerMerged.groupby(["testedCluster", "gene"])
                .agg(
                    {
                        "gene": "count",
                        "bgCluster": lambda x: list(x),
                        "qval": lambda x: list(x),
                        "log2fc": lambda x: list(x),
                        "mean": lambda x: list(x),
                        "coef_mle": lambda x: list(x),
                    }
                )
                .rename(columns={"gene": "counts"})
                .reset_index()
            )
            if detectedCounts <= 0:
                detectedCounts = len(set(ls_compName)) + detectedCounts
                logger.info(f"`detectedCounts` is parsed to {detectedCounts}")

            return df_markerMerged.query(f"counts >= {detectedCounts}")

        dt_marker = adata.uns[key]
        cate = dt_marker["__cat"]
        fc_parse = {
            "vsRest": __vsRest,
            "pairWise": __pairWise,
        }[cate]
        return fc_parse(dt_marker, qvalue, log2fc, mean, detectedCounts)
