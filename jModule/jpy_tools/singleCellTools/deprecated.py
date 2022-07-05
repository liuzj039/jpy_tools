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
                basic.setLayerInfo(adata, scran="log")
                logger.warning("sct_corrected layer IS log-scaled")
            else:
                basic.setLayerInfo(adata, scran="raw")
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
    k_score=30,
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
    refAd.obs["__batch"] = "reference"
    refAd.obs.index = "reference-" + refAd.obs.index
    queryAd.obs["__batch"] = "query"
    queryAd.obs.index = "query-" + queryAd.obs.index
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
        k_score=k_score,
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
    df_predScore = df_predScore.rename(
        columns=lambda x: x.split("prediction.score.")[-1]
    )

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
        k_score=k_score,
    )
    adR_integrated = seurat.IntegrateData(
        anchorset=trl(anchor), normalization_method="LogNormalize", dims=py2r(np.arange(0, npcs) + 1),
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
    ad_integrated = basic.setadataColor(
        ad_integrated, f"labelTransfer_seurat_{refLabel}", dt_color
    )
    sc.pl.umap(ad_integrated, color="batch")
    sc.pl.umap(
        ad_integrated, color=f"labelTransfer_seurat_{refLabel}", legend_loc="on data"
    )
    if copy:
        return queryAdOrg
    if needLoc:
        return ad_integrated


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


def scWGCNA(
    ad: sc.AnnData,
    groupby: str,
    layer: str,
    dir_result: str,
    jobid: str,
    k: int = 75,
    minModuleSize: int = 50,
    min_cells: int = 50,
    n_top_genes: int = 10000,
    batch_key: Optional[str] = None,
    threads: int = 1,
    soft_power: Optional[int] = None,
    max_block_size: Optional[int] = None,
    renv=None,
) -> sc.AnnData:
    """
    perform scWGCNA

    Parameters
    ----------
    ad : sc.AnnData
    groupby : str
        used for generate meta cells. This parameter is independent with `batch_key`!
    layer : str
        must be raw
    dir_result : str
        store `blockwiseConsensusModules` results
    jobid : str
        determine path used for storing results of `blockwiseConsensusModules`
    k : int, optional
        neighbor number, used for generating meta cells, by default 75
    minModuleSize : int, optional
        by default 50
    min_cells : int, optional
        by default 50
    n_top_genes : int, optional
        by default 10000
    batch_key : Optional[str], optional
        by default None
    threads : int, optional
        by default 16
    soft_power : Optional[int], optional
        by default None

    Returns
    -------
    sc.AnnData
        with WGCNA results
    """
    import scanpy.external as sce

    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rGet, rSet, ad2so, so2ad
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    rBase = importr("base")
    rUtils = importr("utils")
    tidyverse = importr("tidyverse")
    WGCNA = importr("WGCNA")
    seurat = importr("Seurat")

    R = ro.r
    R(f"disableWGCNAThreads()")

    if renv is None:
        renv = ro.Environment()

    renv["dir_result"] = dir_result
    if not batch_key:
        batch_key = groupby
    if not max_block_size:
        max_block_size = n_top_genes

    # preprocess
    ad = ad.copy()
    ad.X = ad.layers[layer].copy()
    sc.pp.filter_genes(ad, min_cells=min_cells)
    sc.pp.highly_variable_genes(
        ad,
        layer=layer,
        flavor="seurat_v3",
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        subset=True,
    )
    ad_forWgcna = ad
    so = ad2so(
        ad_forWgcna,
        layer=layer,
        # ls_obs=[x for x in [groupby, batch_key] if x],
    )
    renv["so"] = so

    # construct meta cells
    logger.info("construct meta cells")
    with ro.local_context(renv):
        R(
            f"""
        seurat_list <- list()
        for(group in unique(so${groupby})){{
        print(group)
        cur_seurat <- subset(so, {groupby} == group)
        cur_metacell_seurat <- scWGCNA::construct_metacells(
            cur_seurat, name=group,
            k={k}, reduction='umap',
            assay='RNA', slot='data'
        )
        cur_metacell_seurat${groupby} <- as.character(unique(cur_seurat${groupby}))
        cur_metacell_seurat${batch_key} <- as.character(unique(cur_seurat${batch_key}))
        seurat_list[[group]] <- cur_metacell_seurat
        }}

        # merge all of the metacells objects
        metacell_seurat <- merge(seurat_list[[1]], seurat_list[2:length(seurat_list)])
        """
        )
        ad_meta = so2ad(R("metacell_seurat"))
    print(f"shape of ad_meta: {ad_meta.shape}")
    ad_meta.X = ad_meta.layers["RNA_counts"].copy()
    sc.pp.scale(ad_meta)
    sc.tl.pca(ad_meta)
    sce.pp.harmony_integrate(
        ad_meta, batch_key, adjusted_basis="X_harmony", max_iter_harmony=50
    )
    sc.pp.neighbors(ad_meta, use_rep="X_harmony")
    sc.tl.umap(ad_meta, min_dist=0.2)
    sc.pl.umap(ad_meta, color=[groupby, batch_key], ncols=1)

    # perform wgcna
    logger.info("perform wgcna")
    with ro.local_context(renv):
        R(
            f"""
        # which genes are we using ?
        genes.use <- rownames(metacell_seurat)

        # vector of cell conditions
        group <- as.factor(metacell_seurat${groupby})

        # format the expression matrix for WGCNA
        datExpr <- as.data.frame(GetAssayData(metacell_seurat, assay='RNA', slot='data'))
        datExpr <- as.data.frame(t(datExpr))

        # only keep good genes:
        datExpr <- datExpr[,goodGenes(datExpr)]
        """
        )

        if not soft_power:

            ### wgcna parallel mode is incompatible with rpy2
            tmp_datExprRds = NamedTemporaryFile(suffix=".rds")
            path_datExpr = tmp_datExprRds.name

            tmp_powerTableRds = NamedTemporaryFile(suffix=".rds")
            path_powerTableRds = tmp_powerTableRds.name

            R(f"saveRDS(datExpr, '{path_datExpr}')")
            scriptsR = f"""
            library(WGCNA)

            datExpr <- readRDS('{path_datExpr}')
            powers = c(seq(1,10,by=1), seq(12,30, by=2))
            powerTable = list(
            data = pickSoftThreshold(
                datExpr,
                powerVector=powers,
                verbose = 100,
                networkType="signed",
                corFnc="bicor"
            )[[2]]
            )
            saveRDS(powerTable, '{path_powerTableRds}')
            """
            ls_cmd = ["-q", "-e", scriptsR]
            sh.R(*ls_cmd, _err_to_out=True, _out=sys.stdout)

            R(
                f"""
            # Choose a set of soft-thresholding powers
            powers = c(seq(1,10,by=1), seq(12,30, by=2))
            powerTable <- readRDS('{path_powerTableRds}')
            """
            )

            # # Call the network topology analysis function for each set in turn
            # powerTable = list(
            # data = pickSoftThreshold(
            #     datExpr,
            #     powerVector=powers,
            #     verbose = 100,
            #     networkType="signed",
            #     corFnc="bicor"
            # )[[2]]
            # );
            # """
            # )

            with r_inline_plot(width=768):
                R(
                    """
                # Plot the results:

                colors = c("blue", "red","black")
                # Will plot these columns of the returned scale free analysis tables
                plotCols = c(2,5,6,7)
                colNames = c("Scale Free Topology Model Fit", "Mean connectivity", "mean connectivity",
                "Max connectivity");

                # Get the minima and maxima of the plotted points
                ylim = matrix(NA, nrow = 2, ncol = 4);
                for (col in 1:length(plotCols)){
                ylim[1, col] = min(ylim[1, col], powerTable$data[, plotCols[col]], na.rm = TRUE);
                ylim[2, col] = max(ylim[2, col], powerTable$data[, plotCols[col]], na.rm = TRUE);
                }

                # Plot the quantities in the chosen columns vs. the soft thresholding power
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
            soft_power = int(input("Soft Power"))

        if threads > 1:
            R(f"enableWGCNAThreads({threads})")
        else:
            R(f"disableWGCNAThreads()")
        R(
            f"""
        softPower = {soft_power}
        nSets = 1
        setLabels = 'ODC'
        shortLabels = setLabels

        multiExpr <- list()
        multiExpr[['ODC']] <- list(data=datExpr)
        checkSets(multiExpr) 
        """
        )
        R(
            f"""
        net=blockwiseConsensusModules(multiExpr, blocks = NULL,
                                                maxBlockSize = {max_block_size}, ## This should be set to a smaller size if the user has limited RAM
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
                                                deepSplit = 4,
                                                pamStage=FALSE,
                                                detectCutHeight = 0.995, minModuleSize = {minModuleSize},
                                                mergeCutHeight = 0.2,
                                                saveConsensusTOMs = TRUE,
                                                consensusTOMFilePattern = paste0(dir_result, '/{jobid}_TOM_block.%b.rda'))
        """
        )
        # parse result
        logger.info("parse result")
        R(
            """
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

        # visualization
        logger.info("visualization")
        with r_inline_plot(width=768):
            R(
                """
            plotDendroAndColors(consTree, moduleColors, "Module colors", dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, guideHang = 0.05,
            main = paste0("ODC lineage gene dendrogram and module colors"))
            """
            )

        with r_inline_plot(width=300):
            R(
                """
            plotEigengeneNetworks(PCvalues, "Eigengene adjacency heatmap", 
                                marDendro = c(3,3,2,4),
                                marHeatmap = c(3,4,2,2), plotDendrograms = T, 
                                xLabelsAngle = 90)
            """
            )

        # save results
        logger.info("save results")
        R(f"load(paste0(dir_result, '/{jobid}_TOM_block.1.rda'), verbose=T)")

        R(
            f"""
        probes = colnames(datExpr)
        TOM <- as.matrix(consTomDS)
        dimnames(TOM) <- list(probes, probes)

        cyt = exportNetworkToCytoscape(TOM,
                    weighted = TRUE, threshold = 0.1,
                    nodeNames = probes, nodeAttr = moduleColors)
        """
        )

        ad_meta.obsm["eigengene"] = r2py(R("meInfo"))
        ad_meta.obsm["eigengene"].drop(columns="SampleID", inplace=True)
        ad_meta.varm["KME"] = r2py(R("geneInfo"))
        ad_meta.varp["TOM"] = r2py(R("TOM"))

        cyt = R("cyt")
        df_edge = r2py(rGet(cyt, "$edgeData"))
        df_node = r2py(rGet(cyt, "$nodeData"))

        dt_cyt = {"node": df_node, "edge": df_edge}
        ad_meta.uns["cyt"] = dt_cyt
    return ad_meta

def so2ad(
    so, dir_tmp=None, libPath_R=None, path_R=None, obsReParse=True, verbose=True
) -> sc.AnnData:
    import sh

    if not libPath_R:
        libPath_R = settings.seuratDisk_rLibPath
    if not path_R:
        path_R = settings.seuratDisk_rPath
    # R('.libPaths')(libPath_R)
    # seuratDisk = importr("SeuratDisk")
    if not dir_tmp:
        dir_tmp_ = TemporaryDirectory()
        dir_tmp = dir_tmp_.name
    path_h5ad = f"{dir_tmp}/temp.h5ad"
    path_h5so = f"{dir_tmp}/temp.h5seurat"
    path_rds = f"{dir_tmp}/temp.rds"

    R.saveRDS(so, path_rds)

    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); so <- readRDS('{path_rds}'); SaveH5Seurat(so, '{path_h5so}')",
    ]
    if verbose:
        sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    # seuratDisk.SaveH5Seurat(so, path_h5so, overwrite=True)

    h5so = h5py.File(path_h5so, "r+")
    ls_assays = h5so["/assays"].keys()
    for assay in ls_assays:
        ls_keys = h5so[f"/assays/{assay}"].keys()
        ls_slots = [x for x in ls_keys if x in ["counts", "data", "scale.data"]]
        ls_slots = [x for x in h5so[f"/assays/{assay}"] if x in ls_slots]
        for slot in ls_slots:
            if slot != "scale.data":
                h5so[f"/assays/{assay}_{slot}/data"] = h5so[f"/assays/{assay}/{slot}"]
                h5so[f"/assays/{assay}_{slot}/features"] = h5so[
                    f"/assays/{assay}/features"
                ]
                # h5so[f"/assays/{assay}_{slot}/misc"] = h5so[f"/assays/{assay}/misc"]
            else:
                h5so[f"/assays/{assay}_{slot}/scale.data"] = h5so[
                    f"/assays/{assay}/{slot}"
                ]
                h5so[f"/assays/{assay}_{slot}/data"] = h5so[f"/assays/{assay}/{slot}"]
                h5so[f"/assays/{assay}_{slot}/features"] = h5so[
                    f"/assays/{assay}/features"
                ]
                # h5so[f"/assays/{assay}_{slot}/misc"] = h5so[f"/assays/{assay}/misc"]
                h5so[f"/assays/{assay}_{slot}/scaled.features"] = h5so[
                    f"/assays/{assay}/scaled.features"
                ]
    h5so.close()

    # seuratDisk.Convert(path_h5so, dest="h5ad", overwrite=True)
    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); Convert('{path_h5so}', dest='h5ad', overwrite=T)",
    ]
    if verbose:
        sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    with h5py.File(path_h5ad, "a") as h5ad:
        if "raw" in h5ad.keys():
            del h5ad["raw"]
    ad = sc.read_h5ad(path_h5ad)
    if obsReParse:
        df_obs = r2py(so.slots["meta.data"])
        df_obs = df_obs.combine_first(ad.obs)
        ad.obs = df_obs.copy()
    return ad

def ad2so(
    ad,
    layer="raw",
    ls_obs=None,
    ls_var=None,
    ls_obsm=None,
    dir_tmp=None,
    dataLayer=None,
    scaleLayer=None,
    lightMode=False,
    renv=None,
    path_R=None,
    libPath_R=None,
    verbose=0,
):
    """
    anndata to seuratObject.

    obsm:
        matrix which's name does not start with 'X_' will be discarded.
    uns:
        discarded
    obsp:
        only keep 'distances'
    varp:
        discarded
    lightMode:
        if True, obsm/varm/obsp info will be discarded.

    layer must be raw.
    """
    import sh
    import scipy.sparse as ss

    if not path_R:
        path_R = settings.seuratDisk_rPath
    if not libPath_R:
        libPath_R = settings.seuratDisk_rLibPath

    # ad = ad.copy() # workaround `memoory not mapped` error
    # R('.libPaths')(libPath_R)
    # seuratDisk = importr("SeuratDisk")
    if renv is None:
        renv = ro.Environment()

    if not dir_tmp:
        dir_tmp_ = TemporaryDirectory()
        dir_tmp = dir_tmp_.name
    path_h5ad = f"{dir_tmp}/temp.h5ad"
    path_h5so = f"{dir_tmp}/temp.h5seurat"
    path_rds = f"{dir_tmp}/temp.rds"
    if lightMode:
        ad_partial = sc.AnnData(
            ad.layers[layer].copy(),
            obs=ad.obs,
            var=ad.var,
        )
    else:
        ad_partial = sc.AnnData(
            ad.layers[layer].copy(),
            obs=ad.obs,
            var=ad.var,
            obsm=ad.obsm,
            varm=ad.varm,
            obsp=ad.obsp,
        )
    if not ls_obs is None:
        if isinstance(ls_obs, str):
            ls_obs = [ls_obs]
        ad_partial.obs = ad_partial.obs[ls_obs]
    if not ls_var is None:
        if isinstance(ls_var, str):
            ls_var = [ls_var]
        ad_partial.var = ad_partial.var[ls_var]
    if not ls_obsm is None:
        if isinstance(ls_obsm, str):
            ls_obsm = [ls_obsm]
        ls_rm = []
        for _obsm in ad_partial.obsm.keys():
            if not _obsm in ls_obsm:
                ls_rm.append(_obsm)
        for _obsm in ls_rm:
            del ad_partial.obsm[_obsm]

    ad_partial.X = ss.csr_matrix(
        ad_partial.X
    )  # workaround https://github.com/satijalab/seurat/issues/2374
    _ls = []
    for key in ad_partial.obsm:
        if not key.startswith("X_"):
            _ls.append(key)
    for key in _ls:
        del ad_partial.obsm[key]

    # workaround `memoory not mapped` error
    df_var = ad_partial.var
    ad_partial.var = ad_partial.var[[]]
    ad_partial.raw = ad_partial
    ad_partial.var = df_var

    sc.pp.normalize_total(ad_partial, 1e4)
    sc.pp.log1p(ad_partial)

    ad_partial.write(path_h5ad)

    h5 = h5py.File(path_h5ad, "r+")
    if "obsp/distances" in h5:
        h5["/uns/neighbors/distances"] = h5["/obsp/distances"]
        h5["/uns/neighbors/params/method"] = "nn"

    h5.close()

    # seuratDisk.Convert(path_h5ad, dest="h5Seurat", overwrite=True)
    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); Convert('{path_h5ad}', dest='h5Seurat', overwrite=T); so <- LoadH5Seurat('{path_h5so}'); saveRDS(so, '{path_rds}')",
    ]
    if verbose:
        cmd = sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        cmd = sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    # for x in sh.Command(path_R)(*ls_cmd, _err_to_out=True, _iter=True):
    #     print(x.rstrip())
    so = R.readRDS(path_rds)

    if dataLayer:
        with ro.local_context(renv) as rlc:
            _ad = sc.AnnData(X=ss.csr_matrix(ad.shape), obs=ad.obs[[]], var=ad.var[[]])
            _ad.layers["data"] = ad.layers[dataLayer].copy()
            _se = py2r(_ad)
            rlc["se"] = _se
            rlc["so"] = so
            R(
                """so <- SetAssayData(so, slot = 'data', new.data = assay(se, 'data')) """
            )
            so = rlc["so"]
    if scaleLayer:
        with ro.local_context(renv) as rlc:
            _ad = sc.AnnData(X=ss.csr_matrix(ad.shape), obs=ad.obs[[]], var=ad.var[[]])
            _ad.layers["scale.data"] = ad.layers[scaleLayer].copy()
            _se = py2r(_ad)
            rlc["se"] = _se
            rlc["so"] = so
            R(
                """so <- SetAssayData(so, slot = 'scale.data', new.data = assay(se, 'scale.data')) """
            )
            so = rlc["so"]

    return so