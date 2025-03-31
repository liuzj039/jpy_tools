from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

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
from tempfile import NamedTemporaryFile
import collections
from xarray import corr
import sys
import snapatac2 as snap
from . import basic
from ..otherTools import F

def addSctAssayToSeuratObject(so, ad, ls_feature=[], layer='raw', setDefaultAssay=True):
    '''The function `addSctAssayToSeuratObject` adds a SCT assay to a Seurat object.

    Parameters
    ----------
    so
        The parameter "so" is a Seurat object, which is a data structure used in the Seurat package in R for single-cell RNA sequencing (scRNA-seq) analysis. It contains the scRNA-seq data along with various annotations and analysis results.
    ad
        anndata
    ls_feature
        The parameter `ls_feature` is a list of features (genes) that will be used for creating the SCT assay object. If `ls_feature` is not provided, it will default to using the highly variable genes from the Seurat object (`ad.var.loc[lambda _: _.highly_variable].
    layer, optional
        The 'layer' parameter is an optional parameter that specifies the layer used for SCT.

    Returns
    -------
        the Seurat object `so` with the added SCT assay.

    '''
    import pickle
    from . import normalize
    from ..rTools import py2r, r2py
    from ..otherTools import F

    basic.testAllCountIsInt(ad, layer)
    assert 'sct_vst_pickle' in ad.uns, "sct_vst_pickle not found in adata.uns"
    assert 'sct_clip_range' in ad.uns, "sct_clip_range not found in adata.layers"
    assert 'SCT_data' in ad.obsm
    import rpy2.robjects as ro
    R = ro.r

    vst_out = pickle.loads(eval(ad.uns['sct_vst_pickle']))
    arR_sctData = py2r(ad.obsm['SCT_data'].T)
    arR_sctData = R("""\(x,y,z){
    colnames(x) <- y
    rownames(x) <- z
    x
    }
    """)(arR_sctData, R.c(*ad.obs.index), R.c(*ad.uns['SCT_data_features']))

    assay_out = R.CreateAssayObject(counts=arR_sctData, check_matrix=False)
    if not ls_feature:
        ls_hvgFeatures = ad.var.sort_values('highly_variable_rank').loc[lambda _: _.highly_variable].index.to_list()
    normalize.getSctResiduals(ad, ls_hvgFeatures)
    dfR_sctRes = py2r(ad.obsm['sct_residual'].loc[:, ls_hvgFeatures].T)
    dfR_sctRes = R('as.matrix')(dfR_sctRes)
    dfR_sctRes = R("""\(x,y,z){
    colnames(x) <- y
    rownames(x) <- z
    x
    }
    """)(dfR_sctRes, R.c(*ad.obs.index), R.c(*ls_hvgFeatures))

    assay_out = R("""\(assay.out, features, scale.data, vst.out){
    VariableFeatures(object=assay.out) <- features
    assay.out <- SetAssayData(
        object = assay.out,
        slot = 'data',
        new.data = log1p(x = GetAssayData(object = assay.out, slot = 'counts'))
    )
    assay.out <- SetAssayData(
        object = assay.out,
        slot = 'scale.data',
        new.data = scale.data
    )
    Misc(object = assay.out, slot = 'vst.out') <- vst.out
    assay.out <- as(object = assay.out, Class = "SCTAssay")
    assay.out@assay.orig <- 'RNA'
    slot(object = slot(object = assay.out, name = "SCTModel.list")[[1]], name = "umi.assay") <- 'RNA'  
    assay.out
    }
    """)(assay_out, ls_hvgFeatures, dfR_sctRes, vst_out)
    so = R("""\(x, y){
    x[['SCT']] <- y
    x
    }
    """)(so, assay_out)
    if setDefaultAssay:
        so = R("""\(x, y){
        DefaultAssay(object = x) <- y
        x
        }
        """)(so, 'SCT')
    return so




def starsolo_transferMtxToH5ad(starsoloMtxDir, force=False) -> sc.AnnData:
    import scipy.sparse as ss
    import glob
    import os

    starsoloMtxDir = starsoloMtxDir.rstrip("/") + "/"
    path_out = f"{starsoloMtxDir}adata.h5ad"
    if (os.path.exists(path_out)) and (not force):
        adata = sc.read_h5ad(path_out)
    else:
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

        adata = sc.AnnData(
            X=ss.csr_matrix((len(df_barcode), len(df_feature))),
            obs=df_barcode,
            var=df_feature,
        )
        for mtxName, mtxPath in zip(ls_mtxName, ls_allMtx):
            adata.layers[mtxName] = sc.read_mtx(mtxPath).X.T
            logger.info(f"read {mtxName} done")
        adata.write_h5ad(path_out)
    return adata


def scWGCNA(
    ad: sc.AnnData,
    layer: str,
    dir_result: str,
    jobid: str,
    ls_hvgGene: List[str],
    minModuleSize: int = 50,
    deepSplit: float = 4,
    mergeCutHeight: float = 0.2,
    threads: int = 1,
    softPower: Optional[int] = None,
    renv=None,
    rawLayer="raw",
) -> sc.AnnData:
    """
    perform scWGCNA

    Parameters
    ----------
    ad : sc.AnnData
    layer : str
        must be meta log-normalized
    dir_result : str
        store `blockwiseConsensusModules` results
    jobid : str
        determine path used for storing results of `blockwiseConsensusModules`
    minModuleSize : int, optional
        by default 50
    deepSplit: float
        a larger `deepSplit` with a larger number of modules, by default 4
    mergeCutHeight: float
        a smaller `mergeCutHeight` with a larger number of modules, by default 0.2
    threads : int, optional
        by default 1
    softPower : Optional[int], optional
        by default None

    Returns
    -------
    sc.AnnData
        with WGCNA results
    """
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py, r_inline_plot, rHelp, trl, rGet, rSet, ad2so, so2ad
    import os
    import warnings
    warnings.warn("scWGCNA is deprecated, use geneEnrichInfo's scWGCNA instead", DeprecationWarning)

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

    rlc = ro.local_context(renv)
    rlc.__enter__()

    renv["minModuleSize"] = minModuleSize
    renv["deepSplit"] = deepSplit
    renv["mergeCutHeight"] = mergeCutHeight
    renv["maxBlockSize"] = len(ls_hvgGene)
    renv["jobid"] = jobid
    renv["dir_result"] = dir_result

    ad_meta = ad[:, ls_hvgGene].copy()
    ad_meta.var.index = ad_meta.var.index.map(lambda x: x.replace("_", "-"))
    so = ad2so(
        ad_meta,
        layer=rawLayer,
        ls_obs=[],
        ls_var=[],
        lightMode=True,
        dataLayer=layer,
    )
    renv["so"] = so

    R(
        """
    datExpr <- as.data.frame(GetAssayData(so, assay='RNA', slot='data'))
    datExpr <- as.data.frame(t(datExpr))
    datExpr <- datExpr[,goodGenes(datExpr)]

    lsR_useGene = colnames(datExpr)
    """
    )

    if not softPower:
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
        softPower = int(input("Soft Power"))
    renv["softPower"] = softPower

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

    ad_meta = ad_meta[:, list(renv["lsR_useGene"])]
    ad_meta.obsm["eigengene"] = r2py(R("meInfo"))
    ad_meta.obsm["eigengene"].drop(columns="SampleID", inplace=True)
    ad_meta.varm["KME"] = r2py(R("geneInfo"))
    ad_meta.varp["TOM"] = r2py(R("TOM"))

    # cyt = R("cyt")
    # df_edge = r2py(rGet(cyt, "$edgeData"))
    # df_node = r2py(rGet(cyt, "$nodeData"))

    # dt_cyt = {"node": df_node, "edge": df_edge}
    # ad_meta.uns["cyt"] = dt_cyt

    rlc.__exit__(None, None, None)
    ro.r.gc()
    return ad_meta


def clusterCorrelation(
    ad, cluster: str, layer="raw", method: Literal["spearman", "pearson"] = "spearman"
) -> sc.AnnData:
    """
    Correlation clustering.

    Parameters
    ----------
    ad : :class:`~anndata.AnnData`
        Annotated data matrix.
    layer : `str`
        Layer to cluster. Default: `raw`.
    cluster : `str`
        Cluster layer name.
    method : `str`, spearman | pearson
        Clustering method.

    Returns
    -------
    :class:`~anndata.AnnData`
        Annotated data matrix.
    """
    from scipy.stats import mstats

    ad = ad.copy()
    ad.layers["normalize"] = ad.layers[layer].copy()
    sc.pp.normalize_total(ad, 1e4, layer="normalize")

    ad_psedoBulk = sc.AnnData(
        ad.to_df("normalize").groupby(ad.obs[cluster]).agg("mean")
    )
    sc.pp.log1p(ad_psedoBulk)

    ar_mtx = ad_psedoBulk.X.copy()

    if method == "spearman":
        ar_mtx = mstats.rankdata(ar_mtx, axis=1)
    elif method == "pearson":
        pass
    else:
        raise ValueError(f"method should be spearman or pearson, but got {method}")

    mtx_corr = np.corrcoef(ar_mtx, ar_mtx)[-len(ar_mtx) :, : len(ar_mtx)]
    ad_psedoBulk.obsp[f"corr_{method}"] = mtx_corr
    return ad_psedoBulk


def addMetaCellLayerGroup(
    ad: sc.AnnData,
    layer: str,
    group: str,
    ls_hvgGene: List[str],
    n_neighbors=50,
    boolConnectivity=False,
) -> sc.AnnData:
    """
    Add meta-cell layer to group.

    Parameters
    ----------
    ad : :class:`~anndata.AnnData`
        Annotated data matrix.
    layer : `str`
    group : `str`
        Group name.
    """
    ad = ad.copy()
    dt_ad = {}
    for sample, _ad in basic.splitAdata(ad, group, needName=True):
        _ad.var["highly_variable"] = _ad.var.index.isin(ls_hvgGene)
        sc.tl.pca(_ad)
        addMetaCellLayer(
            _ad,
            layer=layer,
            obsm="X_pca",
            n_neighbors=n_neighbors,
            boolConnectivity=boolConnectivity,
        )
        dt_ad[sample] = _ad
    ad = sc.concat(dt_ad)[ad.obs.index]
    return ad


def addMetaCellLayer(
    ad: sc.AnnData,
    layer: str,
    obsm: str,
    n_neighbors=50,
    obsp=None,
    boolConnectivity=False,
):
    """
    Add meta-cell layer.
    """
    if not obsp:
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep=obsm, key_added="meta")
        obsp = "meta_connectivities"
    if boolConnectivity:
        ar_neighbors = np.eye(ad.shape[0]) + (ad.obsp[obsp] > 0)
        ad.layers[f"{layer}_meta"] = (ar_neighbors @ ad.layers[layer]) / (
            n_neighbors + 1
        )
    else:
        ar_connect = np.eye(ad.shape[0]) + ad.obsp[obsp].A
        ar_neighbors = ar_connect * (1 / ar_connect.sum(0))
        ad.layers[f"{layer}_meta"] = ar_neighbors @ ad.layers[layer]


def getAlignmentScore(ad, batchKey, obsm, knn=20, plot=True, **dt_heatmapKwargs):
    import scanorama

    ls_sample = []
    curSample = None
    for x in ad.obs[batchKey]:
        if not curSample:
            curSample = x
            ls_sample.append(curSample)
        if x == curSample:
            continue
        else:
            if x in ls_sample:
                assert False, "Detected non-contiguous batches."
            else:
                curSample = x
                ls_sample.append(curSample)

    dt_index = (
        ad.obs.groupby(batchKey, sort=False)
        .apply(lambda df: df.index.to_list())
        .to_dict()
    )
    ar_alignment, _, _ = scanorama.find_alignments_table(
        [ad[x].obsm[obsm] for x in dt_index.values()], knn=knn, verbose=0
    )

    ar_alignmentProcessed = np.zeros((len(dt_index), len(dt_index)))
    for x, y in ar_alignment.keys():
        ar_alignmentProcessed[x, y] += ar_alignment[(x, y)]

    df_alignmentProcessed = pd.DataFrame(
        ar_alignmentProcessed + ar_alignmentProcessed.T + np.eye(len(dt_index)),
        index=dt_index.keys(),
        columns=dt_index.keys(),
    )
    df_alignmentProcessed = df_alignmentProcessed.reindex(ad.obs[batchKey].cat.categories).reindex(ad.obs[batchKey].cat.categories, axis=1)
    if plot:
        ax = sns.heatmap(
            df_alignmentProcessed, cmap="Reds", annot=True, **dt_heatmapKwargs
        )
        plt.title(batchKey)
        return df_alignmentProcessed, ax
    else:
        return df_alignmentProcessed


def getClusterRobustness_reclustering(
    ad,
    group,
    fraction=0.8,
    times=10,
    seed=0,
    fc_clustering=None,
    dtFc_evaluation: Dict[str, Callable] = None,
    ls_obsForCalcARI=None,
    plot=True,
) -> Union[pd.DataFrame, list]:
    """
    Calculate cluster robustness based on ARI.

    Parameters
    ----------
    ad : :class:`~anndata.AnnData`
        Annotated data matrix.
    group : `str`
        Group name.
    fracion : `float`
        Fraction of cells to use for clustering.
    times : `int`
        Number of times to run clustering.
    seed : `int`
        Random seed.
    fc_clustering : `str`
        clustering function. input must be anndata and output must be pd.Series.
    lsFc_evaluation:
        evaluation function. input must be (List, List) and output must be float.
    ls_obsForCalcARI : `list`
        Observation names for calculating ARI.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame]
        recutering results and robustness score.
    """
    from sklearn import metrics

    def _getClusterResult(ad):
        ad.X = ad.layers["normalize_log"].copy()
        sc.tl.pca(ad)
        sc.pp.neighbors(ad)
        sc.tl.leiden(ad)
        return ad.obs["leiden"]

    if not fc_clustering:
        fc_clustering = _getClusterResult
    if not ls_obsForCalcARI:
        ls_obsForCalcARI = ad.obs.index.to_list()
    if not dtFc_evaluation:
        dtFc_evaluation = {
            "ari": metrics.adjusted_rand_score,
            "ami": metrics.adjusted_mutual_info_score,
            "fmi": metrics.fowlkes_mallows_score,
        }

    df_clusterRes = pd.DataFrame(index=ad.obs.index)
    for i in tqdm(range(times), desc="reclustering"):
        _ad = sc.pp.subsample(ad, fraction=fraction, copy=True, random_state=seed + i)
        ls_trueLabel = _ad.obs[group]
        ls_predLabel = fc_clustering(_ad)

        df_clusterRes = df_clusterRes.assign(
            **{f"{i}_org": ls_trueLabel, f"{i}_new": ls_predLabel}
        )

        ls_trueLabel = ls_trueLabel.loc[
            [x for x in ls_trueLabel.index if x in ls_obsForCalcARI]
        ]
        ls_predLabel = ls_predLabel.loc[
            [x for x in ls_predLabel.index if x in ls_obsForCalcARI]
        ]
        assert (ls_trueLabel.index == ls_predLabel.index).all(), "label is not equal"

    ls_ari = []
    for i in tqdm(range(times), desc="calculating ARI"):
        df_clusterOnce = df_clusterRes.filter(regex=rf"\b{i}_").dropna()
        dt_clusterOnce = (
            df_clusterOnce.groupby(f"{i}_org")[f"{i}_new"].agg(list).to_dict()
        )

        for fcName, fc_evaluation in dtFc_evaluation.items():
            dt_ariOnce = {
                x: fc_evaluation([x] * len(y), y) for x, y in dt_clusterOnce.items()
            }
            dt_ariOnce["all"] = fc_evaluation(
                df_clusterRes.iloc[:, 0], df_clusterRes.iloc[:, 1]
            )
            sr_ariOnce = pd.Series(dt_ariOnce).rename(f"{i}_{fcName}")
            ls_ari.append(sr_ariOnce)
    df_ari = pd.DataFrame(ls_ari)

    if plot:
        df_forPlot = df_ari.stack().rename_axis(["group", "cluster"]).rename("score").reset_index().assign(
            time=lambda df: df["group"].str.split("_").str[0], method=lambda df: df["group"].str.split("_").str[1]
        )
        axs = sns.FacetGrid(df_forPlot, col = 'method', col_wrap=5)
        axs.map_dataframe(sns.boxplot, x='cluster', y='score')
        plt.show()

    return df_clusterRes, df_ari

def saveAdToLmdb(ad, outputPath, ls_saveObsKey, ls_pseudoBulkUse, forceDense, batchSize=100):
    '''It takes an AnnData object, and saves it to a directory in LMDB format

    Parameters
    ----------
    ad
        AnnData object
    outputPath
        the path to save the output files
    ls_saveObsKey
        a list of keys in ad.obs that you want to save.
    ls_pseudoBulkUse
        a list of columns in ad.obs that will be used to create the pseudo-bulk data.
    
    Notes
    ----------
    sparse matrix will be stored by following scripts:
        a = ss.coo_array(a)
        value = pickle.dumps((a.data, a.row, a.col, a.shape))
    The saved data will be loaded by following scripts:
        data, row, col, shape = pickle.loads(value)
        a = ss.coo_matrix((data, (row, col)), shape=shape)
    '''
    import lmdb
    import tqdm
    import pickle
    import gc
    from . import geneEnrichInfo, basic

    # save gene (cell)
    env = lmdb.open(f"{outputPath}/cell/", map_size=1099511627776)
    txn = env.begin(write=True)

    value = pickle.dumps(ad.obsm['X_umap'][:, 0])
    txn.put(key='x'.encode(), value=value)
    logger.info(f"save UMAP_1 to x")

    value = pickle.dumps(ad.obsm['X_umap'][:, 1])
    txn.put(key='y'.encode(), value=value)
    logger.info(f"save UMAP_2 to y")

    value = pickle.dumps(ad.var.index.values)
    txn.put(key='all_gene'.encode(), value=value)
    logger.info(f"save all gene's names to all_gene")

    value = pickle.dumps(ad.obs.index.values)
    txn.put(key='all_cell'.encode(), value=value)
    logger.info(f"save all cell's names to all_cell")

    for key in ls_saveObsKey:
        value = pickle.dumps(ad.obs[key].values)
        txn.put(key=key.encode(), value=value)
        logger.info(f"save {key} to {key}")

    for i, gene in tqdm.tqdm(enumerate(ad.var.index), total=len(ad.var.index)):
        if forceDense:
            value = pickle.dumps(ad[:, gene].layers['normalize_log'].A.reshape(-1))
        else:
            coo_exp = ad[:, gene].layers['normalize_log'].tocoo()
            value = pickle.dumps((coo_exp.data, coo_exp.row, coo_exp.col, coo_exp.shape))
        txn.put(key=gene.encode(), value=value)
        if i % batchSize == 0:
            txn.commit()
            txn = env.begin(write=True)
            gc.collect()

    txn.commit()
    env.close()

    ad_bulk = geneEnrichInfo._mergeData(ad, ls_pseudoBulkUse)
    basic.initLayer(ad_bulk, total=1e6)

    # save gene (pseudo-bulk)
    env = lmdb.open(f"{outputPath}/pseudo_bulk/", map_size=1099511627776)
    txn = env.begin(write=True)

    value = pickle.dumps(ad_bulk.var.index.values)
    txn.put(key='all_gene'.encode(), value=value)
    logger.info(f"save all gene's names to all_gene")

    value = pickle.dumps(ad_bulk.obs.index.values)
    txn.put(key='all_cell'.encode(), value=value)
    logger.info(f"save all cell's names to all_cell")

    for key in ls_pseudoBulkUse:
        value = pickle.dumps(ad_bulk.obs[key].values)
        txn.put(key=key.encode(), value=value)

    for gene in tqdm.tqdm(ad_bulk.var.index):
        value = pickle.dumps(ad_bulk[:, gene].layers['normalize_log'].reshape(-1))
        txn.put(key=gene.encode(), value=value)

    txn.commit()
    env.close()


def subsetBackedAd(ad:snap.AnnData, ls_obs, ls_var) -> sc.AnnData:
    from tempfile import NamedTemporaryFile
    tempFile = NamedTemporaryFile()

    if isinstance(ls_obs, str):
        ls_obs = [ls_obs]
    if isinstance(ls_var, str):
        ls_var = [ls_var]
        
    ad1 = ad.subset(ls_obs, ls_var, tempFile.name)
    ad_subset = ad1.to_memory()
    ad1.close()
    tempFile.close()
    return ad_subset

def clusterUseScshc(ad: sc.AnnData, clusterKey:str, batchKey:Optional[str]=None, layer:str='raw', **kwargs):
    '''The function `clusterUseScshc` clusters single-cell RNA sequencing data using the scSHC algorithm and assigns the cluster labels to a specified key in the AnnData object.

    Parameters
    ----------
    ad : sc.AnnData
        Anndata object containing the single-cell RNA-seq data.
    clusterKey : str
        The `clusterKey` parameter is a string that specifies the key in the `ad.obs` dataframe where the cluster labels will be stored.
    batchKey : Optional[str]
        The `batchKey` parameter is an optional parameter that specifies the key in the `ad.obs` dataframe that contains the batch information for each cell. If provided, the clustering algorithm will take into account the batch information when performing clustering. If not provided, the clustering algorithm will ignore batch information.
    layer : str, optional
        The `layer` parameter specifies the layer of the AnnData object that contains the count data. By default, it is set to 'raw'.

    '''
    from joblib import Parallel, delayed
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py
    R = ro.r

    def fc(**kwargs):
        scSHC = importr("scSHC")
        rBase = importr("base")
        rBase.set_seed(39)
        lsR_clusterRes, r_nodeInfo = scSHC.scSHC(**kwargs)
        return lsR_clusterRes, r_nodeInfo
    
    scSHC = importr("scSHC")
    ssR_count = ad.layers[layer] >> F(py2r)
    ssR_count = R("""\(x, rN, cN) {
        rownames(x) = rN
        colnames(x) = cN
        t(x)
    }""")(ssR_count, ad.obs.index >> F(lambda _: R.c(*_)), ad.var.index >> F(lambda _: R.c(*_)))
    if batchKey:
        batch = ad.obs[batchKey].values >> F(py2r)
    else:
        batch = R("NULL")

    dt_kwargs = dict(
        data=ssR_count, batch=batch, **kwargs
    )
    lsR_clusterRes, r_nodeInfo = Parallel(2)(delayed(fc)(**x) for x in [dt_kwargs])[0] # seems uncompatable with other imported packages, bypass use another process here
    print(r_nodeInfo)
    ad.obs[clusterKey] = list(lsR_clusterRes)

def mergeClusterUseScshc(ad: sc.AnnData, ls_hvg: List[str], clusterKey:str, newClusterKey:str, batchKey:Optional[str]=None, layer:str='raw', **kwargs):
    '''The function `mergeClusterUseScshc` merges clusters in an AnnData object using the scSHC package in R.

    Parameters
    ----------
    ad : sc.AnnData
        The `ad` parameter is an AnnData object, which is a data structure used in single-cell RNA sequencing analysis. It contains the gene expression data, cell metadata, and other information.
    ls_hvg : List[str]
        The `ls_hvg` parameter is a list of genes that are considered highly variable genes (HVGs). These genes are typically used to identify cell clusters in single-cell RNA sequencing data.
    clusterKey : str
        The `clusterKey` parameter is a string that represents the key in the `ad.obs` object of the AnnData object that contains the cluster labels for each cell.
    newClusterKey : str
        The `newClusterKey` parameter is a string that represents the key for the new cluster assignment in the `ad.obs` object. This parameter is used to store the new cluster assignments after merging clusters using the scSHC algorithm.
    batchKey : Optional[str]
        The `batchKey` parameter is an optional parameter that specifies the key in the `ad.obs` dataframe that contains the batch information for each cell. If provided, the batch information will be used in the clustering analysis. If not provided, the clustering analysis will be performed without considering batch information.
    layer : str, optional
        The `layer` parameter specifies the layer of the AnnData object that contains the count data. It can be set to 'raw' or any other valid layer name.

    '''
    from joblib import Parallel, delayed
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py
    R = ro.r

    def fc(**kwargs):
        scSHC = importr("scSHC")
        rBase = importr("base")
        rBase.set_seed(39)
        lsR_clusterRes, r_nodeInfo = scSHC.testClusters(**kwargs)
        return lsR_clusterRes, r_nodeInfo
    
    scSHC = importr("scSHC")
    ssR_count = ad.layers[layer] >> F(py2r)
    ssR_count = R("""\(x, rN, cN) {
        rownames(x) = rN
        colnames(x) = cN
        t(x)
    }""")(ssR_count, ad.obs.index >> F(lambda _: R.c(*_)), ad.var.index >> F(lambda _: R.c(*_)))
    if batchKey:
        batch = ad.obs[batchKey].values >> F(py2r)
    else:
        batch = R("NULL")
    cluster=ad.obs[clusterKey].values >> F(py2r)
    if not ls_hvg is None:
        var_genes = R.c(*ls_hvg)
    else:
        var_genes = R("NULL")
    dt_kwargs = dict(
        data=ssR_count, batch=batch, cluster=cluster, var_genes=var_genes, **kwargs
    )
    lsR_clusterRes, r_nodeInfo = Parallel(2)(delayed(fc)(**x) for x in [dt_kwargs])[0] # seems uncompatable with other imported packages, bypass use another process here
    print(r_nodeInfo)
    ad.obs[newClusterKey] = list(lsR_clusterRes) >> F(map, lambda _: _.split('new')[-1]) >> F(list)

def calucatePvalueForEachSplitUseScshc(
        ad: sc.AnnData, ls_hvg: List[str], clusterKey: str, batchKey: Optional[str]=None, layer: str='raw', 
        alpha: float=0.05, posthoc: bool=True, num_PCs: int=30, cores: int=12, rCores: int = 1
    ):
    '''The function `calucatePvalueForEachSplitUseScshc` calculates p-values for each split in a hierarchical clustering dendrogram using the scSHC package in R.

    Parameters
    ----------
    ad : sc.AnnData
        The `ad` parameter is an AnnData object, which is a data structure commonly used in single-cell RNA sequencing (scRNA-seq) analysis. It contains the gene expression data and associated metadata for each cell.
    ls_hvg : List[str]
        A list of highly variable genes (HVGs) that will be used for calculating the p-values for each split. These genes should be selected based on their high variability across cells.
    clusterKey : str
        The `clusterKey` parameter is a string that specifies the key in the `ad.obs` dataframe that contains the cluster labels for each cell.
    batchKey : Optional[str]
        The `batchKey` parameter is an optional parameter that specifies the key in the `ad.obs` dataframe that represents the batch information. If provided, the function will perform batch correction before calculating the p-values for each split. If not provided, a temporary batch key will be created with all values set
    layer : str, optional
        The `layer` parameter specifies the layer of the AnnData object to use for the analysis. It can be set to 'raw' or any other layer present in the AnnData object.
    alpha : float
        The alpha parameter is the significance level used for hypothesis testing. It determines the threshold below which the p-value is considered statistically significant. The default value is 0.05, which corresponds to a 5% significance level.
    posthoc : bool, optional
        The `posthoc` parameter determines whether to perform post-hoc analysis after calculating the p-values for each split. If `posthoc` is set to `True`, post-hoc analysis will be performed. If `posthoc` is set to `False`, post-hoc analysis will not be
    num_PCs : int, optional
        The parameter `num_PCs` is the number of principal components to use for dimensionality reduction in the scSHC algorithm. It determines the number of dimensions in which the data will be projected before clustering.
    cores : int, optional
        The "cores" parameter specifies the number of CPU cores to use for parallel processing. It determines how many parallel jobs can be executed simultaneously.

    Returns
    -------
        a dictionary `dt_p` which contains the p-values for each split in the hierarchical clustering. The keys of the dictionary are tuples representing the left and right clusters resulting from each split, and the values are the corresponding p-values.

    '''
    from joblib import Parallel, delayed
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from ..rTools import py2r, r2py
    from .geneEnrichInfo import _mergeData
    from .basic import initLayer
    R = ro.r
    import scipy

    # get each node's element
    def getNodeContents(node, dt_preRenameCluster):
        if node.is_leaf():
            return dt_preRenameCluster[node.id],
        else:
            _ls = []
            _ls.extend(getNodeContents(node.left, dt_preRenameCluster))
            _ls.extend(getNodeContents(node.right, dt_preRenameCluster))
            return tuple(_ls)

    def getNodeLeftAndRight(node, dt_preRenameCluster):
        return getNodeContents(node.left, dt_preRenameCluster), getNodeContents(node.right, dt_preRenameCluster)

    def fc(**kwargs):
        scSHC = importr("scSHC")
        rBase = importr("base")
        rBase.set_seed(39)
        calc = kwargs.pop('calc')
        nodeCellNum = kwargs.pop('node_cell_num')
        totalCellNum = kwargs.pop('total_cell_num')
        if calc:
            pval = scSHC.test_split(**kwargs)[0]
            pval = min(round(pval*(totalCellNum-1)/(nodeCellNum-1), 2), 1)
        else:
            pval = 1
        
        return pval

    scSHC = importr('scSHC')
    if posthoc:
        posthoc = R.T
    else:
        posthoc = R.F

    if batchKey is None:
        ad.obs['temp_batch'] = 'a'
    else:
        ad.obs['temp_batch'] = ad.obs[batchKey]

    ad_pseudobulk = _mergeData(ad, clusterKey, layer)
    initLayer(ad_pseudobulk, logbase=2)
    ad_pseudobulk.obs.index = range(ad_pseudobulk.shape[0])
    dt_preRenameCluster = ad_pseudobulk.obs[clusterKey].to_dict()
    X = ad_pseudobulk[:, ls_hvg].to_df(layer)
    X = X.apply(lambda _: _/_.sum(), axis=1)
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward', optimal_ordering=True)

    ls_label = ad_pseudobulk.obs[clusterKey].to_list()
    R1 = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=0, no_plot=True)
    dt_label = {leaf: ls_label[leaf] for leaf in R1["leaves"]}
    # print(dt_label)
    fig, ax = plt.subplots(figsize=(15, 5))
    dn = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=0, ax=ax, leaf_label_func=lambda x:dt_label[x])
    plt.show()

    nd_root, lsNd_sub = scipy.cluster.hierarchy.to_tree(linkage, rd=True)
    lsNd_sub = [x for x in lsNd_sub if not x.is_leaf()]
    ssR_count = py2r(ad.layers[layer])
    ssR_count = R("""\(x, rN, cN) {
        rownames(x) = rN
        colnames(x) = cN
        t(x)
    }""")(ssR_count, R.unlist(ad.obs.index.to_list()), R.unlist(ad.var.index.to_list()))

    arR_batch = R.unlist(ad.obs['temp_batch'].astype(str).to_list())
    arR_batch.names = R.unlist(ad.obs.index.to_list())

    ls_kwargs = []
    logger.info(f"start calculating p-value for each split")
    for _nd in lsNd_sub:
        ls_leftCluster, ls_rightCluster = getNodeLeftAndRight(_nd, dt_preRenameCluster)
        ad.obs['temp_split'] = np.select(
            [ad.obs[clusterKey].isin(ls_leftCluster), ad.obs[clusterKey].isin(ls_rightCluster)],
            ['left', 'right'],
            None
        )
        ls_keep = ad.obs.value_counts(['temp_batch', 'temp_split']).unstack().min(1).loc[lambda _: _ > 20].index.to_list()
        ls_ids1 = ad.obs.query("temp_split == 'left' & temp_batch in @ls_keep").index.to_list()
        ls_ids2 = ad.obs.query("temp_split == 'right' & temp_batch in @ls_keep").index.to_list()
        nodeCellNum = ad.obs.query("temp_split.notna()").shape[0]
        totalCellNum = ad.shape[1]
        alphaLevel = alpha * (nodeCellNum - 1) / (totalCellNum - 1)
        _dt = {}
        _dt['data'] = ssR_count
        _dt['ids1'] = R.unlist(ls_ids1)
        _dt['ids2'] = R.unlist(ls_ids2)
        _dt['var_genes'] = R.unlist(ls_hvg)
        _dt['num_PCs'] = num_PCs
        _dt['batch'] = arR_batch
        _dt['alpha_level'] = alphaLevel
        _dt['cores'] = rCores
        _dt['posthoc'] = posthoc
        _dt['node_cell_num'] = nodeCellNum
        _dt['total_cell_num'] = totalCellNum
        if len(ls_keep) == 0:
            _dt['calc'] = False
        else:
            _dt['calc'] = True
        ls_kwargs.append(_dt)
    ls_p = Parallel(cores)(delayed(fc)(**x) for x in ls_kwargs)
    ls_p = [x if isinstance(x, (float, int)) else x[0] for x in ls_p]
    dt_p = {}
    for _nd, _p in zip(lsNd_sub, ls_p):
        ls_leftCluster, ls_rightCluster = getNodeLeftAndRight(_nd, dt_preRenameCluster)
        dt_p[(ls_leftCluster, ls_rightCluster)] = _p

    ls_label = ad_pseudobulk.obs[clusterKey].to_list()
    R1 = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=0, no_plot=True)
    dt_label = {leaf: ls_label[leaf] for leaf in R1["leaves"]}
    fig, ax = plt.subplots(figsize=(15, 5))
    dn = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=0, ax=ax, leaf_label_func=lambda x:dt_label[x])
    ls_signPos = []
    for ls_x, ls_y in zip(dn['icoord'], dn['dcoord']):
        ls_signPos.append([(ls_x[0]+ls_x[-1]) / 2, ls_y[1]])
    ls_signPos = sorted(ls_signPos, key=lambda _: _[1])
    for (x, y), p in zip(ls_signPos, ls_p):
        if p < 0.001:
            sign = '***'
        elif p < 0.01:
            sign = '**'
        elif p < 0.05:
            sign = '*'
        else:
            sign = ''
        ax.text(x, y, sign, ha='center')

    plt.show()

    return dt_p, linkage

def clusteringAndCalculateShilouetteScore(
        ad:sc.AnnData, ls_res: List[float], obsm: Union[str, np.ndarray], clusterKey:str='leiden', subsample=None, metric='euclidean', show=True, check=True, pcs:int = 50, cores:int = 1, n_iterations=-1
    ) -> Dict[str, float]:
    '''The function performs clustering using the Leiden algorithm on an AnnData object and calculates the silhouette score for each clustering result.

    Parameters
    ----------
    ad : sc.AnnData
        The parameter `ad` is an AnnData object, which is a data structure commonly used in single-cell RNA sequencing (scRNA-seq) analysis. It contains the gene expression data and associated metadata for each cell.
    ls_res : List[float]
        A list of resolution values to use for the Leiden clustering algorithm.
    obsm : Union[str, np.ndarray]
        The parameter `obsm` is the name of the key in the `ad` object's `.obsm` attribute that contains the data matrix used for clustering. It can be either a string representing the key name or a numpy array containing the data matrix itself.
    subsample
        The `subsample` parameter is an optional parameter that specifies the fraction of cells to subsample from the input `ad` AnnData object. If provided, only a fraction of cells will be used for calculating the silhouette score. If not provided, all cells in the `ad` Ann
    metric, optional
        The `metric` parameter specifies the distance metric to be used for calculating pairwise distances between observations. The default value is 'euclidean', which calculates the Euclidean distance between two points. Other possible values include 'manhattan' for Manhattan distance, 'cosine' for cosine similarity, and many more

    Returns
    -------
        a dictionary where the keys are the resolution values from the input list `ls_res` and the values are the corresponding silhouette scores calculated using the Leiden clustering algorithm.

    '''
    import sklearn
    import tqdm
    import scipy.sparse as ss
    from joblib import Parallel, delayed
    if clusterKey in ad.obsm:
        ls_res = list(ad.obsm[clusterKey].columns)
        logger.info(f"clusterKey {clusterKey} already in ad.obsm, skip clustering")
        logger.info(f"used res: {ls_res}")
        if check:
            for x in ls_res:
                try:
                    float(x)
                except:
                    assert False, f"clusterKey {clusterKey} already in ad.obsm, but not all columns are float"
        for x in ls_res:
            if ad.obsm[clusterKey][x].dtype == 'category':
                pass
            else:
                logger.warning(f"{x} is not category, convert to category")
                ad.obsm[clusterKey][x] = ad.obsm[clusterKey][x].astype(str).astype('category')
    else:
        logger.info(f"clustering using leiden algorithm")
        # report used res
        logger.info(f"used res: {ls_res}")

        if cores == 1:
            lsDf = []
            for res in tqdm.tqdm(ls_res, desc="res"):
                sc.tl.leiden(ad, resolution=float(res), key_added=f"temp_{res}", n_iterations=n_iterations)
                lsDf.append(ad.obs[f"temp_{res}"])
                del(ad.obs[f"temp_{res}"])
        else:
            _ad = sc.AnnData(ss.csc_matrix(ad.shape), obs=ad.obs.copy(), var=ad.var.copy())
            _ad.obsp['connectivities'] = ad.obsp['connectivities'].copy()
            def fc(ad, res):
                sc.tl.leiden(ad, resolution=float(res), key_added=f"temp_{res}", obsp='connectivities')
                return ad.obs[f"temp_{res}"].copy()
            lsDf = Parallel(cores)(delayed(fc)(_ad, res) for res in tqdm.tqdm(ls_res, desc="res"))
        ad.obsm[clusterKey] = pd.concat(lsDf, axis=1).rename(columns=lambda _: _.split('temp_')[1]).sort_index(axis=1, key=lambda _: _.astype(float))
        
    if subsample:
        if subsample > 1:
            subsample = min(subsample / ad.shape[0], 1)
            logger.info(f"subsample > 1, convert to {subsample}")
        
        _ad = sc.pp.subsample(ad, fraction=subsample, copy=True)
    else:
        _ad = ad
    if isinstance(obsm, str):
        obsm = _ad.obsm[obsm]
    if obsm.shape[1] > pcs:
        obsm = obsm[:, :pcs]
    ar_dist = sklearn.metrics.pairwise_distances(obsm, metric=metric)
    dt_score = {}
    for res in tqdm.tqdm(ls_res, desc="silhouette_score"):
        ls_label =  _ad.obsm[clusterKey][str(res)]
        if len(set(ls_label)) == 1:
            dt_score[res] = 0
        else:
            dt_score[res] = sklearn.metrics.silhouette_score(ar_dist, _ad.obsm[clusterKey][str(res)], metric='precomputed')

    if show:
        import IPython
        p = (
            so.Plot(x=dt_score.keys() >> F(map, float) >> F(list), y=dt_score.values())
            .add(so.Dots())
            .add(so.Line())
            .scale()
        )
        IPython.display.display(p)

    return {x:dt_score[x] for x in sorted(dt_score.keys(), key=lambda _: dt_score[_], reverse=True)}


def sketchBasedLabelTransfer(
        ad_sketch:sc.AnnData, ad_full:sc.AnnData, sketchEmbedding:str, fullEmbedding:str, sketchLabels:Union[str, List[str]],
        metric:str='cosine', nNeighbors:int=30, minDist:float=0.3, sketchUmapKey:str='X_umap', fullUmapKey:str='X_umap', fullLabelKey:str='ingest'
    ):
    '''The function `sketchBasedLabelTransfer` performs label transfer from a sketch dataset to a full dataset using a specified embedding method and metric.
    similar with `ProjectData` in Seurat

    Parameters
    ----------
    ad_sketch : sc.AnnData
        Anndata object containing the sketch dataset.
    ad_full : sc.AnnData
        Anndata object containing the full dataset with embeddings and labels.
    sketchEmbedding : str
        The name of the embedding in the `ad_sketch` AnnData object that will be used for sketch-based label transfer.
    fullEmbedding : str
        The `fullEmbedding` parameter is a string that specifies the key in `ad_full.obsm` where the full dataset embedding is stored. This embedding should be a matrix of shape (n_samples, n_features).
    sketchLabels : Union[str, List[str]]
        The `sketchLabels` parameter is the label(s) that you want to transfer from the sketch dataset to the full dataset. It can be either a single label or a list of labels.
    metric : str, optional
        The metric parameter specifies the distance metric to be used for calculating the nearest neighbors. The default value is 'cosine', which calculates the cosine similarity between vectors. Other options include 'euclidean', 'manhattan', and 'correlation'.
    nNeighbors : int, optional
        The `nNeighbors` parameter specifies the number of nearest neighbors to consider when constructing the neighborhood graph for the sketch data.
    minDist : float
        The `minDist` parameter in the `sketchBasedLabelTransfer` function is a float value that controls the minimum distance between points in the UMAP embedding. It determines how tightly the points are clustered together in the UMAP plot. A smaller value of `minDist` will result in more
    sketchUmapKey : str, optional
        The parameter `sketchUmapKey` is a string that specifies the key in the `ad_sketch.obsm` dictionary where the UMAP embedding of the sketch data will be stored. This embedding is computed using the `sc.tl.umap` function and is used for label transfer.
    fullUmapKey : str, optional
        The parameter `fullUmapKey` is a string that specifies the key in the `ad_full.obsm` dictionary where the UMAP embedding of the full dataset will be stored.
    fullLabelKey : str, optional
        The `fullLabelKey` parameter is a string that specifies the key in the `ad_full` AnnData object where the transferred labels will be stored. This key will be used to access the transferred labels later on.

    '''
    if isinstance(sketchLabels, str):
        sketchLabels = [sketchLabels]
    for x in sketchLabels:
        assert x in ad_sketch.obs.columns, f"{x} not in ad_sketch.obs.columns"

    ad_sketchRpca = sc.AnnData(
        X=ad_sketch.obsm[sketchEmbedding].values, obs=ad_sketch.obs, var=ad_sketch.obsm[sketchEmbedding] >> F(lambda _: range(_.shape[1])) >> F(map, lambda _: f"embedding_{_}") >> F(list) >> F(lambda _: pd.DataFrame(index=_))
        )
    ad_sketchRpca.obsm['X_pca'] = ad_sketchRpca.X

    ad_fullRpca = sc.AnnData(
        X=ad_full.obsm[fullEmbedding].values, obs=ad_full.obs, var=ad_full.obsm[fullEmbedding] >> F(lambda _: range(_.shape[1])) >> F(map, lambda _: f"embedding_{_}") >> F(list) >> F(lambda _: pd.DataFrame(index=_))
        )
    ad_fullRpca.obsm['X_pca'] = ad_fullRpca.X

    sc.pp.neighbors(ad_sketchRpca, n_pcs=ad_sketchRpca.obsm['X_pca'].shape[1], metric=metric, n_neighbors=nNeighbors, use_rep='X_pca')
    sc.tl.umap(ad_sketchRpca, min_dist=minDist)
    sc.tl.ingest(ad_fullRpca, ad_sketchRpca, obs=sketchLabels, embedding_method='umap')

    ad_sketch.obsm[sketchUmapKey] = ad_sketchRpca.obsm['X_umap'].copy()
    ad_full.obsm[fullUmapKey] = ad_fullRpca.obsm['X_umap'].copy()

    ad_full.obsm[fullLabelKey] = ad_fullRpca.obs[sketchLabels].copy()
