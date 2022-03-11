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
from tempfile import NamedTemporaryFile
import collections
from xarray import corr
import sys
from . import basic


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

def pwStack(ls_ax, ncols=5):
    import patchworklib as pw
    from more_itertools import chunked
    from cool import F
    ls_ax = chunked(ls_ax, ncols) | F(list)
    if len(ls_ax) == 1:
        axs = pw.stack(ls_ax[0])
    else:
        axs = pw.stack([pw.stack(x) for x in ls_ax[:-1]], operator="/")
        ls_name = list(axs.bricks_dict.keys())
        for i, ax in enumerate(ls_ax[-1]):
            axs = axs[ls_name[i]] / ax
    return axs