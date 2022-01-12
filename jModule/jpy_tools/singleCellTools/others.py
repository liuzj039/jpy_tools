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
    rawLayer = 'raw'
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

    renv['minModuleSize'] = minModuleSize
    renv['deepSplit'] = deepSplit
    renv['mergeCutHeight'] = mergeCutHeight
    renv['maxBlockSize'] = len(ls_hvgGene)
    renv['jobid'] = jobid
    renv['dir_result'] = dir_result

    ad_meta = ad[:, ls_hvgGene].copy()
    ad_meta.var.index = ad_meta.var.index.map(lambda x:x.replace('_', '-'))
    so = ad2so(
        ad_meta,
        layer=rawLayer,
        ls_obs=[],
        ls_var=[],
        lightMode=True,
        dataLayer=layer,
    )
    renv['so'] = so

    R("""
    datExpr <- as.data.frame(GetAssayData(so, assay='RNA', slot='data'))
    datExpr <- as.data.frame(t(datExpr))
    datExpr <- datExpr[,goodGenes(datExpr)]

    lsR_useGene = colnames(datExpr)
    """)

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
    renv['softPower'] = softPower

    if threads > 1:
        R(f"enableWGCNAThreads({threads})")
    else:
        R(f"disableWGCNAThreads()")

    R("""
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
    """)

    with r_inline_plot(width=768):
        R("""
        plotDendroAndColors(consTree, moduleColors, "Module colors", dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, guideHang = 0.05,
                            main = paste0("ODC lineage gene dendrogram and module colors"))""")
    with r_inline_plot(width=768):
        R("""
        plotEigengeneNetworks(PCvalues, "Eigengene adjacency heatmap", 
                            marDendro = c(3,3,2,4),
                            marHeatmap = c(3,4,2,2), plotDendrograms = T, 
                            xLabelsAngle = 90)
        """)
    R("""
    load(paste0(dir_result, "/", jobid, "_TOM_block.1.rda"), verbose=T)

    probes = colnames(datExpr)
    TOM <- as.matrix(consTomDS)
    dimnames(TOM) <- list(probes, probes)

    # cyt = exportNetworkToCytoscape(TOM,
    #             weighted = TRUE, threshold = 0.1,
    #             nodeNames = probes, nodeAttr = moduleColors)
    """)

    ad_meta = ad_meta[:, list(renv['lsR_useGene'])]
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
    ad.layers['normalize'] = ad.layers[layer].copy()
    sc.pp.normalize_total(ad, 1e4, layer='normalize')

    ad_psedoBulk = sc.AnnData(ad.to_df('normalize').groupby(ad.obs[cluster]).agg("mean"))
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


def addMetaCellLayer(ad:sc.AnnData, layer:str, obsm:str, n_neighbors = 50, obsp = None, boolConnectivity = False):
    """
    Add meta-cell layer.
    """
    if not obsp:
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep=obsm, key_added = 'meta')
        obsp = 'meta_connectivities'
    if boolConnectivity:
        ar_neighbors = np.eye(ad.shape[0]) + (ad.obsp[obsp] > 0)
        ad.layers[f"{layer}_meta"] = (ar_neighbors @ ad.layers[layer]) / (n_neighbors + 1)
    else:
        ar_connect =  np.eye(ad.shape[0]) + ad.obsp[obsp].A
        ar_neighbors = ar_connect * (1 / ar_connect.sum(0))
        ad.layers[f"{layer}_meta"] = ar_neighbors @ ad.layers[layer]
