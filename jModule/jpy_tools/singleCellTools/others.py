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
    ad:sc.AnnData,
    groupby:str,
    layer:str,
    dir_result:str,
    jobid:str,
    k:int=75,
    minModuleSize:int=50,
    min_cells:int=50,
    n_top_genes:int=10000,
    batch_key: Optional[str]=None,
    threads: int=16,
    soft_power:Optional[int]=None, 
    max_block_size:Optional[int]=None
) -> sc.AnnData:
    """
    perform scWGCNA

    Parameters
    ----------
    ad : sc.AnnData
    groupby : str
        used for generate meta cells
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
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    rBase = importr("base")
    rUtils = importr("utils")
    tidyverse = importr("tidyverse")
    WGCNA = importr("WGCNA")
    seurat = importr("Seurat")

    R = ro.r
    R(f"enableWGCNAThreads({threads})")

    ro.globalenv["dir_result"] = dir_result
    if not batch_key:
        batch_key = groupby
    if not max_block_size:
        max_block_size = n_top_genes

    # preprocess
    sc.pp.filter_genes(ad, min_cells=min_cells)
    sc.pp.highly_variable_genes(
        ad,
        layer=layer,
        flavor="seurat_v3",
        batch_key=batch_key,
        n_top_genes=n_top_genes,
    )
    ad_forWgcna = ad[:, ad.var["highly_variable"]]

    so = ad2so(ad_forWgcna, layer=layer)
    ro.globalenv["so"] = so

    # construct meta cells
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
    sc.pl.umap(ad_meta, color=[groupby, batch_key])

    # perform wgcna
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
        R(
            f"""
        # Choose a set of soft-thresholding powers
        powers = c(seq(1,10,by=1), seq(12,30, by=2));

        # Call the network topology analysis function for each set in turn
        powerTable = list(
        data = pickSoftThreshold(
            datExpr,
            powerVector=powers,
            verbose = 100,
            networkType="signed",
            corFnc="bicor"
        )[[2]]
        );
        """
        )

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
        soft_power = int(input('Soft Power'))
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

    cyt = R("cyt")
    df_edge = r2py(rGet(cyt, "$edgeData"))
    df_node = r2py(rGet(cyt, "$nodeData"))

    dt_cyt = {"node": df_node, "edge": df_edge}
    ad_meta.uns["cyt"] = dt_cyt
    return ad_meta

