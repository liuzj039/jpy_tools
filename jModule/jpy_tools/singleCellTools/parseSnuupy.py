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