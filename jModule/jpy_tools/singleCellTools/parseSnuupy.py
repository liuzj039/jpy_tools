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
from joblib import Parallel, delayed
from statsmodels.stats.proportion import proportions_ztest, test_proportions_2indep
from statsmodels.stats import multitest
import collections
from xarray import corr
import scipy.sparse as ss
from . import basic
import pysam
from collections import defaultdict
from cool import F
import muon
from functools import reduce
import muon as mu

def addSnuupyBamTag(path_bam, path_out, tag="CB"):
    bam_org = pysam.AlignmentFile(path_bam)
    bam_out = pysam.AlignmentFile(path_out, mode="wb", template=bam_org)

    for read in tqdm(
        bam_org, "add barcode tag", total=bam_org.unmapped + bam_org.mapped
    ):
        readBc = read.qname.split("_")[0]
        read.set_tag(tag, readBc)
        bam_out.write(read)

    bam_org.close()
    bam_out.close()
    pysam.index(path_out)


def getDiffSplicedPvalue(
    ad_groupSpliceRatio, min_total=5, threads=12, method="ztest", unsplicedLayer = 'unspliced', totalLayer = 'total'
) -> pd.DataFrame:
    """
    get pvalue for spliced vs unspliced

    ad_groupSpliceRatio: result of `calcGroupUnsplicedRatio`
    """
    lsDf_spliceInfo = []
    for i in range(len(ad_groupSpliceRatio)):
        currentObsIndex = ad_groupSpliceRatio[i].obs.index[0]
        ad_current = ad_groupSpliceRatio[i]
        ad_others = ad_groupSpliceRatio[
            ad_groupSpliceRatio.obs.index != currentObsIndex
        ]
        df_currentUnspliced = (
            ad_current.to_df(unsplicedLayer).sum().rename("currentUnspliced")
        )
        df_currentTotal = ad_current.to_df(totalLayer).sum().rename("currentTotal")

        df_othersUnspliced = (
            ad_others.to_df(unsplicedLayer).sum().rename("othersUnspliced")
        )
        df_othersTotal = ad_others.to_df(totalLayer).sum().rename("othersTotal")

        df_currentSpliceInfo = pd.concat(
            [df_currentUnspliced, df_currentTotal, df_othersUnspliced, df_othersTotal],
            axis=1,
        ).assign(cluster=currentObsIndex)
        lsDf_spliceInfo.append(df_currentSpliceInfo)
    df_spliceInfo = pd.concat(lsDf_spliceInfo)

    df_spliceInfo = df_spliceInfo.query(
        "currentTotal > @min_total & othersTotal > @min_total"
    )

    fc_ztest = lambda x: proportions_ztest(
        [x.currentUnspliced, x.othersUnspliced], [x.currentTotal, x.othersTotal]
    )[1]
    fc_waldtest = lambda x: test_proportions_2indep(
        x.currentUnspliced,
        x.currentTotal,
        x.othersUnspliced,
        x.othersTotal,
        method="wald",
        compare="diff",
    )[1]
    fc_test = {"ztest": fc_ztest, "wald": fc_waldtest}[method]

    ls_pvalue = Parallel(threads, batch_size=100)(
        delayed(fc_test)(x)
        for x in tqdm(df_spliceInfo.itertuples(), total=len(df_spliceInfo))
    )
 
    df_spliceInfo["pvalue"] = ls_pvalue
    df_spliceInfo = df_spliceInfo.fillna(1)
    df_spliceInfo["qvalue"] = df_spliceInfo.groupby('cluster')['pvalue'].transform(lambda sr:multitest.fdrcorrection(sr)[1])
    df_spliceInfo = df_spliceInfo.eval(
        "currentUnsplicedRatio = currentUnspliced / currentTotal"
    ).eval("othersUnsplicedRatio = othersUnspliced / othersTotal")
    return df_spliceInfo


def calcUnsplicedRatioFromSnuupyMtx(
    ad, layer, useAmbiguousCalcUnsplicedRatio=False
) -> sc.AnnData:
    """
    calculate unspliced ratio from snuupy mtx

    ad:
        mode splice
    layer:
        raw counts
    """
    ad.var = ad.var.assign(
        gene=lambda df: df.index.str.split("_").str[:-2].str.join("_"),
        spliceGroup=lambda df: df.index.str.split("_").str[-2],
    )
    ls_allGene = ad.var["gene"].unique().tolist()
    ad_splice = sc.AnnData(
        ss.csr_matrix((ad.shape[0], len(ls_allGene))),
        obs=ad.obs.copy(),
        var=pd.DataFrame(index=ls_allGene),
    )
    dt_name2layer = {"True": "spliced", "False": "unspliced"}

    for group, ls_var in (
        ad.var.groupby("spliceGroup").apply(lambda df: df.index.to_list()).items()
    ):
        layerName = dt_name2layer.get(group, group)
        df_groupMtx = ad[:, ls_var].to_df(layer)
        df_groupMtx = (
            df_groupMtx.rename(columns=lambda x: "_".join(x.split("_")[:-2]))
            .reindex(columns=ls_allGene)
            .fillna(0)
        )
        ad_splice.layers[layerName] = df_groupMtx
    if useAmbiguousCalcUnsplicedRatio:
        ad_splice.X = ad_splice.layers["unspliced"] / (
            ad_splice.layers["spliced"]
            + ad_splice.layers["unspliced"]
            + ad_splice.layers["Ambiguous"]
        )
    else:
        ad_splice.X = ad_splice.layers["unspliced"] / (
            ad_splice.layers["spliced"] + ad_splice.layers["unspliced"]
        )
    return ad_splice


def calcGroupUnsplicedRatio(
    ad: anndata.AnnData,
    layer: str = "raw",
    cluster="leiden",
    useAmbiguousCalcUnsplicedRatio=False,
    minCounts=1,
) -> sc.AnnData:
    """
    calculate group unspliced ratio from snuupy mtx

    ad:
        mode splice
    layer:
        raw counts
    """
    ad_splice = calcUnsplicedRatioFromSnuupyMtx(
        ad, layer, useAmbiguousCalcUnsplicedRatio
    )
    ls_cluster = ad_splice.obs[cluster].unique().tolist()
    ad_groupSplice = sc.AnnData(
        ss.csr_matrix((len(ls_cluster), ad_splice.shape[1])),
        obs=pd.DataFrame(index=ls_cluster),
        var=ad_splice.var.copy(),
    )
    for group in ad_splice.layers:
        df_groupMtx = ad_splice.to_df(group)
        df_groupMtxCluster = (
            df_groupMtx.groupby(ad_splice.obs[cluster]).sum().reindex(ls_cluster)
        )
        ad_groupSplice.layers[group] = df_groupMtxCluster
    if useAmbiguousCalcUnsplicedRatio:
        ad_groupSplice.layers["total"] = (
            ad_groupSplice.layers["spliced"]
            + ad_groupSplice.layers["unspliced"]
            + ad_groupSplice.layers["Ambiguous"]
        )
    else:
        ad_groupSplice.layers["total"] = (
            ad_groupSplice.layers["spliced"] + ad_groupSplice.layers["unspliced"]
        )
    ad_groupSplice.layers["unsplicedRatio"] = (
        ad_groupSplice.layers["unspliced"] / ad_groupSplice.layers["total"]
    )
    ad_groupSplice.X = ad_groupSplice.layers["unsplicedRatio"].copy()
    ad_groupSplice = ad_groupSplice[
        :, ad_groupSplice.layers["total"].sum(0) >= minCounts
    ]
    return ad_groupSplice


def createMdFromSnuupy(
    ad: anndata.AnnData, removeAmbiguous: bool = True
) -> "mu.MuData":
    import muon as mu
    import scipy.sparse as ss

    ad = updateOldMultiAd(ad)
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
    for i, line in tqdm(enumerate(irInfoDf.itertuples()), total=allLinesCounts):
        barcode = line.Name.split("_")[0]
        lineCountMtxDt = intronCountMtxDt.get(barcode, {})
        lineRetenMtxDt = intronRetenMtxDt.get(barcode, {})
        if pd.isna(line.ExonOverlapInfo):
            intronCov = []
        else:
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

    ls_var = list(intronCountMtxDf.columns | intronRetenMtxDf.columns)
    ls_obs = list(intronCountMtxDf.index | intronRetenMtxDf.index)
    intronRetenMtxDf = intronRetenMtxDf.reindex(index = ls_obs).reindex(columns = ls_var).fillna(0)
    intronCountMtxDf = intronCountMtxDf.reindex(index = ls_obs).reindex(columns = ls_var).fillna(0)

    ad_final = sc.AnnData(X = ss.csr_matrix(intronCountMtxDf.values), obs = pd.DataFrame(index = ls_obs), var = pd.DataFrame(index = ls_var))
    ad_final.layers['unspliced'] = ss.csr_matrix(intronRetenMtxDf.values)
    ad_final.layers['total'] = ad_final.X.copy()
    ad_final.layers['spliced'] = ad_final.X - ad_final.layers['unspliced']

    # intronCountMtxAd = basic.creatAnndataFromDf(intronCountMtxDf)
    # intronCountMtxAd.X = ss.csr_matrix(intronCountMtxAd.X)
    # intronRetenMtxAd = basic.creatAnndataFromDf(intronRetenMtxDf)
    # intronRetenMtxAd.X = ss.csr_matrix(intronRetenMtxAd.X)

    # useIntronLs = list(intronRetenMtxAd.var.index | intronCountMtxAd.var.index)
    # useCellLs = list(intronRetenMtxAd.obs.index | intronCountMtxAd.obs.index)

    # intronRetenMtxAd = intronRetenMtxAd[useCellLs, useIntronLs].copy()
    # intronCountMtxAd = intronCountMtxAd[useCellLs, useIntronLs].copy()

    # intronCountMtxAd.layers['total'] = intronCountMtxAd.X.copy()
    # intronCountMtxAd.layers['unspliced'] = intronRetenMtxAd.X
    # intronCountMtxAd.layers['spliced'] = intronCountMtxAd.X - intronRetenMtxAd.X

    return ad_final

def addIntronSplicedAdToMd(md: mu.MuData, dt_spliceInfoPath: Mapping[str, str], indexUnique:str, modName = 'splice_intron', threads=4):
    md.update()
    lsAd = Parallel(threads)(
        delayed(getSpliceInfoOnIntronLevel)(x) for x in dt_spliceInfoPath.values()
    )
    ad_spliceIntronLevel = sc.concat(
        {x: y for x, y in zip(dt_spliceInfoPath.keys(), lsAd)}, index_unique=indexUnique, join='outer'
    )
    ad_spliceIntronLevel = ad_spliceIntronLevel[ad_spliceIntronLevel.obs.eval('index in @md.obs.index')].copy()
    md.mod[modName] = ad_spliceIntronLevel
    md.update()


def getSpliceInfoFromSnuupyAd(nanoporeAd):
    """
    用于从snuupy crMode产生的NanoporeMtx中提取产生splice和unsplice的read

    return:
        adata:
            X: unsplice + splice
            layer[unspliced, spliced]
    """
    nanoporeCountAd = nanoporeAd[:, ~nanoporeAd.var.index.str.contains("_")]
    unsplicedAd = nanoporeAd[:, nanoporeAd.var.index.str.contains("False_fullySpliced")]
    unsplicedAd.var.index = unsplicedAd.var.index.str.split("_").str[0]
    splicedAd = nanoporeAd[:, nanoporeAd.var.index.str.contains("True_fullySpliced")]
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
            for x in [x.strip().split("=")[-1].strip() for x in resultStr.split("\n")]
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
        return fisher_exact([[xUnsplice, xSplice], [yUnsplice, ySplice]], fisherMethod)[
            1
        ]

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

        clusterSpliceIntronInfoDf["pvalue"] = clusterSpliceIntronInfoDf.parallel_apply(
            calcuPvalue, axis=1
        )
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

def concatMd(dtMd, label, indexUnique, mod="inner"):
    dt_sampleMod = {x: set(list(y.mod.keys())) for x, y in dtMd.items()}
    if mod == "inner":
        ls_useMod = list(reduce(lambda x, y: list(set(x) & set(y)), list(dt_sampleMod.values())))
    elif mod == 'outer':
        ls_useMod = list(reduce(lambda x, y: list(set(x) | set(y)), list(dt_sampleMod.values())))
    else:
        assert False, "Unknown mod parameter"
    dtAd_mod = {}
    for mod in ls_useMod:
        dtAd_singleMod = {x:y[mod] for x,y in dtMd.items() if mod in y.mod}
        ad_mod = sc.concat(dtAd_singleMod, join='outer', label=label, index_unique=indexUnique)
        dtAd_mod[mod] = ad_mod
    md = mu.MuData(dtAd_mod)
    return md

class SnuupySpliceInfo(object):
    """Snuupy splice info object"""
    def __init__(self, df=None, path=None, suffix=None, ad=None, isor=False):
        if df is None:
            assert not path is None, "no path found"
            df = pd.read_table(path)

        if not isor:
            df = df.assign(
                barcode=lambda df: df["Name"].str.split("_").str[0],
                readExonCounts=lambda df: df["ExonOverlapInfo"].str.count(",") + 1,
            )
            if suffix:
                df["barcode"] = df["barcode"] + suffix
        self.ad = ad
        self.df = df
        self.oneExonGene = (
            df.query("GeneExonCounts == 1")["geneId"].to_list() | F(set) | F(list)
        )

    def getSpliceMtx(self, fullLength=False, dt_usedIntron=None):
        "dt_usedIntron: {geneId: [intronIds]}: {'AT1': [0,1]}"
        df = self.df.copy().fillna("")
        df["IntronOverlapInfo"] = (
            df["IntronOverlapInfo"].str.split(",").map(lambda z: [x for x in z if x])
        )
        dt_expSpliced = defaultdict(lambda: defaultdict(lambda: 0))
        dt_expUnspliced = defaultdict(lambda: defaultdict(lambda: 0))
        dt_expAmb = defaultdict(lambda: defaultdict(lambda: 0))
        if fullLength:
            df = df.query("readExonCounts == GeneExonCounts")

        for line in tqdm(df.itertuples(), total=len(df), desc="Get Splice Mtx"):
            if (line.readExonCounts == 1) and (len(line.IntronOverlapInfo) == 0):
                dt_expAmb[line.barcode][line.geneId] += 1
                continue
            if dt_usedIntron is None:
                if len(line.IntronOverlapInfo) == 0:
                    dt_expSpliced[line.barcode][line.geneId] += 1
                else:
                    dt_expUnspliced[line.barcode][line.geneId] += 1
            else:
                if not line.geneId in dt_usedIntron:
                    dt_expAmb[line.barcode][line.geneId] += 1
                elif len(line.IntronOverlapInfo) == 0:
                    dt_expSpliced[line.barcode][line.geneId] += 1
                elif set(dt_usedIntron[line.geneId]) & set(line.IntronOverlapInfo):
                    dt_expUnspliced[line.barcode][line.geneId] += 1
                else:
                    dt_expSpliced[line.barcode][line.geneId] += 1
        return dt_expSpliced, dt_expUnspliced, dt_expAmb

    def addSpliceInfoToAd(
        self,
        fullLength=False,
        dt_usedIntron=None,
        useAmbiguousCalcUnsplicedRatio=False,
        layerPrefix="splice",
    ):
        assert self.ad, "Not Found Default Adata"
        ad = self.ad
        dt_expSpliced, dt_expUnspliced, dt_expAmb = self.getSpliceMtx(
            fullLength, dt_usedIntron
        )
        logger.info("Add splice matrix to adata")
        ad.layers[f"{layerPrefix}_spliced"] = (
            pd.DataFrame(dt_expSpliced)
            .T.reindex(ad.obs.index)
            .reindex(columns=ad.var.index)
            .fillna(0)
        )
        ad.layers[f"{layerPrefix}_spliced"] = ss.csr_matrix(ad.layers[f"{layerPrefix}_spliced"])
        logger.info("Add unsplice matrix to adata")
        ad.layers[f"{layerPrefix}_unspliced"] = (
            pd.DataFrame(dt_expUnspliced)
            .T.reindex(ad.obs.index)
            .reindex(columns=ad.var.index)
            .fillna(0)
        )
        ad.layers[f"{layerPrefix}_unspliced"] = ss.csr_matrix(ad.layers[f"{layerPrefix}_unspliced"])
        logger.info("Add ambiguous matrix to adata")
        ad.layers[f"{layerPrefix}_amb"] = (
            pd.DataFrame(dt_expAmb)
            .T.reindex(ad.obs.index)
            .reindex(columns=ad.var.index)
            .fillna(0)
        )
        ad.layers[f"{layerPrefix}_amb"] = ss.csr_matrix(ad.layers[f"{layerPrefix}_amb"])
        logger.info("Add splice ratio to adata")
        if useAmbiguousCalcUnsplicedRatio:
            ad.obs[f"{layerPrefix}_incompletelySplicedRatio"] = ad.to_df(
                f"{layerPrefix}_unspliced"
            ).sum(1) / (
                ad.to_df(f"{layerPrefix}_spliced").sum(1)
                + ad.to_df(f"{layerPrefix}_unspliced").sum(1)
                + ad.to_df(f"{layerPrefix}_amb").sum(1)
            )
        else:
            ad.obs[f"{layerPrefix}_incompletelySplicedRatio"] = ad.to_df(
                f"{layerPrefix}_unspliced"
            ).sum(1) / (
                ad.to_df(f"{layerPrefix}_spliced").sum(1)
                + ad.to_df(f"{layerPrefix}_unspliced").sum(1)
            )

    def calcGroupUnsplicedRatio(
        self,
        cluster,
        layerPrefix="splice",
        useAmbiguousCalcUnsplicedRatio=False,
        minCounts=1,
    ) -> sc.AnnData:
        ad = self.ad

        ls_cluster = ad.obs[cluster].unique().tolist()

        ad_groupSplice = sc.AnnData(
            ss.csr_matrix((len(ls_cluster), ad.shape[1])),
            obs=pd.DataFrame(index=ls_cluster),
            var=ad.var.copy(),
        )
        for group in [
            f"{layerPrefix}_unspliced",
            f"{layerPrefix}_spliced",
            f"{layerPrefix}_amb",
        ]:
            df_groupMtx = ad.to_df(group)
            df_groupMtxCluster = (
                df_groupMtx.groupby(ad.obs[cluster]).sum().reindex(ls_cluster)
            )
            ad_groupSplice.layers[group] = df_groupMtxCluster
        if useAmbiguousCalcUnsplicedRatio:
            ad_groupSplice.layers[f"{layerPrefix}_total"] = (
                ad_groupSplice.layers[f"{layerPrefix}_spliced"]
                + ad_groupSplice.layers[f"{layerPrefix}_unspliced"]
                + ad_groupSplice.layers[f"{layerPrefix}_amb"]
            )
        else:
            ad_groupSplice.layers[f"{layerPrefix}_total"] = (
                ad_groupSplice.layers[f"{layerPrefix}_spliced"]
                + ad_groupSplice.layers[f"{layerPrefix}_unspliced"]
            )
        ad_groupSplice.layers[f"{layerPrefix}_incompletelySplicedRatio"] = (
            ad_groupSplice.layers[f"{layerPrefix}_unspliced"]
            / ad_groupSplice.layers[f"{layerPrefix}_total"]
        )
        ad_groupSplice.X = ad_groupSplice.layers[
            f"{layerPrefix}_incompletelySplicedRatio"
        ].copy()
        ad_groupSplice = ad_groupSplice[
            :, ad_groupSplice.layers[f"{layerPrefix}_total"].sum(0) >= minCounts
        ]
        return ad_groupSplice

    def generateMd(self, layerPrefix="splice"):
        ls_spliceVar = [*(self.ad.var.index + '_spliced'), *(self.ad.var.index + '_unspliced')]
        ls_spliceObs = self.obs.index.copy()
        ar_splice = np.c_[self.ad.layers[f'{layerPrefix}_spliced'], self.ad.layers[f'{layerPrefix}_unspliced']]
        ad_splice = sc.AnnData(ar_splice, obs=pd.DataFrame(ls_spliceObs), var=pd.DataFrame(ls_spliceVar))
        md = muon.MuData({"illu":self.ad, "splice":ad_splice})
        return md

    def __or__(self, obj):
        df = pd.concat([self.df, obj.df])
        return SnuupySpliceInfo(df=df, isor=True)