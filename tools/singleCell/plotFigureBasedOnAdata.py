"""
@Date: 2020-07-28 19:38:25
LastEditors: liuzj
LastEditTime: 2020-08-03 11:12:07
@Description: file content
@Author: liuzj
FilePath: /liuzj/projects/singleCellEndo/01_pipeline/pipeline0630/needAdata/plotFigureBasedOnAdata.py
"""
import matplotlib
matplotlib.use('AGG')

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
import pickle
import click
import yaml
import sh


def addCellIrInfo(adata, irInfo):
    cellInformation = irInfo.groupby("barcode").apply(lambda x: pd.Series(
        [len(x.dropna(subset=["IntronOverlapInfo"])) / len(x)]))
    cellInformation.columns = ["irRatio"]
    cellInformation.index = cellInformation.index + "-1"
    adata.obs = adata.obs.join(cellInformation)
    return adata


def plotCellClusterIr(adata, savePath, passedCluster=False):
    if not passedCluster:
        _temp = adata.obs
        ax = sns.boxplot(
            _temp["louvain"].values.astype(int),
            _temp["irRatio"].values,
            palette=["#EE7785"],
        )
    else:
        _temp = adata.obs.query("louvain in @passedCluster")
        ax = sns.violinplot(
            _temp["louvain"].values.astype(int),
            _temp["irRatio"].values,
            palette=["#EE7785"],
        )

    ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()], size=12)
    ax.set_ylabel("Intron Retention Ratio(Cell)", size=14, weight="bold")

    ax.set_xticklabels([x.get_text() for x in ax.get_xticklabels()], size=12)
    ax.set_xlabel("Cluster", size=14, weight="bold")

    plt.savefig(savePath, format="svg")


def parseOneRead(line):
    lineIntrons = line.ExonOverlapInfo.split(",")
    lineIntrons = np.array(lineIntrons).astype(int)

    lineGeneIntronCounts = line.GeneExonCounts - 1
    intronIrDenominator = np.zeros(lineGeneIntronCounts)

    intronIrDenominator[min(lineIntrons):max(lineIntrons)] = 1

    intronIrNumerator = np.zeros(lineGeneIntronCounts)

    if pd.isna(line.IntronOverlapInfo):
        return np.array([intronIrNumerator, intronIrDenominator])
    else:
        lineIrIntrons = line.IntronOverlapInfo.split(",")
        for singleLineIrIntron in lineIrIntrons:
            singleLineIrIntron = int(singleLineIrIntron)
            intronIrNumerator[singleLineIrIntron] += 1
        intronIrDenominator = ((intronIrDenominator +
                                intronIrNumerator).astype(bool).astype(int))
        return np.array([intronIrNumerator, intronIrDenominator])


def parseOneGene(dtframe, groupbyName):
    intronIrFraction = [parseOneRead(x) for x in dtframe.itertuples()]
    intronIrFraction = reduce(lambda a, b: a + b, intronIrFraction)
    intronIrFraction = np.concatenate([
        intronIrFraction,
        (intronIrFraction[0] / intronIrFraction[1]).reshape(1, -1)
    ])
    intronIrFraction = "\t".join(
        [",".join(x) for x in intronIrFraction.astype(str)])

    geneReadCounts = len(dtframe)
    geneIr = len(dtframe.dropna()) / geneReadCounts
    geneName = dtframe.iloc[0].loc["geneId"]
    geneIntronCounts = dtframe.iloc[0, 2] - 1
    geneCluster = dtframe.iloc[0].loc[groupbyName]
    return f"{geneName}\t{geneIntronCounts}\t{geneReadCounts}\t{intronIrFraction}\t{geneIr}\t{geneCluster}\n"


def getOneClusterIrResults(dtframe, groupbyName):
    oneClusterContents = ""
    dtframeGroupby = iter(dtframe.groupby("geneId"))
    for dtframeChunk in dtframeGroupby:
        oneClusterContents += parseOneGene(dtframeChunk[1], groupbyName)
    return oneClusterContents


def getAllClusterIrResults(dtframe, groupbyName="louvain"):
    allClusterContents = ""
    dtframeGroupby = iter(dtframe.groupby(groupbyName))
    for dtframeSingleCluster in dtframeGroupby:
        allClusterContents += getOneClusterIrResults(dtframeSingleCluster[1],
                                                     groupbyName)
    return allClusterContents


def getCommonNameLine(dtframe):
    #     print(dtframe.columns)
    nameSet = dtframe.groupby("cluster")["Name"].agg(lambda x: set(x)).tolist()
    commonSet = nameSet[0].intersection(*nameSet[1:])
    dtframe = dtframe.query("Name in @commonSet")
    return dtframe


def plotGeneClusterIr(geneIrInfoFiltered,
                      savePath,
                      passedCluster=False,
                      yLabel="Gene"):
    if not passedCluster:
        _temp = geneIrInfoFiltered
        ax = sns.boxplot(
            _temp["cluster"].values.astype(int),
            _temp["readIrRatio"].values,
            palette=["#EE7785"],
        )
    else:
        _temp = geneIrInfoFiltered.query("cluster in @passedCluster")
        ax = sns.violinplot(
            _temp["cluster"].values.astype(int),
            _temp["readIrRatio"].values,
            palette=["#EE7785"],
        )

    ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()], size=12)
    ax.set_ylabel(f"Intron Retention Ratio({yLabel})", size=14, weight="bold")

    ax.set_xticklabels([x.get_text() for x in ax.get_xticklabels()], size=12)
    ax.set_xlabel("Cluster", size=14, weight="bold")

    plt.savefig(savePath, format="svg")


def getGeneIntronIrInfo(irInfo, groupbyColumn="louvain", cutoff=5):
    geneIrInfoString = getAllClusterIrResults(irInfo, groupbyColumn)
    header = [
        "Name",
        "intronCounts",
        "readCoverage",
        "intronRetentionReadCounts",
        "intronCoverage",
        "intronRetentionRatio",
        "readIrRatio",
        "cluster",
    ]
    geneIrInfo = pd.read_table(StringIO(geneIrInfoString),
                               header=None,
                               names=header).query("readCoverage > @cutoff")
    intronIrInfo = pd.read_table(StringIO(geneIrInfoString),
                                 header=None,
                                 names=header)
    geneIrInfoFiltered = getCommonNameLine(geneIrInfo)
    return geneIrInfoFiltered, intronIrInfo


def prepareForPlotScatterPlot(irInfo, useLouvain, representProtein, cutoff):
    allClusterCompareInfo = {}
    with ProcessPoolExecutor(64) as multiP:
        for singleCluster in useLouvain:
            singleClusterCompareInfo = {}
            _useLouvain = useLouvain[:]
            _useLouvain.remove(singleCluster)
            _irInfo = irInfo.copy(True)

            _irInfo["otherClusters"] = "Other Clusters"
            _irInfo.loc[_irInfo["louvain"] == singleCluster,
                        "otherClusters"] = singleCluster

            singleClusterCompareInfo["otherClusters"] = multiP.submit(
                getGeneIntronIrInfo, _irInfo, "otherClusters")
            print(f"start {singleCluster} others")

            for singleOtherCluster in _useLouvain:
                irInfo1v1 = _irInfo.query(
                    f"louvain in ['{singleCluster}','{singleOtherCluster}']")
                singleClusterCompareInfo[singleOtherCluster] = multiP.submit(
                    getGeneIntronIrInfo, irInfo1v1)
                print(f"start {singleCluster} {singleOtherCluster}")

            allClusterCompareInfo[singleCluster] = singleClusterCompareInfo

    allClusterGeneCompareInfo = {
        i: {k: l.result()[0]
            for k, l in j.items()}
        for i, j in allClusterCompareInfo.items()
    }
    allClusterIntronCompareInfo = {
        i: {
            k: transformFormatToGeneral(l.result()[1], representProtein, False,
                                        cutoff)
            for k, l in j.items()
        }
        for i, j in allClusterCompareInfo.items()
    }
    return allClusterGeneCompareInfo, allClusterIntronCompareInfo


def plotSubClusterCompareOthers(clusterIrData, columnName, ax):

    clusterIrData = clusterIrData.pivot_table(index="Name",
                                              values="readIrRatio",
                                              columns="cluster")
    clusterIrData.columns = [str(x) for x in clusterIrData.columns]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sns.scatterplot(
        columnName[0],
        columnName[1],
        data=clusterIrData,
        color="#000000",
        alpha=0.4,
        ax=ax,
    )

    columnName = [
        f"Cluster {x}" if not x.startswith("Other") else x for x in columnName
    ]

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()], size=12)
    ax.set_xlabel(columnName[0], size=13, weight="bold")

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()], size=12)
    ax.set_ylabel(columnName[1], size=13, weight="bold")


def plotAllClusterCompareOthers(geneIrCompareWithinClusterInfo, savePath):
    i = 0
    subFigCounts = len(geneIrCompareWithinClusterInfo)
    fig, ax = plt.subplots(subFigCounts,
                           subFigCounts,
                           figsize=(subFigCounts * 5, subFigCounts * 5))

    for singleClusterName, otherClusterInfo in geneIrCompareWithinClusterInfo.items(
    ):
        j = 0
        if j == i:
            j += 1

        for singleOtherClusterName, clusterInfo in otherClusterInfo.items():
            if singleOtherClusterName == "otherClusters":
                singleOtherClusterName = "Other Clusters"
                k = i
                j -= 1
            else:
                k = j
            print(i, j, k)
            plotSubClusterCompareOthers(
                clusterInfo, [singleClusterName, singleOtherClusterName],
                ax[i][k])
            j += 1
            if j == i:
                j += 1
        i += 1
    plt.savefig(savePath, format="svg")


def transformFormatToGeneral(intronIrInfo,
                             representProtein,
                             getUnfiltered=False,
                             cutoff=5):
    intronIrInfo["intronCoverage"] = intronIrInfo["intronCoverage"].map(
        lambda x: np.fromstring(x, sep=","))
    intronIrInfo["intronRetentionReadCounts"] = intronIrInfo[
        "intronRetentionReadCounts"].map(lambda x: np.fromstring(x, sep=","))
    intronIrInfo = intronIrInfo.reindex(
        ["Name", "intronRetentionReadCounts", "intronCoverage", "cluster"],
        axis=1)
    allContents = ""
    for singleGene in intronIrInfo.itertuples():
        i = 1
        geneName = representProtein[singleGene.Name]
        clusterName = singleGene.cluster
        for intronCoverage, intronRetentionCounts in zip(
                singleGene.intronCoverage,
                singleGene.intronRetentionReadCounts):
            intronName = geneName + "_intron" + str(i)
            intronCoverage = int(intronCoverage)
            intronRetentionCounts = int(intronRetentionCounts)
            allContents += f"{intronName}\t{intronCoverage}\t{intronRetentionCounts}\t{clusterName}\n"
            i += 1
    intronIrInfo = pd.read_table(StringIO(allContents), header=None)
    intronIrInfo.columns = [
        "Name", "readCoverage", "readRetentionCounts", "cluster"
    ]
    intronIrInfo["readIrRatio"] = (intronIrInfo["readRetentionCounts"] /
                                   intronIrInfo["readCoverage"])
    intronIrInfo = intronIrInfo.reindex(
        [
            "Name", "readCoverage", "readRetentionCounts", "readIrRatio",
            "cluster"
        ],
        axis=1,
    )
    intronIrInfo_ = intronIrInfo.query("readCoverage >= @cutoff")
    intronIrInfo_ = getCommonNameLine(intronIrInfo_)
    if getUnfiltered:
        return intronIrInfo, intronIrInfo_
    else:
        return intronIrInfo_


@click.command()
@click.option("--adata", "ADATA_PATH")
@click.option("--ir", "IR_INFO_PATH")
@click.option("--config", "CONFIG_PATH")
@click.option("--outDir", "OUTPUT_DIR_PATH")
@click.option("-c",
              "USE_LOUVAIN",
              multiple=True,
              default=['all'],
              show_default=True)
@click.option("--cutoff",
              "CUT_OFF",
              default=5,
              type=int,
              help="common gene/intron coverage reads cut off",
              show_default=True)
def main(ADATA_PATH, IR_INFO_PATH, USE_LOUVAIN, OUTPUT_DIR_PATH, CUT_OFF,
         CONFIG_PATH):
    """
    用于从10X nanopore数据中获得不同cluster ir 比较的情况

    示例:
    
        python plotFigureBasedOnAdata.py --adata /public/home/liuzj/projects/singleCell/02_result/root/step7_scanpy/clusterInfo.h5ad --ir /public/home/liuzj/projects/singleCell/02_result/root/step6_nanoporeResult/step14_getIrInfo/irInfo.tsv -c 7 -c 10 -c 16 -c 4 -c 9 -c 13 -c 14 --config ./plotFigureBasedOnAdata.yaml --cutoff 5 --outDir /public/home/liuzj/projects/singleCell/02_result/root/step7_scanpy/clusterIrInfo/
    """
    sh.rm('-rf', OUTPUT_DIR_PATH)
    sh.mkdir(OUTPUT_DIR_PATH)

    configFile = yaml.load(open(CONFIG_PATH, "r"))
    REPRE_BED_PATH = configFile["REPRE_BED_PATH"]
    SELECTED_GENE = configFile["SELECTED_GENE"]
    SELECTED_INTRON = configFile["SELECTED_INTRON"]

    representProteinFile = REPRE_BED_PATH
    representProtein = pd.read_table(representProteinFile,
                                     usecols=[3],
                                     header=None)
    representProtein.index = representProtein[3].str.split(".").str[0]
    representProtein = representProtein.to_dict()[3]

    adata = sc.read_h5ad(ADATA_PATH)

    irInfo = pd.read_table(IR_INFO_PATH)



    print(USE_LOUVAIN)

    if USE_LOUVAIN[0] == 'all':
        useLouvain = list(adata.obs["louvain"].unique())
    else:
        useLouvain = [str(x) for x in USE_LOUVAIN]
        adata = adata[adata.obs["louvain"].isin(useLouvain)]

    irInfo = irInfo.query("GeneExonCounts > 1")
    
    irInfo["barcode"] = irInfo["Name"].str.split("_|-").str[0]
    irInfo['exonOverlapCounts'] = irInfo['ExonOverlapInfo'].str.split(',').map(lambda x:len(x))
    irInfo = irInfo.query("exonOverlapCounts == GeneExonCounts")
    irInfo.drop('exonOverlapCounts', axis=1, inplace=True)


    selectedGene = set(
        pd.read_table(SELECTED_GENE, header=None)[0].str.split(".").str[0])
    irInfo = irInfo.query("geneId in @selectedGene")

    adata = addCellIrInfo(adata, irInfo)

    #    ###
    #    绘制cluster中每个cell的IR情况
    #    ###

    irSvg = f"{OUTPUT_DIR_PATH}cellIrInfo.svg"
    ax = sc.pl.umap(
        adata,
        color="irRatio",
        title="Intron Retention Ratio ",
        color_map="Reds",
        return_fig=True,
    )
    ax.savefig(irSvg, format="svg")
    plt.cla()
    plt.close("all")

    adata.obs.to_csv(f"{OUTPUT_DIR_PATH}cellIrInfo.tsv", sep="\t")

    allClusterCellIrDistributionSvg = (
        f"{OUTPUT_DIR_PATH}cellAllClusterIrDistribution.svg")
    plotCellClusterIr(adata, allClusterCellIrDistributionSvg)
    plt.cla()
    plt.close("all")

    selectedClusterCellIrDistributionSvg = (
        f"{OUTPUT_DIR_PATH}cellSelectedClusterIrDistribution.svg")
    plotCellClusterIr(adata, selectedClusterCellIrDistributionSvg, useLouvain)
    plt.cla()
    plt.close("all")

    #    ###
    #    绘制cluster中每个gene及intron的IR情况
    #    ###
    irInfo["barcode"] = irInfo["barcode"] + "-1"
    irInfo = irInfo.join(adata.obs["louvain"], on="barcode", how="right")

    geneIrInfoFiltered, intronIrInfo = getGeneIntronIrInfo(irInfo)

    intronIrInfo.to_csv(f"{OUTPUT_DIR_PATH}geneAllClusterIrDistribution.tsv",
                        sep="\t",
                        index=False)
    geneIrInfoFiltered.to_csv(
        f"{OUTPUT_DIR_PATH}geneFilteredAllClusterIrDistribution.tsv",
        sep="\t",
        index=False,
    )

    geneAllClusterFigurePath = f"{OUTPUT_DIR_PATH}geneAllClusterIrDistribution.svg"
    plotGeneClusterIr(geneIrInfoFiltered, geneAllClusterFigurePath)
    plt.cla()
    plt.close("all")

    geneAllClusterFigurePath = f"{OUTPUT_DIR_PATH}geneSelectedClusterIrDistribution.svg"
    plotGeneClusterIr(geneIrInfoFiltered, geneAllClusterFigurePath, useLouvain)
    plt.cla()
    plt.close("all")

    selectedIntron = set(pd.read_table(SELECTED_INTRON, header=None)[0])

    intronIrInfoParsed, intronIrInfoFiltered = transformFormatToGeneral(
        intronIrInfo, representProtein, True, cutoff=CUT_OFF)

    intronIrInfoParsed = intronIrInfoParsed.query("Name in @selectedIntron")
    intronIrInfoFiltered = intronIrInfoFiltered.query(
        "Name in @selectedIntron")

    intronIrInfoParsed.to_csv(
        f"{OUTPUT_DIR_PATH}intronAllClusterIrDistribution.tsv",
        sep="\t",
        index=False)

    intronAllClusterFigurePath = f"{OUTPUT_DIR_PATH}intronAllClusterIrDistribution.svg"
    plotGeneClusterIr(intronIrInfoFiltered,
                      intronAllClusterFigurePath,
                      yLabel="Intron")
    plt.cla()
    plt.close("all")

    intronAllClusterFigurePath = (
        f"{OUTPUT_DIR_PATH}intronSelectedClusterIrDistribution.svg")
    plotGeneClusterIr(intronIrInfoFiltered,
                      geneAllClusterFigurePath,
                      useLouvain,
                      yLabel="Intron")
    plt.cla()
    plt.close("all")

    (
        geneIrCompareWithinClusterInfo,
        intronIrCompareWithinClusterInfo,
    ) = prepareForPlotScatterPlot(irInfo, useLouvain, representProtein,
                                  CUT_OFF)
    geneIrCompareWithinClusterInfoPickle, intronIrCompareWithinClusterInfoPickle = (
        f"{OUTPUT_DIR_PATH}geneIrCompareWithinClusterInfo.pkl",
        f"{OUTPUT_DIR_PATH}intronIrCompareWithinClusterInfo.pkl",
    )
    with open(geneIrCompareWithinClusterInfoPickle,
              "wb") as geneIrFile, open(intronIrCompareWithinClusterInfoPickle,
                                        "wb") as intronIrFile:
        pickle.dump(geneIrCompareWithinClusterInfo, geneIrFile)
        pickle.dump(intronIrCompareWithinClusterInfo, intronIrFile)

    #    ###
    #    绘制不同cluster中每个gene及intron的IR情况的对比
    #    ###
    geneSelectedCompareFigurePath = f"{OUTPUT_DIR_PATH}geneSelectedCompare.svg"
    plotAllClusterCompareOthers(geneIrCompareWithinClusterInfo,
                                geneSelectedCompareFigurePath)
    plt.cla()
    plt.close("all")

    geneSelectedCompareFigurePath = f"{OUTPUT_DIR_PATH}intronSelectedCompare.svg"
    plotAllClusterCompareOthers(intronIrCompareWithinClusterInfo,
                                geneSelectedCompareFigurePath)
    plt.cla()
    plt.close("all")


main()
