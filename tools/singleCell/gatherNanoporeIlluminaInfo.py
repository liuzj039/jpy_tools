'''
@Date: 2020-07-29 14:41:34
@LastEditors: liuzj
@LastEditTime: 2020-07-29 15:01:15
@Author: liuzj
@desc: 用于获得整合上游的分析 获得每个cluster的每个gene的10X counts Nanopore counts 和 Nanopore ir
@FilePath: /liuzj/projects/singleCellEndo/01_pipeline/pipeline0630/needAdata/gatherNanoporeIlluminaInfo.py
'''

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
import click


def getClusterIlluminaGeneCounts(clusterCell, adata, clusterName):
    adata = adata[adata.obs.index.isin(clusterCell)]
    adata.var['illuminaCounts'] = adata.X.sum(0).A1
    adata.var['cluster'] = clusterName
    adata.var.drop(['gene_ids','feature_types','genome'], axis=1, inplace=True)
    return adata.var


@click.command()
@click.option('-g', 'GENE_DIST', help = 'geneAllClusterIrDistribution.tsv, plotFigureBasedOnAdata.py的输出')
@click.option('-c', 'CLUSTER_FILE', help = 'cellIrInfo.tsv, 同上')
@click.option('-a', 'RAW_CELLRANGER_FILE', help = 'cellranger count h5 raw')
@click.option('-o', 'OUT_PATH', help = 'out tsv')
def main(GENE_DIST, CLUSTER_FILE, RAW_CELLRANGER_FILE, OUT_PATH):
    clusterInfo = pd.read_table(CLUSTER_FILE, index_col=0)
    clusterInfo = clusterInfo.loc[:, ["louvain"]]
    clusterInfo.reset_index(inplace=True)
    clusterInfo = clusterInfo.groupby("louvain")["index"].agg(lambda x: set(x))

    adata = sc.read_10x_h5(
        RAW_CELLRANGER_FILE,
        genome=None,
        gex_only=True,
    )

    illuminaCounts = pd.concat(
        [getClusterIlluminaGeneCounts(x, adata, y) for y, x in clusterInfo.iteritems()]
    )
    illuminaCounts.reset_index(inplace=True)

    geneDist = pd.read_table(GENE_DIST, index_col=0)
    geneDist = geneDist.reindex(["readCoverage", "readIrRatio", "cluster"], axis=1)
    geneDist.reset_index(inplace=True)
    geneDist = geneDist.merge(
        illuminaCounts,
        how="left",
        left_on=["cluster", "Name"],
        right_on=["cluster", "index"],
    )
    geneDist = geneDist.reindex(
        ["Name", "illuminaCounts", "readCoverage", "readIrRatio", "cluster"], axis=1
    )
    geneDist.columns = [
        "Name",
        "illuminaCounts",
        "nanoporeCounts",
        "readIrRatio",
        "cluster",
    ]

    geneDist.to_csv(OUT_PATH, sep="\t", index=None)

main()