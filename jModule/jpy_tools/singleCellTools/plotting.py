"""
plotting tools
"""
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


def obsmToObs(
    ad: sc.AnnData,
    key_obsm: str,
    embedding: Optional[List[str]] = None,
    prefix: str = "",
) -> sc.AnnData:
    import scipy.sparse as ss

    if not embedding:
        embedding = list(ad.obsm.keys())
    ad_tmp = sc.AnnData(ss.csc_matrix(ad.shape), obs=ad.obs.copy(), var=ad.var.copy())
    for key in embedding:
        ad_tmp.obsm[key] = ad.obsm[key].copy()

    ad_tmp.obs = ad.obsm[key_obsm].combine_first(ad_tmp.obs).copy()
    ad_tmp.obs = ad_tmp.obs.rename(
        columns={x: f"{prefix}{x}" for x in ad.obsm[key_obsm].columns}
    )
    return ad_tmp


def getPartialByPos(ad: sc.AnnData, rightUpper: List[float], leftBottom: List[float]):
    return ad[
        (np.array([4, 11]) < ad.obsm["X_umap"]).all(1)
        & (ad.obsm["X_umap"] < np.array([5, 12])).all(1)
    ]


def plotCellScatter(
    adata,
    plotFeature: Sequence[str] = ["n_counts", "n_genes", "percent_ct"],
    func_ct=lambda x: (
        (x.var_names.str.startswith("ATCG")) | (x.var_names.str.startswith("ATMG"))
    ),
    batch=None,
):
    adata.obs = adata.obs.assign(n_genes=(adata.X > 0).sum(1), n_counts=adata.X.sum(1))
    # adata.var = adata.var.assign(n_cells=(adata.X > 0).sum(0))
    ctGene = func_ct(adata)

    adata.obs["percent_ct"] = np.sum(adata[:, ctGene].X, axis=1) / np.sum(
        adata.X, axis=1
    )
    sc.pl.violin(adata, plotFeature, multi_panel=True, jitter=0.4, groupby=batch)


def plotLabelPercentageInCluster(
    adata, groupby, label, labelColor: Optional[dict] = None, needCounts=True
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
        adata.obs.groupby(groupby)[label].apply(lambda x: x.value_counts()).unstack()
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
            plt.Rectangle((0, 0), 1, 1, fc=labelColor[singleLabel], edgecolor="none")
        )
        legendLabelLs.append(singleLabel)
    legendHandleLs, legendLabelLs = legendHandleLs[::-1], legendLabelLs[::-1]
    plt.legend(legendHandleLs, legendLabelLs, bbox_to_anchor=[1, 1], frameon=False)
    plt.xlabel(groupby.capitalize())
    plt.ylabel(f"Percentage")
    if needCounts:
        for i, label in enumerate(groupbyWithLabelCounts_CumsumPercDf.index):
            plt.text(
                i,
                105,
                f"$\it{{N}}$ = {len(adata[adata.obs[groupby] == label])}",
                rotation=90,
                ha="center",
                va="bottom",
            )
    sns.despine(top=True, right=True)
    return ax


def plotClusterSankey(
    adata: anndata.AnnData,
    clusterNameLs: Sequence[str],
    figsize=[5, 5],
    defaultJupyter: Literal["notebook", "lab"] = "notebook",
    **dargs
):
    """
    Returns
    -------
    pyecharts.charts.basic_charts.sankey.Sankey
        Utilize Function render_notebook can get the final figure
    """
    from ..otherTools import sankeyPlotByPyechart

    df = adata.obs.filter(clusterNameLs).astype(str)

    [basic.setadataColor(adata, x) for x in clusterNameLs]
    colorDictLs = [basic.getadataColor(adata, x) for x in clusterNameLs]

    sankey = sankeyPlotByPyechart(
        df, clusterNameLs, figsize, colorDictLs, defaultJupyter=defaultJupyter, **dargs
    )
    return sankey


def plotGeneInDifferentBatch(
    ad, ls_gene, batchKey, layer, figsize, ls_name=[], cmap="Reds", ncols=2, **dt_arg
):
    from math import ceil

    ls_batch = list(ad.obs[batchKey].unique())
    ls_batch = [ls_batch, *[[x] for x in ls_batch]]

    nrows = ceil(len(ls_batch) / ncols)

    if not ls_name:
        ls_name = ls_gene
    assert len(ls_name) == len(
        ls_gene
    ), "The length of `ls_name` is not equal to `ls_gene`"

    xMin = ad.obsm["X_umap"][:, 0].min()
    xMax = ad.obsm["X_umap"][:, 0].max()
    yMin = ad.obsm["X_umap"][:, 1].min()
    yMax = ad.obsm["X_umap"][:, 1].max()
    if 'size' not in dt_arg:
        size = 120000 / len(ad)
        dt_arg['size'] = size

    for gene, name in zip(ls_gene, ls_name):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        axs = axs.reshape(-1)

        for batch, ax in zip(ls_batch, axs):
            _ad = ad[ad.obs[batchKey].isin(batch)]
            batch = batch[0] if len(batch) == 1 else "all"
            sc.pl.umap(
                _ad,
                color=gene,
                title=f"{name}\n({batch})",
                layer=layer,
                cmap=cmap,
                ax=ax,
                show=False,
                vmax=ad[:, gene].to_df(layer).quantile(0.999),
                vmin=0,
                **dt_arg,
            )
            plt.sca(ax)
            plt.xlim(xMin - 0.5, xMax + 0.5)
            plt.ylim(yMin - 0.5, yMax + 0.5)
            
        plt.tight_layout()
        plt.show()
        plt.close()


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
    from ..otherTools import addColorLegendToAx

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


def clustermap(
    ad: sc.AnnData,
    dt_gene: Mapping[str, List[str]],
    obsAnno: Union[str, List[str]],
    layer: str,
    space_obsAnnoLegend: float = 0.12,
    figsize=(10, 10),
    cbarPos=(0.72, 0.15, 0.01, 0.18),
    sort=True,
    dt_geneColor: Optional[Mapping[str, str]] = None,
    add_gene_name: bool = True,
    col_label: bool = False,
    **dt_arg,
):
    from ..otherTools import addColorLegendToAx

    if isinstance(obsAnno, str):
        obsAnno = [obsAnno]
    df_geneModule = pd.DataFrame(
        [(x, z) for x, y in dt_gene.items() for z in y], columns=["module", "gene"]
    ).set_index("gene")
    if sort:
        df_cellAnno = ad.obs[obsAnno].sort_values(obsAnno)
        ad = ad[df_cellAnno.index]
    else:
        df_cellAnno = ad.obs[obsAnno]

    df_mtx = ad.to_df(layer).loc[:, df_geneModule.index]

    for anno in obsAnno:
        dt_color = basic.getadataColor(ad, anno)
        df_cellAnno[anno] = df_cellAnno[anno].map(dt_color)

    if not dt_geneColor:
        from scanpy.plotting import palettes

        length = len(dt_gene)
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
        dt_geneColor = {x: y for x, y in zip(dt_gene.keys(), palette)}

    df_geneModuleChangeColor = df_geneModule.assign(
        module=lambda df: df["module"].map(dt_geneColor)
    )

    axs = sns.clustermap(
        df_mtx,
        col_cluster=False,
        cmap="Reds",
        col_colors=df_geneModuleChangeColor,
        row_colors=df_cellAnno,
        dendrogram_ratio=0.1,
        figsize=figsize,
        cbar_pos=cbarPos,
        **dt_arg,
    )
    _dt = df_geneModule.groupby("module").apply(len).to_dict()
    dt_geneCounts = {x: _dt[x] for x in dt_gene.keys()}
    if not dt_geneColor:
        from scanpy.plotting import palettes

        length = len(dt_gene)
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
        dt_geneColor = {x: y for x, y in zip(dt_gene.keys(), palette)}
    df_geneModule["module"] = df_geneModule["module"].map(dt_geneColor)

    if add_gene_name:
        plt.sca(axs.ax_col_colors)
        pos_current = 0
        for name, counts in dt_geneCounts.items():
            pos_next = pos_current + counts
            plt.text(
                (pos_current + pos_next) / 2,
                -0.2,
                name,
                rotation=90,
                va="bottom",
                ha="center",
            )
            pos_current = pos_next
            plt.yticks([])
    plt.sca(axs.ax_col_colors)
    plt.xticks([])
    plt.yticks([])

    for i, anno in enumerate(obsAnno):
        dt_color = basic.getadataColor(ad, anno)
        addColorLegendToAx(
            axs.ax_heatmap,
            anno,
            dt_color,
            1,
            bbox_to_anchor=(1.05 + space_obsAnnoLegend * i, 1),
            frameon=False,
        )

    plt.sca(axs.ax_heatmap)
    if not col_label:
        plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    return axs
