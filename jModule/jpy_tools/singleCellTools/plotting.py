"""
plotting tools
"""
from logging import log
from math import e
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from io import StringIO
from concurrent.futures import ProcessPoolExecutor as Mtp, thread
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
from ..otherTools import F
import collections
from xarray import corr
import matplotlib as mpl
from more_itertools import chunked
import patchworklib as pw
import scipy.sparse as ss
from joblib import Parallel, delayed
import marsilea as ma
import marsilea.plotter as mp
import legendkit
import seaborn.objects as so
from . import basic, geneEnrichInfo
from ..soExt import Axhline, Axvline

def umapMultiBatch(
    ad,
    ls_gene,
    groups,
    layer,
    needAll=True,
    ncols=2,
    figsize=(4, 3),
    cmap="Reds",
    dir_result=None,
    ls_title=None,
    size=None,
    horizontal=False,
    vmin=None,
    vmax=None,
    format="png",
    dpi="figure",
    fileNameIsTitle=False,
    show=True,
    clearBk=True,
    cbRatio = 0.01,
    supTitleXPos = 0.5,
    disableSuptitle=False,
    disableProgressBar=False,
):
    import gc

    if clearBk:
        pw.clear()
        gc.collect()
    if isinstance(ls_gene, str):
        ls_gene = [ls_gene]
    if not ls_title:
        ls_title = ls_gene
    if isinstance(ls_title, str):
        ls_title = [ls_title]
    if len(ls_gene) <= 1:
        disableProgressBar = True
    else:
        disableProgressBar = disableProgressBar
    if groups[1] is None:
        groups[1] = ad.obs[groups[0]].astype("category").cat.categories.to_list()
    dt_adObs = ad.obs.groupby(groups[0]).apply(lambda df: df.index.to_list()).to_dict()
    dt_adObs = {x: dt_adObs[x] for x in groups[1]}
    dt_ad = {x: ad[y] for x, y in dt_adObs.items()}
    if needAll:
        dt_ad = dict(All=ad, **dt_ad)
    if not size:
        size = 12e4 / len(ad)

    vmin = vmin if vmin else 0
    vmaxSpecify = vmax
    for gene, title in tqdm(
        zip(ls_gene, ls_title), total=len(ls_gene), disable=disableProgressBar
    ):
        if not vmaxSpecify:
            vmax = ad[:, gene].to_df(layer).iloc[:, 0]
            vmax = sorted(list(vmax))
            vmax = vmax[-2]
            if vmax <= 1:
                vmax = 1
        else:
            vmax = vmaxSpecify
        ls_ax = []
        for sample, _ad in dt_ad.items():
            ax = pw.Brick(figsize=figsize)
            sc.pl.umap(ad, ax=ax, show=False, size=size)
            sc.pl.umap(
                _ad,
                color=gene,
                show=False,
                ax=ax,
                title=sample,
                layer=layer,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                size=size,
            )
            plt.close()
            ls_ax.append(ax)

        ax_add_count = len(ls_ax)%ncols
        if ax_add_count != 0:
            for count in range(ncols - ax_add_count):
                ax_add = pw.Brick(figsize=figsize)
                ax_add.axis('off')
                ls_ax.append(ax_add)
        ls_ax = chunked(ls_ax, ncols) | F(list)

        _bc = pw.param["margin"]
        pw.param["margin"] = 0.3
        if len(ls_ax) == 1:
            axs = pw.stack(ls_ax[0])
        else:
            axs = pw.stack([pw.stack(x) for x in ls_ax], operator="/")
            ls_name = list(axs.bricks_dict.keys())

        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        if horizontal:
            ax_cb = pw.Brick(figsize=(1, cbRatio))
            mpl.colorbar.ColorbarBase(
                ax_cb, cmap=cmap, norm=norm, orientation="horizontal"
            )

            pw.param["margin"] = 0.1
            axs = axs / ax_cb
            pw.param["margin"] = _bc
        else:
            ax_cb = pw.Brick(figsize=(cbRatio, 1))
            mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)

            pw.param["margin"] = 0.1
            axs = axs | ax_cb
            pw.param["margin"] = _bc

        if not disableSuptitle:
            axs.case.set_title(title, x=supTitleXPos, pad=10, size=16)
        if dir_result:
            if fileNameIsTitle:
                fileName = title.replace("\n", "_").replace("/", "_")
            else:
                fileName = gene
            axs.savefig(f"{dir_result}/{fileName}.{format}", dpi=dpi)
        elif show:
            pw.show(axs.savefig())
        else:
            pass
    if len(ls_gene) == 1:
        return axs


def saveUmapMultiBatch(ad, threads, batchSize, ls_gene, ls_title, layer, backend = 'multiprocessing', **dt_kwargs):
    from more_itertools import chunked, sliced

    def _iterAd(ad, batchSize, ls_gene, ls_title, layer):
        _ad = ad[:, ls_gene]
        ad = sc.AnnData(
            _ad.X,
            obs=_ad.obs,
            var=_ad.var,
            obsm={"X_umap": ad.obsm["X_umap"]},
            layers={layer: _ad.layers[layer]},
        )
        for ls_chunkGene, ls_chunkTitle in tqdm(
            zip(chunked(ls_gene, batchSize), chunked(ls_title, batchSize)),
            total=len(ls_gene) // batchSize,
        ):
            yield ad[:, ls_chunkGene], ls_chunkGene, ls_chunkTitle

    # assert threads * batchSize >= len(ls_gene), "threads * batchSize must be greater than or equal to the number of genes"
    if not ls_title:
        ls_title = ls_gene

    for ad_chunk, ls_chunkGene, ls_chunkTitle in _iterAd(
        ad, batchSize, ls_gene, ls_title, layer
    ):
        it_chunkGene = sliced(ls_chunkGene, threads)
        it_chunkTitle = sliced(ls_chunkTitle, threads)
        Parallel(n_jobs=threads, backend=backend)(
            delayed(umapMultiBatch)(
                ad_chunk[:, ls_processGene].copy(),
                ls_gene=ls_processGene,
                ls_title=ls_processTitle,
                layer=layer,
                **dt_kwargs,
            )
            for ls_processGene, ls_processTitle in zip(it_chunkGene, it_chunkTitle)
        )

    #     ls_batchGene.append(ls_chunkGene)
    #     ls_batchTitle.append(ls_chunkTitle)
    #     ls_batchAd.append(ad)
    # for ad_batch, ls_

    # Parallel(n_jobs=threads)(
    #     delayed(umapMultiBatch)(
    #         _ad, ls_gene=ls_gene, ls_title=ls_title, layer=layer, **dt_kwargs
    #     )
    #     for _ad, ls_gene, ls_title in zip(ls_batchAd, ls_batchGene, ls_batchTitle)
    # )


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
    ad_tmp.uns = ad.uns
    for key in embedding:
        ad_tmp.obsm[key] = ad.obsm[key].copy()

    ad_tmp.obs = ad.obsm[key_obsm].combine_first(ad_tmp.obs).copy()
    ad_tmp.obs = ad_tmp.obs.rename(
        columns={x: f"{prefix}{x}" for x in ad.obsm[key_obsm].columns}
    )
    ad_tmp.uns["plot_obsm"] = ad_tmp.obsm[key_obsm].columns
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
    assert ss.issparse(adata.X), "X layer must be a scipy.sparse matrix"
    adata.obs = adata.obs.assign(
        n_genes=(adata.X > 0).sum(1).A1, n_counts=adata.X.sum(1).A1
    )
    # adata.var = adata.var.assign(n_cells=(adata.X > 0).sum(0))
    ctGene = func_ct(adata)

    adata.obs["percent_ct"] = (
        np.sum(adata[:, ctGene].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    )
    sc.pl.violin(adata, plotFeature, multi_panel=True, jitter=0.4, groupby=batch)


def plotLabelPercentageInCluster(
    adata,
    groupby,
    label,
    labelColor: Optional[dict] = None,
    needCounts=True,
    ax=None,
    dt_kwargsForLegend={"bbox_to_anchor": [1, 1]},
    swapAxes=False,
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
    if not swapAxes:
        for singleLabel in groupbyWithLabelCounts_CumsumPercDf.columns[::-1]:
            ax = sns.barplot(
                x=groupbyWithLabelCounts_CumsumPercDf.index,
                y=groupbyWithLabelCounts_CumsumPercDf[singleLabel],
                color=labelColor[singleLabel],
                ax=ax,
            )
            plt.sca(ax)
            legendHandleLs.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc=labelColor[singleLabel], edgecolor="none", label=singleLabel
                )
            )
            legendLabelLs.append(singleLabel)
        legendHandleLs, legendLabelLs = legendHandleLs[::-1], legendLabelLs[::-1]
        # plt.legend(legendHandleLs, legendLabelLs, frameon=False, **dt_kwargsForLegend)
        plt.legend(handles=legendHandleLs, frameon=False, **dt_kwargsForLegend)

        plt.xlabel(groupby.capitalize())
        plt.ylabel(f"Percentage")
        if needCounts:
            for i, label in enumerate(groupbyWithLabelCounts_CumsumPercDf.index):
                plt.text(
                    i,
                    105,
                    f"$\it{{N}}$ = {adata[adata.obs[groupby] == label].shape[0]}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
    else:
        for singleLabel in groupbyWithLabelCounts_CumsumPercDf.columns[::-1]:
            ax = sns.barplot(
                y=groupbyWithLabelCounts_CumsumPercDf.index,
                x=groupbyWithLabelCounts_CumsumPercDf[singleLabel],
                color=labelColor[singleLabel],
                ax=ax,
            )
            plt.sca(ax)
            legendHandleLs.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc=labelColor[singleLabel], edgecolor="none", label=singleLabel
                )
            )
            legendLabelLs.append(singleLabel)
        # plt.legend(
        #     legendHandleLs[::-1],
        #     legendLabelLs[::-1],
        #     frameon=False,
        #     **dt_kwargsForLegend,
        # )
        plt.legend(handles=legendHandleLs[::-1], frameon=False, **dt_kwargsForLegend)

        plt.ylabel(groupby.capitalize())
        plt.xlabel(f"Percentage")
        if needCounts:
            for i, label in enumerate(groupbyWithLabelCounts_CumsumPercDf.index):
                plt.text(
                    101,
                    i,
                    f"$\it{{N}}$ = {adata[adata.obs[groupby] == label].shape[0]}",
                    rotation=0,
                    ha="left",
                    va="center",
                )
    # sns.despine(top=True, right=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plotClusterSankey(
    adata: anndata.AnnData,
    clusterNameLs: Sequence[str],
    figsize=[5, 5],
    defaultJupyter: Literal["notebook", "lab"] = "notebook",
    **dargs,
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
    if "size" not in dt_arg:
        size = 120000 / len(ad)
        dt_arg["size"] = size

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
    row_cluster=False,
    col_cluster=False,
    **clustermapParamsDt: Dict,
):
    import sys
    from ..otherTools import addColorLegendToAx

    sys.setrecursionlimit(100000)

    matrixLabel = matrixLabel.rstrip("_consensus") + "_consensus"

    dt_color = adata.obs.filter(clusterResultLs)
    for clusterName in clusterResultLs:
        basic.setadataColor(adata, clusterName)
        clusterColorMapDt = basic.getadataColor(adata, clusterName)
        dt_color[clusterName] = dt_color[clusterName].map(clusterColorMapDt)

    cellIndexOrderSq = adata.obs.sort_values(matrixLabel.rstrip("_consensus")).index
    consensusMatrix = pd.DataFrame(
        adata.obsm[matrixLabel], index=adata.obs.index
    ).reindex(cellIndexOrderSq)

    g = sns.clustermap(
        consensusMatrix,
        cmap=cmap,
        metric=metrix,
        row_colors=dt_color,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
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
    space_obsAnnoLegend: Union[float, List[float]] = 0.12,
    figsize=(10, 10),
    cbarPos=(0.02, 0.8, 0.05, 0.18),
    sort=True,
    dt_geneColor: Optional[Mapping[str, str]] = None,
    add_gene_name: bool = True,
    reverseGeneNameInColColor: bool = False,
    col_label: bool = False,
    legendAlign: Literal["h", "v"] = "h",
    addSplitLine=True,
    addGeneSplitLine=False,
    row_cluster=False,
    col_cluster=False,
    addObsLegend=True,
    forceShowModuleColor=False,
    cmap="Reds",
    legendPos = 1.05,
    **dt_arg,
):
    from ..otherTools import addColorLegendToAx

    if isinstance(obsAnno, str):
        obsAnno = [obsAnno]
    if isinstance(space_obsAnnoLegend, float):
        space_obsAnnoLegend = [space_obsAnnoLegend] * len(obsAnno)
    if addSplitLine:
        row_cluster = False
        splitBasedOn = obsAnno[0]
    if addGeneSplitLine:
        col_cluster = False
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
    if (len(dt_gene) == 1):
        if not forceShowModuleColor:
            df_geneModuleChangeColor = None
    axs = sns.clustermap(
        df_mtx,
        cmap=cmap,
        col_colors=df_geneModuleChangeColor,
        row_colors=df_cellAnno,
        figsize=figsize,
        cbar_pos=cbarPos,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
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
        if reverseGeneNameInColColor:
            _dt_geneCounts = {x:dt_geneCounts[x] for x in list(dt_geneCounts.keys())[::-1]}
            plt.sca(axs.ax_col_colors)
            pos_current = 0
            for name, counts in _dt_geneCounts.items():
                pos_next = pos_current + counts
                plt.text(
                    (pos_current + pos_next) / 2,
                    0.9,
                    name,
                    rotation=0,
                    va="bottom",
                    ha="center",
                )
                pos_current = pos_next
                plt.yticks([])
        else:
            plt.sca(axs.ax_col_colors)
            pos_current = 0
            for name, counts in dt_geneCounts.items():
                pos_next = pos_current + counts
                plt.text(
                    (pos_current + pos_next) / 2,
                    0.9,
                    name,
                    rotation=0,
                    va="bottom",
                    ha="center",
                )
                pos_current = pos_next
                plt.yticks([])

    if not ((len(dt_gene) == 1) | (not forceShowModuleColor)):
        plt.sca(axs.ax_col_colors)
        plt.xticks([])
        plt.yticks([])
    if addObsLegend:
        legendPox = [legendPos, 1]
        for i, (anno, space) in enumerate(zip(obsAnno, space_obsAnnoLegend)):
            dt_color = basic.getadataColor(ad, anno)
            addColorLegendToAx(
                axs.ax_heatmap,
                anno,
                dt_color,
                1,
                bbox_to_anchor=legendPox,
                frameon=False,
                ha="left",
            )
            if legendAlign == "h":
                legendPox = [legendPos + space, 1]
            elif legendAlign == "v":
                legendPox = [legendPos, 1 - space]

    plt.sca(axs.ax_heatmap)
    if not col_label:
        plt.xticks([])
    plt.yticks([])
    plt.xlabel("")

    ## add split line
    xMin, xMax = axs.ax_heatmap.get_xlim()
    yMin, yMax = axs.ax_heatmap.get_ylim()
    plt.axvline(xMin, color="black", lw=1, alpha=0.7)
    plt.axvline(xMax, color="black", lw=1, alpha=0.7)
    plt.axhline(yMin, color="black", lw=1, alpha=0.7)
    plt.axhline(yMax, color="black", lw=1, alpha=0.7)
    if addSplitLine:
        dt_obsCounts = ad.obs[splitBasedOn].value_counts(sort=False).to_dict()
        yPos = 0
        for i, (name, counts) in enumerate(dt_obsCounts.items()):
            yPos = yPos + counts
            plt.axhline(yPos, color="black", lw=1, alpha=0.7)
    if addGeneSplitLine:
        dt_geneCounts = {x: len(y) for x, y in dt_gene.items()}
        xPos = 0
        for i, (name, counts) in enumerate(dt_geneCounts.items()):
            xPos = xPos + counts
            plt.axvline(xPos, color="black", lw=1, alpha=0.7)
    if cbarPos is (0.02, 0.8, 0.05, 0.18):
        axs.ax_cbar.set_position(
            [
                axs.ax_heatmap.get_position().x1 * legendPos,
                axs.ax_heatmap.get_position().y0,
                0.02,
                0.1,
            ]
        )
    return axs


def heatmap_rank(
    ad: sc.AnnData,
    dt_gene: Mapping[str, List[str]],
    obsAnno: Union[str, List[str]],
    sortby: str,
    layer: str,
    space_obsAnnoLegend: float = 0.12,
    figsize=(10, 10),
    cbarPos=(0.9, 0.33, 0.1, 0.02),
    dt_geneColor: Optional[Mapping[str, str]] = None,
    standardScale="row",
    cbar_kws=dict(orientation="horizontal"),
    cmap="Spectral_r",
    bins=None,
    qcut=True,
    **dt_arg,
):
    from ..otherTools import addColorLegendToAx

    ad = ad[ad.obs[sortby].sort_values().index]

    if isinstance(obsAnno, str):
        obsAnno = [obsAnno]

    df_cellAnno = ad.obs[obsAnno]
    df_geneModule = (
        pd.Series(dt_gene)
        .explode()
        .to_frame()
        .reset_index()
        .set_index(0)
        .rename(columns=dict(index="module"))
    )

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

    df_geneModule["module"] = (
        df_geneModule["module"]
        .astype("category")
        .cat.reorder_categories(dt_gene.keys())
        .sort_values()
    )

    df_geneModuleChangeColor = df_geneModule.assign(
        module=lambda df: df["module"].map(dt_geneColor)
    )

    # _dt = df_geneModule.groupby("module").apply(len).to_dict()
    # dt_geneCounts = {x: _dt[x] for x in dt_gene.keys()}
    dt_geneCounts = df_geneModule["module"].value_counts(sort=False).to_dict()

    if not bins:
        mtx = ad[:, df_geneModuleChangeColor.index].to_df(layer).T
        axs = sns.clustermap(
            mtx,
            col_colors=df_cellAnno,
            row_colors=df_geneModuleChangeColor["module"],
            cmap=cmap,
            row_cluster=False,
            col_cluster=False,
            standard_scale=standardScale,
            figsize=figsize,
            cbar_pos=cbarPos,
            cbar_kws=cbar_kws,
            **dt_arg,
        )
        plt.sca(axs.ax_heatmap)
        plt.xticks([])
        plt.yticks([])
        plt.sca(axs.ax_col_colors)
        plt.xticks([])

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
    else:
        if qcut:
            sr_cutCate = pd.qcut(ad.obs[sortby], bins)
        else:
            sr_cutCate = pd.cut(ad.obs[sortby], bins)
        sr_cutCate = sr_cutCate.cat.rename_categories(
            new_categories=lambda x: (x.left + x.right) / 2
        )
        mtx = (
            ad[:, df_geneModuleChangeColor.index]
            .to_df(layer)
            .groupby(sr_cutCate)
            .agg("mean")
            .T
        )
        mtx = mtx.dropna(axis=1)

        axs = sns.clustermap(
            mtx,
            row_colors=df_geneModuleChangeColor["module"],
            cmap=cmap,
            row_cluster=False,
            col_cluster=False,
            standard_scale=standardScale,
            figsize=figsize,
            cbar_pos=cbarPos,
            cbar_kws=cbar_kws,
            **dt_arg,
        )
        plt.sca(axs.ax_heatmap)
        plt.yticks([])
        plt.xticks([])
        plt.xlabel("")

    plt.sca(axs.ax_row_colors)
    pos_current = 0
    for name, counts in dt_geneCounts.items():
        pos_next = pos_current + counts
        plt.text(
            -0.2,
            (pos_current + pos_next) / 2,
            f"{name}",
            va="center",
            ha="right",
        )
        pos_current = pos_next
        plt.xticks([])

    axs.ax_cbar.set_title("Gene Expression")
    axs.ax_cbar.tick_params(axis="x", length=10)

    return axs


def plotGeneModuleByNetworkx(
    df_adjacency: pd.DataFrame,
    cutoff: Optional[float] = None,
    ls_hubGenes: Optional[List[str]] = None,
    dt_needLabelNodes: Optional[Dict[str, str]] = None,
    dt_baseOptions: Dict[str, str] = {
        "node_color": "black",
        "node_size": 200,
        "width": 0.3,
    },
    dt_hubOptions: Dict[str, str] = {
        "node_color": "red",
        "node_size": 200,
        "width": 0.3,
    },
    dt_labelOptions: Dict[str, str] = {
        "font_size": 12,
        "bbox": {"ec": "k", "fc": "white", "alpha": 0.7},
    },
    ax=None,
    layoutMethod="kamada_kawai_layout",
    forceSustainOneLinkage=False,
):
    """This function plots a gene module using networkx

    Parameters
    ----------
    df_adjacency : pd.DataFrame
        a pandas dataframe with the gene names as the index and the column names. The values in the dataframe are the weights of the edges.
    cutoff : Optional[float]
        float, optional
    ls_hubGenes : Optional[List[str]]
        a list of hub genes.
    dt_needLabelNodes : Optional[Dict[str, str]]
        a dictionary of nodes that need to be labeled.
    dt_baseOptions : Dict[str, str]
        the default options for all nodes
    dt_hubOptions : Dict[str, str]
    dt_labelOptions : Dict[str, str]

    """
    import networkx as nx

    if ax is None:
        fig, ax = plt.subplots()
    if cutoff is None:
        cutoff = df_adjacency.max().min()

    df_adjacency = (
        df_adjacency.stack()
        .reset_index()
        .rename(columns={"level_0": "source", "level_1": "target", 0: "connectivity"})
    )

    df_sustained = df_adjacency.query("connectivity > @cutoff")
    if forceSustainOneLinkage:
        df_adjMax = df_adjacency.sort_values(
            ["source", "connectivity"], ascending=False
        ).drop_duplicates(subset=["source"])
        df_sustained = pd.concat([df_sustained, df_adjMax]).drop_duplicates(
            ["source", "target"]
        )

    G = nx.from_pandas_edgelist(df_sustained, edge_attr="connectivity")
    pos = eval(f"nx.drawing.layout.{layoutMethod}")(G)
    nx.draw(G, pos, ax=ax, **dt_baseOptions)
    if not ls_hubGenes is None:
        ls_hubGenes = [x for x in ls_hubGenes if x in G.nodes()]
        nx.draw(G, pos, nodelist=ls_hubGenes, ax=ax, **dt_hubOptions)
    if not dt_needLabelNodes is None:
        dt_needLabelNodes = {
            x: dt_needLabelNodes[x] for x in dt_needLabelNodes.keys() if x in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, dt_needLabelNodes, ax=ax, **dt_labelOptions)
    ax = plt.gca()
    return ax

def makeKdeForCluster(ad, key='Cluster', ls_cluster=None, levels=[0.1], ax=None, nobs=10000, palette=None, **dt_args):
    '''> This function takes a `AnnData` object, a key in the `obs` attribute, a list of clusters, a list of levels, and a matplotlib axis object, and returns a matplotlib axis object with kdes drawn on it

    Parameters
    ----------
    ad
        AnnData object
    key, optional
        the column name of the cluster labels
    ls_cluster
        list of clusters to plot. If None, plot all clusters.
    levels, optional
        the levels to draw the kde
    ax
        the axis to plot on

    '''
    assert ax, 'Please provide an axis object'
    if ls_cluster is None:
        ls_cluster = ad.obs[key].unique().tolist()
    if isinstance(ls_cluster, str):
        ls_cluster = [ls_cluster]

    if palette is None:
        dt_colors = basic.getadataColor(ad, key)
    else:
        dt_colors = palette

    _ad = ad[ad.obs.eval(f"{key} in @ls_cluster")]
    df_umap = _ad.obs[[key]].assign(UMAP_1=list(_ad.obsm['X_umap'][:,0]), UMAP_2=list(_ad.obsm['X_umap'][:,1]))
    nobs = min(nobs, df_umap.shape[0])
    df_umap = df_umap.groupby(key, group_keys=False).apply(lambda x: x.sample(frac=nobs/df_umap.shape[0], random_state=39))

    sns.kdeplot(df_umap, x='UMAP_1', y='UMAP_2', hue=key, levels=levels, common_norm=False, ax=ax, palette=dt_colors, **dt_args)

def makeEllipseForCluster(ad, key='Cluster', ls_cluster=None, std=3, ax=None, **dt_args):
    '''> This function takes a `AnnData` object, a key in the `obs` attribute, a list of clusters, a standard deviation, and a matplotlib axis object, and returns a matplotlib axis object with ellipses drawn on it

    Parameters
    ----------
    ad
        AnnData object
    key, optional
        the column name of the cluster labels
    ls_cluster
        list of clusters to plot. If None, plot all clusters.
    std, optional
        the number of standard deviations to determine the ellipse's radiuses.
    ax
        the axis to plot on

    '''
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='black', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor, edgecolor=edgecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    assert ax, 'Please provide an axis object'
    if ls_cluster is None:
        ls_cluster = ad.obs[key].unique().tolist()
    if isinstance(ls_cluster, str):
        ls_cluster = [ls_cluster]
    for cluster in ls_cluster:
        _ad = ad[ad.obs.eval("Cluster == @cluster")]

        confidence_ellipse(_ad.obsm['X_umap'][:, 0], _ad.obsm['X_umap'][:, 1], ax, std, **dt_args)
    return ax

def dotplotByMa(
        ad: sc.AnnData, ls_gene: List[str], groupby: str, layer:str, geneSymbol:Optional[str]=None, minMaxScale: bool=False, cmap='Reds',
        sizes=(0,1), dotsizes=(0,200), width=None, height=None, colorLegKws=dict(title='Mean expression\nin group', orientation='horizontal'),
        logExpInput: bool = True, sizeLegKws=dict(title='Fraction of cells\nin group', spacing='uniform', labelspacing=0.1), **kwargs
    ) -> ma.base.ClusterBoard:
    '''The function `dotplotByMa` creates a dot plot visualization of gene expression data in an AnnData object, grouped by a specified variable.

    Parameters
    ----------
    ad : sc.AnnData
        Anndata object containing the gene expression data.
    ls_genes : List[str]
        A list of gene names or symbols that you want to plot in the dotplot.
    groupby : str
        The `groupby` parameter is used to specify the column in the AnnData object that contains the grouping information. This column is used to group the cells for calculating mean expression and fraction of cells in each group.
    layer : str
        The `layer` parameter specifies the layer of the AnnData object that contains the gene expression values.
    geneSymbol : Optional[str]
        geneSymbol is an optional parameter that specifies the column name in the AnnData object's var attribute that contains the gene symbols. If provided, the function will use this column to map the gene symbols to the corresponding gene names in the dotplot visualization.
    minMaxScale : bool, optional
        The `minMaxScale` parameter is a boolean flag that determines whether to perform min-max scaling on the mean expression values. If set to `True`, the mean expression values will be scaled to the range [0, 1] using the minimum and maximum values in each column. If set to
    sizes
        The `sizes` parameter is a tuple that specifies the range of values to use for sizing the dots in the dot plot. The first value in the tuple represents the minimum size, and the second value represents the maximum size.
    dotsizes
        The `dotsizes` parameter in the `dotplotByMa` function is used to specify the size range of the dots in the dot plot. It is a tuple with two values, where the first value represents the minimum dot size and the second value represents the maximum dot size. The dots in
    width
        The width of the dot plot.
    height
        The `height` parameter is used to specify the height of the dot plot in pixels. It determines the vertical size of the dot plot visualization.
    colorLegKws
        The `colorLegKws` parameter is a dictionary that contains keyword arguments for customizing the color legend. It can include the following keys:
    sizeLegKws
        The `sizeLegKws` parameter is a dictionary that contains keyword arguments for configuring the size legend in the dot plot. It can include the following keys:

    Returns
    -------
        a `ma.base.ClusterBoard` object.

    '''
    from matplotlib.colors import Normalize

    if geneSymbol:
        _dt = ad.var.rename_axis("index").reset_index().set_index(geneSymbol)['index'].to_dict()
        ls_name = ls_gene
        ls_gene = [_dt[x] for x in ls_gene] 
    else:
        ls_name = ls_gene

    if logExpInput:
        logger.info('logExpInput is set to True. Please make sure that the input expression values is log2 transformed.')
        _df = np.exp2(ad[:, ls_gene].to_df(layer)) - 1
        df_meanExp = _df.groupby(ad.obs[groupby]).agg('mean')
        df_meanExp = np.log2(df_meanExp + 1)
    else:
        df_meanExp = ad[:, ls_gene].to_df(layer).groupby(ad.obs[groupby]).agg('mean')
        
    df_proExp = (ad[:, ls_gene].to_df(layer) > 0).groupby(ad.obs[groupby]).agg('mean').T.clip(*sizes)

    if minMaxScale:
        df_meanExp = (df_meanExp - df_meanExp.min(0)) / (df_meanExp.max(0) - df_meanExp.min(0))
    df_meanExp = df_meanExp.T


    h = ma.SizedHeatmap(
        df_proExp.values, color=df_meanExp.values, cmap=cmap,
        color_legend_kws=colorLegKws,
        size_legend_kws=sizeLegKws,
        width=width,
        height=height,
        sizes=dotsizes
    )
    h.add_left(mp.Labels(ls_name), pad=0.1)
    h.add_bottom(mp.Labels(df_proExp.columns), pad=0.1)
    h.add_legends()

    return h

class PlotAnndata(object):
    """
    The PlotAnndata class provides methods for visualizing single-cell RNA sequencing data using AnnData objects.

    Attributes
    ----------
    ad : AnnData object
        The AnnData object containing the single-cell RNA sequencing data.
    rawLayer : str
        The name of the layer in the AnnData object containing the raw gene expression data.

    Methods
    -------
    __init__(self, ad, rawLayer)
        Initializes a new instance of the PlotAnndata class.
    __repr__(self)
        Returns a string representation of the PlotAnndata object.
    getAdColors(self, label)
        Returns a dictionary mapping categorical observation values to colors.
    getPb(self, ls_group)
        Returns a new AnnData object containing the pseudobulk data for the specified groups.
    heatmapGeneExpInPb(self, ls_group, ls_leftAnno, dt_genes, layer='normalize_log', height=10, width=10, cmap='Reds', standardScale=None, showGeneCounts=False)
        Generates a heatmap of gene expression in a single-cell RNA sequencing dataset.
    embedding(self, embed, color=None, title=None, layer=None, groupby=None, wrap=4, size=2, cmap='Reds', vmin=0, vmax=None, ncol=1, figsize=(4,3), titleInFig=False, dt_theme={'ytick.left':False, 'ytick.labelleft':False, 'xtick.bottom':False, 'xtick.labelbottom':False, 'legend.markerscale': 3})
        Generates a scatter plot of data points based on a specified embedding.
    """
    
    def __init__(self, ad, rawLayer='raw'):
        self.ad = ad
        self.rawLayer = rawLayer
        # if 'pseudobulk_anndata' in self.ad.uns:
        #     pass
        # else:
        #     self.ad.uns['pseudobulk_anndata'] = {}
        # self.dtAd_pb = self.ad.uns['pseudobulk_anndata']

    def __repr__(self):
        contents = ['PlotAnndata:\n' + self.ad.__repr__()]
        contents.append('Pseudobulk AnnData:')
        for groups, ad_pb in self.dtAd_pb.items():
            contents.append(f"\t\t{groups}")
        return '\n'.join(contents)
    
    @property
    def dtAd_pb(self):
        if 'pseudobulk_anndata' not in self.ad.uns:
            self.ad.uns['pseudobulk_anndata'] = {}
        return self.ad.uns['pseudobulk_anndata']

    def getAdColors(self, label):
        ad = self.ad
        if f"{label}_colors" not in ad.uns:
            sc.pl._utils._set_default_colors_for_categorical_obs(ad, label)
        if len(ad.obs[label].cat.categories) != len(ad.uns[f"{label}_colors"]):
            sc.pl._utils._set_default_colors_for_categorical_obs(ad, label)
            
        return {
            x: y
            for x, y in zip(ad.obs[label].cat.categories, ad.uns[f"{label}_colors"])
        }

    def setAdColors(self, label, dt_color=None, hex=True):
        adata = self.ad
        adata._sanitize()
        if dt_color:
            if not hex:
                from matplotlib.colors import to_hex
                dt_color = {x: to_hex(y) for x, y in dt_color.items()}

            _dt = self.getAdColors(label)
            _dt.update(dt_color)
            dt_color = _dt
            adata.uns[f"{label}_colors"] = [
                dt_color[x] for x in adata.obs[label].cat.categories
            ]
        else:
            if f"{label}_colors" not in adata.uns:
                sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)


    def getPb(self, ls_group, force=False):
        ad = self.ad
        ad._sanitize()
        if isinstance(ls_group, str):
            ls_group = [ls_group]
        ls_group = sorted(ls_group)
        pbKey = ','.join(ls_group)
        pbKey = pbKey + ':' + self.rawLayer
        runPb = False
        if force:
            runPb = True
        if pbKey in self.dtAd_pb.keys():
            ad_pb = self.dtAd_pb[pbKey]
            for group in ls_group:
                ls_adGroupContent = ad.obs[group].cat.categories.tolist()
                ls_pbGroupContent = ad_pb.obs[group].cat.categories.tolist()
                if set(ls_adGroupContent) != set(ls_pbGroupContent):
                    logger.info(f"The group {group} in the pseudobulk AnnData object is different from the group in the original AnnData object.")
                    logger.info(f"The group in the pseudobulk AnnData object is {ls_pbGroupContent}")
                    logger.info(f"The group in the original AnnData object is {ls_adGroupContent}")
                    ls_overlap = list(set(ls_adGroupContent) & set(ls_pbGroupContent))
                    logger.info(f"The overlapping groups are {ls_overlap}")
                    if set(ls_adGroupContent) == set(ls_overlap):
                        logger.info(f"The pseudobulk AnnData object will be updated to include only the overlapping groups")
                        ad_pb = ad_pb[ad_pb.obs[group].isin(ls_overlap)]
                    else:
                        logger.warning(f"The pseudobulk AnnData object will be re-generated.")
                        runPb = True
                        break
            else:        
                self.dtAd_pb[pbKey] = ad_pb
        else:
            runPb = True

        if runPb:
            ad_pb = geneEnrichInfo._mergeData(ad, ls_group, layer=self.rawLayer)
            basic.initLayer(ad_pb, total=1e6, logbase=2)
            ad_pb.layers['normalize'] = np.exp2(ad_pb.layers['normalize_log']) - 1
            self.dtAd_pb[pbKey] = ad_pb

        return ad_pb

    def histogram(
            self, variable, groupby=None, wrap=4, bins=50, binrange=None, markLine:Optional[List[float]]=None, 
            addStat:Optional[Literal['mean', 'median']]=None, fc_additional:Optional[Callable]=None,
            dt_kwargsForHist={'common_norm':False, 'stat': 'percent'}
        ):
        '''The `plotHist` function is used to plot histograms of a variable in an AnnData object, with optional grouping and additional statistical measures.

        Parameters
        ----------
        variable
            The variable parameter is the name of the variable you want to plot the histogram for. It should be a column name in the AnnData object's obs dataframe.
        groupby
            The `groupby` parameter is used to specify a variable in the AnnData object that you want to group the data by. This can be useful if you want to create separate histograms for different groups within your data.
        wrap, optional
            The `wrap` parameter determines the number of columns in the facet grid when grouping the data by a variable. It specifies how many subplots should be displayed in each row before moving on to the next row.
        bins, optional
            The `bins` parameter specifies the number of bins to use for the histogram. It determines the width of each bin in the histogram.
        binrange
            The `binrange` parameter is used to specify the range of values for the bins in the histogram. It takes a tuple of two values, where the first value is the lower bound of the range and the second value is the upper bound of the range. For example, if you want the histogram
        markLine : Optional[List[float]]
            The `markLine` parameter is an optional list of float values that can be used to add vertical dashed lines to the histogram plot. Each value in the list represents the position of the line on the x-axis.
        addStat : Optional[Literal['mean', 'median']]
            The `addStat` parameter is an optional parameter that allows you to add additional statistics to the histogram plot. It accepts a list of strings, where each string represents a statistic to be added. The supported statistics are 'mean' and 'median'.
        fc_additional : Optional[Callable]
            The `fc_additional` parameter is an optional callable function that allows you to add additional plot elements or customize the plot further. It takes in a `Plot` object `p` as input and should return the modified `Plot` object. You can use this parameter to add any additional plot elements or
        dt_kwargsForHist
            The `dt_kwargsForHist` parameter is a dictionary that contains additional keyword arguments for the `Hist` plot. These arguments are used to customize the histogram plot. The available options are:

        Returns
        -------
            The function `plotHist` returns two values: `p` and `fig`. `p` is an instance of the `so.Plot` class, which represents the plot object, and `fig` is the matplotlib figure object that can be used to display or save the plot.

        '''
        ad = self.ad
        ad._sanitize()
        assert variable in ad.obs.columns, f"The variable {variable} is not in the AnnData object."
        if groupby is None:
            df = ad.obs[[variable]]
            # groupby = '_hist_temp'
            # df[groupby] = 'All'
        else:
            assert groupby in ad.obs.columns, f"The groupby variable {groupby} is not in the AnnData object."
            df = ad.obs[[variable, groupby]]
        df['_hist_temp'] = 'All'
        if binrange is None:
            pass
        else:
            df[variable] = df[variable].clip(*binrange)
        p = (
            so.Plot(df.reset_index())
            .add(
                so.Bars(color="#9DC3E7"),
                so.Hist(bins=bins, binrange=binrange, **dt_kwargsForHist),
                x=variable,
            )
        )
        if groupby is None:
            pass
        else:
            p = p.facet(col=groupby, wrap=wrap)
        if addStat is None:
            pass
        else:
            if isinstance(addStat, str):
                addStat = [addStat]
            for stat in addStat:
                p = p.add(
                    Axvline(linestyle="--"),
                    so.Agg(func=stat),
                    orient="y",
                    x=variable,
                    group='_hist_temp'
                )
        if markLine is None:
            pass
        else:
            if isinstance(markLine, (float, int)):
                markLine = [markLine]
            for line in markLine:
                if groupby is None:
                    p = p.add(Axvline(linestyle="--",), x=[line], data={})
                else:
                    p = p.add(Axvline(linestyle="--",), x=[line] * ad.obs[groupby].nunique(), data={}, col=ad.obs[groupby].cat.categories.tolist())
        if fc_additional is None:
            pass
        else:
            p = fc_additional(p)

        fig = p.plot()._figure
        for ax in fig.axes:
            # ax.xaxis.label.set_visible(True)
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)

        return p, fig

    def barplot(self, x, gene, groupby=None, layer:Literal['normalize_log', 'normalize']='normalize_log', figsize=(6,4), fc_additional = lambda _: _):
        self.ad._sanitize()
        if isinstance(gene, str):
            gene = [gene]
        ad_bulk = self.getPb([x, groupby]) if groupby else self.getPb([x])
        if groupby:
            df_exp = pd.concat(
                [
                    ad_bulk.obs[[x, groupby]]
                    .join(ad_bulk[:, _].to_df(layer))
                    .rename(columns={_: "Exp"})
                    .assign(gene=_)
                    for _ in gene
                ],
                ignore_index=True,
            )
        else:
            df_exp = pd.concat(
                [
                    ad_bulk.obs[[x]]
                    .join(ad_bulk[:, _].to_df(layer))
                    .rename(columns={_: "Exp"})
                    .assign(gene=_)
                    for _ in gene
                ],
                ignore_index=True,
            )
        p = (
            so.Plot(df_exp, x=x, y="Exp", color=groupby)
            .facet(col="gene", wrap=2)
            .share(y=False)
            .add(so.Bar(), so.Dodge())
            .layout(size=figsize)
        )

        if groupby:
            dt_colors = self.getAdColors(groupby)
            p = p.scale(color=dt_colors)
        p = fc_additional(p)
        fig = p.plot()._figure
        for ax in fig.axes:
            ax.set_title(ax.get_title(), fontstyle="italic")
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
        return p, fig

    def heatmapGeneExpInPb(self, ls_group, ls_leftAnno, dt_genes, layer='normalize_log', height=10, width=10, cmap='Reds', standardScale=None, showGeneCounts=False):
        raise NameError('Please use heatmapGeneExp as alternative')

    def heatmapGeneExp(
            self, ls_group: Union[None, List[str]], ls_leftAnno:List[str], dt_genes:Dict[str, List[str]], layer='normalize_log', height=10, width=10, cmap='Reds', standardScale=None, showGeneCounts=False,
            addGeneName:bool=False, addGeneCatName:bool=True, geneSpace=0.005, cellSpace=0.003, needExp=False, useObsm=False, cellSplitBy=None, **dt_forHeatmap):
        '''The `heatmapGeneExp` function generates a heatmap of gene expression in a single-cell RNA sequencing dataset, with options for customization such as color mapping, scaling, and showing gene counts.

        Parameters
        ----------
        ls_group
            A list of groups to include in the heatmap. These groups will be used to subset the data and create separate heatmaps for each group.
        ls_leftAnno
            A list of column names in the dataframe `ad_pb` that will be used to group the data for the left annotation labels in the heatmap.
        dt_genes
            The `dt_genes` parameter is a dictionary where the keys represent gene categories and the values are lists of genes belonging to each category.
        layer, optional
            The 'layer' parameter specifies the layer of gene expression data to use for creating the heatmap. The default value is 'normalize_log'.
        height, optional
            The height parameter determines the height of the heatmap in inches.
        width, optional
            The width parameter determines the width of the heatmap in inches.
        cmap, optional
            cmap stands for colormap. It is used to specify the color map for the heatmap. The default value is 'Reds', which is a sequential colormap ranging from light to dark red.
        standardScale
            The `standardScale` parameter is used to specify whether or not to standardize the data before creating the heatmap. If `standardScale` is set to `None`, the data will not be standardized. If `standardScale` is set to a value other than `None`, the data will be
        showGeneCounts, optional
            A boolean parameter that determines whether to show the gene counts in the heatmap. If set to True, the gene counts will be displayed next to the gene labels in the heatmap. If set to False, only the gene labels will be displayed.

        Returns
        -------
            a heatmap object.

        '''
        ad_pb = self.ad if ls_group is None else self.getPb(ls_group)
        ad_pb = ad_pb[ad_pb.obs.sort_values(ls_leftAnno).index]
        # ls_genes = sum(list(dt_genes.values()), [])
        # ls_geneCate = [x for x,y in dt_genes.items() for z in y]
        if useObsm:
            dt_genes = {x: [z for z in y if z in ad_pb.obsm[useObsm].columns] for x, y in dt_genes.items()}
            ls_genes = sum(list(dt_genes.values()), [])
            ls_geneCate = [x for x,y in dt_genes.items() for z in y]
            df_exp = ad_pb.obsm[useObsm][ls_genes]
        else:
            dt_genes = {x: [z for z in y if z in ad_pb.var.index] for x, y in dt_genes.items()}
            ls_genes = sum(list(dt_genes.values()), [])
            ls_geneCate = [x for x,y in dt_genes.items() for z in y]
            df_exp = ad_pb[:, ls_genes].to_df(layer)
        if standardScale is None:
            df_heatmap = df_exp
        else: 
            df_heatmap = df_exp.apply(lambda _: (_-_.min())/(_.max()-_.min()), axis=standardScale).fillna(0)

        h = ma.Heatmap(df_heatmap.values, cmap=cmap, height=height, width=width, cluster_data=df_heatmap.values, **dt_forHeatmap)

        pad = 0.1
        for group in ls_leftAnno[::-1]:
            ls_obsGroup = ad_pb.obs[group]
            h.add_left(
                mp.Colors(ls_obsGroup, palette=self.getAdColors(group), label=group), size=0.15, pad=pad, name=group
            )
            pad=0

        h.vsplit(labels=ls_geneCate, order=list(dt_genes.keys()), spacing=geneSpace)
        if showGeneCounts:
            ls_geneLabel = dt_genes.keys() >> F(map, lambda _: f"{_}\n({len(dt_genes[_])})") >> F(list)
        else:
            ls_geneLabel = dt_genes.keys() >> F(list)
        if addGeneCatName:
            h.add_top(
                mp.Chunk(ls_geneLabel, props={'rotation': 90})
            )
        if addGeneName:
            h.add_bottom(
                mp.Labels(ls_genes)
            )
        if cellSplitBy:
            h.hsplit(labels=ad_pb.obs[cellSplitBy], order=ad_pb.obs[cellSplitBy].cat.categories.tolist(), spacing=cellSpace)
        h.add_legends()
        if needExp:
            return h, df_heatmap
        else:
            return h
    
    def _embedding(self, embed='umap', color=None, title=None, layer=None, groupby=None, wrap=4, size=2, 
                  cmap='Reds', vmin=0, vmax=None, ls_color=None, ls_group=None, addBackground=False, share=True, axisLabel=None, useObs=None, titleLocY = 0.9,
                  figsize=(8,6), legendCol=1, legendInFig=False, fc_legendInFig=lambda _: _.split(':')[0], needLegend=True, subsample=None, showTickLabels=False, italicTitle=None,
                  dt_theme={'ytick.left':False, 'ytick.labelleft':False, 'xtick.bottom':False, 'xtick.labelbottom':False, 'legend.markerscale': 3}):
        '''The `embedding` function in Python generates a scatter plot of data points based on a specified embedding, with the option to color the points based on a specified variable, and additional customization options.

        Parameters
        ----------
        embed
            The `embed` parameter is a string that specifies the embedding to use for plotting. It should be in the format "X_embedding_name", where "embedding_name" is the name of the embedding stored in the `obsm` attribute of the `AnnData` object.
        color
            The "color" parameter is used to specify the variable that will be used to color the points in the embedding plot. It can be either a column name in the `ad.obs` dataframe or a layer name in the `ad` object.
        title
            The title parameter is used to specify the title of the plot. If it is not provided, the value of the color parameter will be used as the title.
        layer
            The "layer" parameter is used to specify the layer of the AnnData object that should be used for plotting. It is an optional parameter and if not provided, the default layer will be used.
        groupby
            The `groupby` parameter is used to specify a column in the AnnData object's `obs` attribute that will be used to group the data points in the plot. This can be useful for visualizing different groups or clusters in the data.
        wrap, optional
            The `wrap` parameter determines the number of subplots to display per row when using the `groupby` parameter for faceting the plots. It specifies the maximum number of subplots to display in a single row before wrapping to the next row.
        size, optional
            The `size` parameter determines the size of the points in the scatter plot. It is used to control the size of the dots in the plot.
        cmap, optional
            The `cmap` parameter in the `embedding` function is used to specify the colormap for the color mapping in the plot. It determines the range of colors that will be used to represent different values of the `color` variable. The default value is `'Reds'`, which is a colormap
        vmin, optional
            The parameter `vmin` is used to set the minimum value for the color scale in the plot. It determines the lower bound of the color range.
        vmax
            The parameter `vmax` is used to set the maximum value for the color scale in the plot. If not specified, it defaults to the maximum value in the `color` column of the data.
        figsize
            The `figsize` parameter is used to specify the size of the figure (plot) in inches. It takes a tuple of two values, where the first value represents the width and the second value represents the height of the figure. For example, `figsize=(4, 3)` will
        titleInFig, optional
            The `titleInFig` parameter is a boolean flag that determines whether the title of the plot should be displayed within the figure itself. If `titleInFig` is set to `True`, the title will be displayed as a text annotation within the plot. If `titleInFig` is set
        dt_theme
            The `dt_theme` parameter is a dictionary that allows you to customize the appearance of the plot. It contains key-value pairs where the key is a string representing a specific plot element (e.g., 'ytick.left' for left y-axis ticks) and the value is a boolean indicating whether to

        Returns
        -------
            two values: `g` and `fig`. `g` is an instance of the `so.Plot` class, which represents the plot object, and `fig` is the matplotlib figure object.

        '''
        ad = self.ad
        ad._sanitize()
        embed = 'X_' + embed.split('X_', 1)[-1]
        embedLabel = embed.split('X_', 1)[-1].upper()
        if axisLabel is None:
            axisLabel = embedLabel
        assert embed in self.ad.obsm.keys(), f"Embedding {embed} not found in AnnData"
        if groupby is None:
            pass
        else:
            if ls_group is None:
                ls_group = ad.obs[groupby].cat.categories.tolist()
            else:
                assert set(ls_group) <= set(ad.obs[groupby].cat.categories), f"Groups {set(ls_group) - set(ad.obs[groupby].cat.categories)} not found in AnnData"
        if title is None:
            title = color

        if useObs is None:
            if color in ad.obs.columns:
                useObs = True
            else:
                useObs = False
        if useObs:
            italicTitle = False if italicTitle is None else italicTitle
            if ad.obs[color].dtype.name == 'category':
                dt_colors = self.getAdColors(color)
                dt_colors['None'] = 'silver'
                if ls_color is None:
                    ls_color = ad.obs[color].cat.categories
                # ls_group = [x for x in ls_group if ~pd.isna(x)]
                logger.debug(f"ls_color: {ls_color}")
                df = ad.obs[[color]].copy()
                legend = False
            else:
                df = ad.obs[[color]].copy()
                legend = False
                useObs = False
        else:
            italicTitle = True if italicTitle is None else italicTitle
            df = ad[:, color].to_df(layer)
            legend = False
            
        # print(italicTitle)
        df['x'] = ad.obsm[embed][:,0]
        df['y'] = ad.obsm[embed][:,1]
        
        if groupby:
            df['groupby'] = ad.obs[groupby]
        else:
            df['groupby'] = 'All'

        if subsample:
            df = df.sample(subsample, random_state=39)

        if useObs:
            df[color].cat.add_categories('None', inplace=True)
            df[color] = df[color].cat.reorder_categories(['None',*df[color].cat.categories[:-1]])
            df.loc[lambda _: ~_[color].isin(ls_color), color] = 'None'
            df_clusterLabelLoc = df.dropna(subset=[color]).groupby([color, 'groupby'])[['x', 'y']].agg('median').reset_index()
            df_clusterLabelLoc = df_clusterLabelLoc.query(f"`{color}` != 'None'")

        if useObs & (not groupby is None) & addBackground:
            # set cells from other groups to grey color
            lsDf = []
            for group in ls_group:
                _df = df.copy()
                _df.loc[lambda _: _['groupby'] != group, color]  = 'None'
                _df = _df.assign(groupby=group)
                lsDf.append(_df)
            df = pd.concat(lsDf).reset_index(drop=True)
            # print(ls_group)
            df['groupby'] = df['groupby'].astype('category').cat.set_categories(ls_group)

        df = df.sort_values(color)

        g = (
            so.Plot(df, x='x', y='y')
            .add(so.Dots(fillalpha=1, pointsize=size), color=color, legend=legend)
            .share(x=share, y=share)
            .theme(dt_theme)
            .label(x=f'{axisLabel} 1', y=f'{axisLabel} 2', color='')
            # .layout(size=figsize)
        )

        if useObs:
            if legendInFig:
                df_clusterLabelLoc[color] = df_clusterLabelLoc[color].map(fc_legendInFig)
                g = g.add(
                    so.Text(
                        halign='center', valign='center', artist_kws={'weight':'bold'}
                    ), 
                    data=df_clusterLabelLoc, x='x', y='y', text=color, 
                    col='groupby' if groupby else None
                )
            g = g.scale(color=dt_colors)
        else:
            if vmax is None:
                vmax = df[color].max()
            g = g.scale(color=so.Continuous(cmap, norm=(vmin, vmax)))

        if groupby:
            g = g.facet(col='groupby', wrap=wrap, order=ls_group)
        fig = plt.figure(figsize=figsize)
        p = g.on(fig).plot()
        # p._repr_png_()
        # fig = p._figure 

        if needLegend:
            if useObs:
                dt_colors = dt_colors.copy()
                dt_colors.pop('None')

                fig.transAxes = fig.transFigure
                fig.legend_ = 1

                legendkit.cat_legend(ax=fig, colors=list(dt_colors.values()), labels=list(dt_colors.keys()), handle='circle', loc='out right center', deviation=-0.07, ncol=legendCol)

                # fig.legend(handles=[legendkit.handles.CircleItem(color=x) for x in dt_colors.values()], labels=dt_colors.keys(), loc='center left', bbox_to_anchor=(1, 0.5), ncol=legendCol)
            else:
                from matplotlib.colors import Normalize
                from legendkit import colorart
                fig.transAxes = fig.transFigure
                colorart(cmap=cmap, norm=Normalize(vmin, vmax), ax=fig, loc='out right center', deviation=-0.07, height=figsize[1] * 3)
        
        for ax in fig.axes:
            if showTickLabels:
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.yaxis.set_tick_params(labelleft=True)
            else:
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.yaxis.set_tick_params(labelleft=False)
                
            ax.xaxis.label.set_visible(True)
        # if italicTitle:
        #         ax.set_title(ax.get_title(), fontstyle='italic')

        fig.suptitle(title, y=titleLocY, va='baseline', fontstyle='italic' if italicTitle else 'normal')
        plt.close()
        return g, fig

    def embedding(self, embed='umap', color=None, title=None, layer=None, groupby=None, wrap=4, size=2, 
                  cmap='Reds', vmin=0, vmax=None, ls_color=None, ls_group=None, addBackground=False, share=True, axisLabel=None, useObs=None, titleLocY = 0.9,
                  figsize=(8,6), legendCol=1, legendInFig=False, fc_legendInFig=lambda _: _.split(':')[0], needLegend=True, subsample=None, showTickLabels=False, italicTitle=None,
                  dt_theme={'ytick.left':False, 'ytick.labelleft':False, 'xtick.bottom':False, 'xtick.labelbottom':False, 'legend.markerscale': 3}):
        '''The `embedding` function in Python generates a scatter plot of data points based on a specified embedding, with the option to color the points based on a specified variable, and additional customization options.

        Parameters
        ----------
        embed
            The `embed` parameter is a string that specifies the embedding to use for plotting. It should be in the format "X_embedding_name", where "embedding_name" is the name of the embedding stored in the `obsm` attribute of the `AnnData` object.
        color
            The "color" parameter is used to specify the variable that will be used to color the points in the embedding plot. It can be either a column name in the `ad.obs` dataframe or a layer name in the `ad` object.
            if color is List[str], a figure with subplots will be generated.
        title
            The title parameter is used to specify the title of the plot. If it is not provided, the value of the color parameter will be used as the title.
        layer
            The "layer" parameter is used to specify the layer of the AnnData object that should be used for plotting. It is an optional parameter and if not provided, the default layer will be used.
        groupby
            The `groupby` parameter is used to specify a column in the AnnData object's `obs` attribute that will be used to group the data points in the plot. This can be useful for visualizing different groups or clusters in the data.
        wrap, optional
            The `wrap` parameter determines the number of subplots to display per row when using the `groupby` parameter for faceting the plots. It specifies the maximum number of subplots to display in a single row before wrapping to the next row.
        size, optional
            The `size` parameter determines the size of the points in the scatter plot. It is used to control the size of the dots in the plot.
        cmap, optional
            The `cmap` parameter in the `embedding` function is used to specify the colormap for the color mapping in the plot. It determines the range of colors that will be used to represent different values of the `color` variable. The default value is `'Reds'`, which is a colormap
        vmin, optional
            The parameter `vmin` is used to set the minimum value for the color scale in the plot. It determines the lower bound of the color range.
        vmax
            The parameter `vmax` is used to set the maximum value for the color scale in the plot. If not specified, it defaults to the maximum value in the `color` column of the data.
        figsize
            The `figsize` parameter is used to specify the size of the figure (plot) in inches. It takes a tuple of two values, where the first value represents the width and the second value represents the height of the figure. For example, `figsize=(4, 3)` will
        titleInFig, optional
            The `titleInFig` parameter is a boolean flag that determines whether the title of the plot should be displayed within the figure itself. If `titleInFig` is set to `True`, the title will be displayed as a text annotation within the plot. If `titleInFig` is set
        dt_theme
            The `dt_theme` parameter is a dictionary that allows you to customize the appearance of the plot. It contains key-value pairs where the key is a string representing a specific plot element (e.g., 'ytick.left' for left y-axis ticks) and the value is a boolean indicating whether to

        Returns
        -------
        if only one color is provided, returns two values: `g` and `fig`. `g` is an instance of the `so.Plot` class, which represents the plot object, and `fig` is the matplotlib figure object.
        if multiple colors are provided, returns a FigConcate.

        '''
        if isinstance(color, str):
            return self._embedding(
                embed=embed, color=color, title=title, layer=layer, groupby=groupby, wrap=wrap, size=size, cmap=cmap, vmin=vmin, vmax=vmax, ls_color=ls_color, ls_group=ls_group, addBackground=addBackground, share=share, italicTitle=italicTitle, axisLabel=axisLabel, useObs=useObs, titleLocY=titleLocY, figsize=figsize, legendCol=legendCol, legendInFig=legendInFig, fc_legendInFig=fc_legendInFig, needLegend=needLegend, subsample=subsample, showTickLabels=showTickLabels, dt_theme=dt_theme)
        else:
            from ..otherTools import FigConcate, FigConcateWrap
            # assert groupby is None, "groupby is not supported when multiple colors are provided"
            if groupby is None:
                figwrap = FigConcateWrap()
                for _color in color:
                    _, fig = self._embedding(
                        embed=embed, color=_color, title=title, layer=layer, groupby=groupby, size=size, cmap=cmap, vmin=vmin, vmax=vmax, ls_color=ls_color, ls_group=ls_group, addBackground=addBackground, share=share, italicTitle=italicTitle, axisLabel=axisLabel, useObs=useObs, titleLocY=titleLocY, figsize=figsize, legendCol=legendCol, legendInFig=legendInFig, fc_legendInFig=fc_legendInFig, needLegend=needLegend, subsample=subsample, showTickLabels=showTickLabels, dt_theme=dt_theme)
                    figwrap.addFig(fig >> F(FigConcate))
                return figwrap.wrapAndGenerate(wrap)
            else:
                figwrap = FigConcateWrap()
                logger.warning("Both groupby and multiple colors are provided. `Wrap` will be ignored.")
                for _color in color:
                    _, fig = self._embedding(
                        embed=embed, color=_color, title=title, layer=layer, groupby=groupby, wrap=None, size=size, cmap=cmap, vmin=vmin, vmax=vmax, ls_color=ls_color, ls_group=ls_group, addBackground=addBackground, share=share, italicTitle=italicTitle, axisLabel=axisLabel, useObs=useObs, titleLocY=titleLocY, figsize=figsize, legendCol=legendCol, legendInFig=legendInFig, fc_legendInFig=fc_legendInFig, needLegend=needLegend, subsample=subsample, showTickLabels=showTickLabels, dt_theme=dt_theme)
                    figwrap.addFig(fig >> F(FigConcate))
                return figwrap.wrapAndGenerate(wrap=1)
            

    def catplot(self, x, y, kind, hue=None, **dt_args):
        ad = self.ad
        if hue is None:
            hue = x
            df = ad.obs[[x, y]].copy()
        else:
            df = ad.obs[[x, y, hue]].copy()
        
        dt_colors = self.getAdColors(hue)

        p = sns.catplot(
            df,
            x=x,
            y=y,
            kind=kind,
            hue=hue,
            palette=dt_colors,
            **dt_args
        )
        return p

    def clustree(self, clusterKey, subsample=None, ls_res=None, **dt_args):
        from rpy2.robjects.packages import importr
        from ..rTools import py2r

        clustree = importr('clustree')
        if ls_res is None:
            ls_res = self.ad.obsm[clusterKey].columns
        ls_res = ls_res >> F(map, str) >> F(list)
        for x in ls_res:
            assert str(x) in self.ad.obsm[clusterKey].columns, f"{x} not in {clusterKey}"
        df_cluster = self.ad.obsm[clusterKey][ls_res].rename(columns=lambda _: clusterKey + str(_))
        dfr_cluster = df_cluster.sample(n=subsample, random_state=0) >> F(py2r)
        p = clustree.clustree(dfr_cluster, prefix=clusterKey, **dt_args)
        return p


    def clusterPercentage(self, x, y, addCounts=True, figsize=(5, 4), fc_after = lambda _: _, dt_theme={}):
        ad = self.ad
        dt_colors = self.getAdColors(y)

        sr_counts = ad.obs.value_counts([x, y]).unstack().fillna(0).astype(int).sum(1)
        df_ratio = ad.obs.value_counts([x, y]).unstack().fillna(0).astype(int).apply(lambda x: x/x.sum(), 1) * 100
        df_ratio = df_ratio.stack().rename('ratio').reset_index()

        g = (
            so.Plot(df_ratio, x, y='ratio', color=y)
            .add(so.Bar(alpha=1), so.Stack())
            .label(y='Percentage (%)')
            .scale(color=dt_colors)
            .layout(size=figsize)
            .theme(dt_theme)
        )
        g = fc_after(g)
        if addCounts:
            g = g.add(so.Text(artist_kws={'rotation': 90}, valign='baseline'), data={}, x=sr_counts.keys(), y=[100] * len(sr_counts), text=[f"N = {x}" for x in sr_counts], color=None)

        p = g.plot()
        p._repr_png_()
        fig = p._figure
        ax = fig.axes[0]
        for text in ax.texts:
            text.set(clip_on=False)
        return g, fig
    
    def clusterSankey(self, source, target, recoverDefaultRc=True, otherLabel='Others',**dt_args):
        dt_rc = dict(plt.rcParams)
        
        import pysankey
        ad = self.ad
        ad._sanitize()

        df_anno = ad.obs[[source, target]].copy()
        df_anno[source] = df_anno[source].cat.add_categories(otherLabel)
        df_anno[target] = df_anno[target].cat.add_categories(otherLabel)
        df_anno = df_anno.fillna(otherLabel)
        # print(df_anno[source].cat.categories)
        df_anno[source] = df_anno[source].map(lambda _: _ + ' ')
        df_anno[target] = df_anno[target].map(lambda _: ' ' + _)

        _dt = self.getAdColors(source)
        _dt = {f"{x} ":y for x,y in _dt.items()}
        dt_colors = self.getAdColors(target)
        dt_colors = {f" {x}":y for x,y in dt_colors.items()}
        dt_colors = dict(**dt_colors, **_dt)
        dt_colors[f" {otherLabel}"] = '#696969'
        dt_colors[f"{otherLabel} "] = '#696969'
        # return df_anno

        ax = pysankey.sankey(df_anno[source], df_anno[target], colorDict=dt_colors, **dt_args)
        if recoverDefaultRc:
            plt.rcParams.update(dt_rc)
        return ax
    
    def clusterCompareHeatmap(self, source, target, scaleAxis=0, figsize=(8,8), dendrogram=True):
        ad = self.ad
        df_counts = ad.obs[[source, target]].value_counts().unstack().fillna(0).astype(int)
        if scaleAxis is None:
            df_scale = df_counts
        else:
            df_scale = df_counts.apply(lambda _: _ / _.sum(), axis=scaleAxis)
        
        h = ma.Heatmap(df_scale.values, cmap='Reds', width=figsize[0], height=figsize[1])

        h.add_left(
            mp.Labels(df_scale.index)
        )
        h.add_bottom(
            mp.Labels(df_scale.columns)
        )
        if dendrogram:
            h.add_dendrogram('left', method='ward')
            h.add_dendrogram('bottom', method='ward')

        h.add_left(mp.Title(source))
        h.add_bottom(mp.Title(target))

        h.add_legends()
        return h

    def geneCompare(self, g1, g2, min_g1=0, max_g1=None, min_g2=0, max_g2=None, cellSize=2, layer='normalize_log', figsize=(10, 6)):
        ad = self.ad
        _ad = ad[:, [g1, g2]]
        df_for2dScatter = _ad.to_df(layer)
        df_for2dScatter['total'] = df_for2dScatter[g1] + df_for2dScatter[g2]
        if max_g1 is None:
            max_g1 = df_for2dScatter[g1].max()
        if max_g2 is None:
            max_g2 = df_for2dScatter[g2].max()
        
        df_for2dScatter['x'] = ad.obsm['X_umap'][:, 0]
        df_for2dScatter['y'] = ad.obsm['X_umap'][:, 1]
        df_for2dScatter = df_for2dScatter.sort_values('total')

        lowestRgb = 0.9
        def getRgb(x, y, min_g1, max_g1, min_g2, max_g2):
            b = lowestRgb - (x - min_g1) / (max_g1 - min_g1) * lowestRgb
            r = lowestRgb - (y - min_g2) / (max_g2 - min_g2) * lowestRgb
            g = lowestRgb - (x - min_g1) / (max_g1 - min_g1) * lowestRgb * 0.5 - (y - min_g2) / (max_g2 - min_g2) * lowestRgb * 0.5
            return r,g,b
        ufunc_getRgb = lambda x,y, min_g1, max_g1, min_g2, max_g2:np.clip(np.dstack(np.frompyfunc(getRgb, 6, 3)(x, y, min_g1, max_g1, min_g2, max_g2)).astype(np.float32), 0, 1)

        c1 = np.clip(df_for2dScatter[g1], min_g1,max_g1)
        c2 = np.clip(df_for2dScatter[g2], min_g2,max_g2)
        fig, ax1 = plt.subplots(figsize=figsize)
        plt.sca(ax1)

        plt.scatter(df_for2dScatter['x'], df_for2dScatter['y'], c=ufunc_getRgb(c1.values, c2.values, min_g1, max_g1, min_g2, max_g2)[0], s=cellSize)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2 = plt.axes([0.95, 0.4, 0.15, 0.2])
        x,y = np.meshgrid(
            np.linspace(min_g1,max_g1,100),
            np.linspace(min_g2,max_g2,100),
        )
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.imshow(
            ufunc_getRgb(x, y, min_g1, max_g1, min_g2, max_g2),
            aspect = 'auto',
            origin = 'lower',
        )
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set(xticks=[0, 100], xticklabels=[min_g1, max_g1], yticks=[0, 100], yticklabels=[min_g2, max_g2], xlabel=g1, ylabel=g2)
        fig.suptitle(f"{g1} vs {g2}", fontsize=12)
        return fig