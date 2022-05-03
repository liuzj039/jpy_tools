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
from cool import F
import collections
from xarray import corr
import matplotlib as mpl
from more_itertools import chunked
import patchworklib as pw
import scipy.sparse as ss
from joblib import Parallel, delayed

def show_figure(fig):

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    fig.show()
    plt.show()


pw.show = show_figure


from . import basic


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
    format='png',
    dpi = 'figure',
    fileNameIsTitle = False,
    show=True,
):
    if isinstance(ls_gene, str):
        ls_gene = [ls_gene]
    if not ls_title:
        ls_title = ls_gene
    if isinstance(ls_title, str):
        ls_title = [ls_title]
    if len(ls_gene) <= 1:
        disableProgressBar = True
    if groups[1] is None:
        groups[1] = ad.obs[groups[0]].unique().to_list()
    dt_adObs = ad.obs.groupby(groups[0]).apply(lambda df: df.index.to_list()).to_dict()
    dt_adObs = {x: dt_adObs[x] for x in groups[1]}
    dt_ad = {x: ad[y] for x, y in dt_adObs.items()}
    if needAll:
        dt_ad = dict(All=ad, **dt_ad)
    if not size:
        size = 12e4 / len(ad)

    vmin = vmin if vmin else 0
    vmaxSpecify = vmax
    for gene, title in tqdm(zip(ls_gene, ls_title), total=len(ls_gene), disable=disableProgressBar):
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
        ls_ax = chunked(ls_ax, ncols) | F(list)

        _bc = pw.param["margin"]
        pw.param["margin"] = 0.3
        if len(ls_ax) == 1:
            axs = pw.stack(ls_ax[0])
        else:
            axs = pw.stack([pw.stack(x) for x in ls_ax[:-1]], operator="/")
            ls_name = list(axs.bricks_dict.keys())
            for i, ax in enumerate(ls_ax[-1]):
                axs = axs[ls_name[i]] / ax

        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        if horizontal:
            ax_cb = pw.Brick(figsize=(1, 0.01))
            mpl.colorbar.ColorbarBase(
                ax_cb, cmap=cmap, norm=norm, orientation="horizontal"
            )

            pw.param["margin"] = 0.1
            axs = axs / ax_cb
            pw.param["margin"] = _bc
        else:
            ax_cb = pw.Brick(figsize=(0.025, 1))
            mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)

            pw.param["margin"] = 0.1
            axs = axs | ax_cb
            pw.param["margin"] = _bc

        axs.case.set_title(title, pad=10, size=16)
        if dir_result:
            if fileNameIsTitle:
                fileName = title.replace('\n', '_').replace('/', '_')
            else:
                fileName = gene
            axs.savefig(f"{dir_result}/{fileName}.{format}", dpi=dpi)
        elif show:
            pw.show(axs.savefig())
        else:
            pass
    if len(ls_gene) == 1:
        return axs

def saveUmapMultiBatch(ad, threads, batchSize, ls_gene, ls_title, layer, **dt_kwargs):
    from more_itertools import chunked
    def _iterAd(ad, batchSize, ls_gene, ls_title, layer):
        _ad = ad[:, ls_gene]
        ad = sc.AnnData(_ad.X, obs=_ad.obs, var=_ad.var, obsm = {'X_umap': ad.obsm['X_umap']}, layers={layer: _ad.layers[layer]})
        for ls_chunkGene, ls_chunkTitle in zip(chunked(ls_gene, batchSize), chunked(ls_title, batchSize)):
            yield ad[:, ls_chunkGene].copy(), ls_chunkGene, ls_chunkTitle

    if not ls_title:
        ls_title = ls_gene
    ls_batchGene = []
    ls_batchTitle = []
    ls_batchAd = []
    for ad, ls_chunkGene, ls_chunkTitle in _iterAd(ad, batchSize, ls_gene, ls_title, layer):
        ls_batchGene.append(ls_chunkGene)
        ls_batchTitle.append(ls_chunkTitle)
        ls_batchAd.append(ad)
    Parallel(n_jobs=threads)(delayed(umapMultiBatch)(_ad, ls_gene=ls_gene, ls_title=ls_title, layer=layer, **dt_kwargs) for _ad, ls_gene, ls_title in zip(ls_batchAd, ls_batchGene, ls_batchTitle))
    

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
            ax=ax,
        )
        plt.sca(ax)
        legendHandleLs.append(
            plt.Rectangle((0, 0), 1, 1, fc=labelColor[singleLabel], edgecolor="none")
        )
        legendLabelLs.append(singleLabel)
    legendHandleLs, legendLabelLs = legendHandleLs[::-1], legendLabelLs[::-1]
    plt.legend(legendHandleLs, legendLabelLs, frameon=False, **dt_kwargsForLegend)
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
    cbarPos=(0.72, 0.15, 0.01, 0.18),
    sort=True,
    dt_geneColor: Optional[Mapping[str, str]] = None,
    add_gene_name: bool = True,
    col_label: bool = False,
    legendAlign: Literal["h", "v"] = "h",
    addSplitLine=True,
    row_cluster=False,
    col_cluster=False,
    addObsLegend=True,
    forceShowModuleColor=False,
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
    if (len(dt_gene) == 1) & (not forceShowModuleColor):
        df_geneModuleChangeColor = None
    axs = sns.clustermap(
        df_mtx,
        cmap="Reds",
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

    if not ((len(dt_gene) == 1) & (not forceShowModuleColor)):
        plt.sca(axs.ax_col_colors)
        plt.xticks([])
        plt.yticks([])
    if addObsLegend:
        legendPox = [1.05, 1]
        for i, (anno, space) in enumerate(zip(obsAnno, space_obsAnnoLegend)):
            dt_color = basic.getadataColor(ad, anno)
            addColorLegendToAx(
                axs.ax_heatmap,
                anno,
                dt_color,
                1,
                bbox_to_anchor=legendPox,
                frameon=False,
            )
            if legendAlign == "h":
                legendPox = [1.05 + space, 1]
            elif legendAlign == "v":
                legendPox = [1.05, 1 - space]

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
    dt_labelOptions: Dict[str, str] = {"font_size":12, 'bbox' : {"ec": "k", "fc": "white", "alpha": 0.7}},
    ax = None
):
    '''This function plots a gene module using networkx
    
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
    
    '''
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
    G = nx.from_pandas_edgelist(
        df_adjacency.query("connectivity > @cutoff"), edge_attr="connectivity"
    )
    pos = nx.drawing.layout.kamada_kawai_layout(G)
    nx.draw(G, pos, ax=ax, **dt_baseOptions)
    if not ls_hubGenes is None:
        ls_hubGenes = [x for x in ls_hubGenes if x in G.nodes()]
        nx.draw(G, pos, nodelist=ls_hubGenes, ax=ax, **dt_hubOptions)
    if not dt_needLabelNodes is None:
        dt_needLabelNodes = {x: dt_needLabelNodes[x] for x in dt_needLabelNodes.keys() if x in G.nodes()}
        nx.draw_networkx_labels(G, pos, dt_needLabelNodes,ax = ax, **dt_labelOptions)
    ax = plt.gca()
    return ax


