from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from loguru import logger
from io import StringIO
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
import stlearn as st
import scipy.sparse as ss
from cool import F
from ..otherTools import setSeed

def loadBGI(path_gem, binSize) -> sc.AnnData:
    df_gem = pd.read_table(path_gem, comment="#")
    df_gem = df_gem.pipe(
        lambda df: df.assign(
            xBin=((df["x"] // binSize) * binSize + binSize / 2).astype(int),
            yBin=((df["y"] // binSize) * binSize + binSize / 2).astype(int),
        )
    ).assign(nameBin=lambda df: df["xBin"].astype(str) + "_" + df["yBin"].astype(str))
    df_mtxLong = (
        df_gem.groupby(["nameBin", "geneID"])["MIDCount"].agg("sum").reset_index()
    )
    ar_mtx = np.zeros(
        (
            df_mtxLong["nameBin"].unique().shape[0],
            df_mtxLong["geneID"].unique().shape[0],
        )
    )
    ls_obs = df_mtxLong["nameBin"].unique().tolist()
    _dt = {x: i for i, x in enumerate(ls_obs)}

    df_mtxLong["nameBin_index"] = df_mtxLong["nameBin"].map(_dt)

    ls_var = df_mtxLong["geneID"].unique().tolist()
    _dt = {x: i for i, x in enumerate(ls_var)}

    df_mtxLong["geneID_index"] = df_mtxLong["geneID"].map(_dt)
    for tp_line in tqdm(df_mtxLong.itertuples(), total=len(df_mtxLong)):
        ar_mtx[tp_line.nameBin_index, tp_line.geneID_index] += tp_line.MIDCount
    ad = sc.AnnData(
        ar_mtx, obs=pd.DataFrame(index=ls_obs), var=pd.DataFrame(index=ls_var)
    )
    df_spatial = pd.DataFrame(
        [ad.obs.index.str.split("_").str[0], ad.obs.index.str.split("_").str[1]],
        columns=ad.obs.index,
        index=["imagecol", "imagerow"],
    ).T.astype(int)
    ad = st.create_stlearn(ad.to_df(), df_spatial, "bgi")
    ad.X = ss.csr_matrix(ad.X)
    return ad


def addFigAsBackground(
    ad,
    fig,
    colName="imagecol",
    rowName="imagerow",
    libraryId="empty",
    backup: Optional[str] = None,
):
    """
    Add a figure as background to the AnnData object.
    """
    canvas = fig.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    image = image[(image != 255).all(2).any(1)]
    image = image[:, (image != 255).all(2).any(0)]
    image = image / 255

    image = image[::-1, :, :]

    if backup is not None:
        ad.uns[backup] = {}
        ad.uns[backup][colName] = ad.obs[colName].copy()
        ad.uns[backup][rowName] = ad.obs[rowName].copy()

    ad.obs["imagecol"] = ad.obs[colName]
    ad.obs["imagerow"] = ad.obs[rowName]

    ad.obs["imagecol"] -= ad.obs["imagecol"].min()
    ad.obs["imagerow"] -= ad.obs["imagerow"].min()

    x_factor = ad.obs["imagecol"].max(0) / image.shape[1]
    y_factor = ad.obs["imagerow"].max(0) / image.shape[0]

    ad.obs["imagecol"] = ad.obs["imagecol"] / x_factor
    ad.obs["imagerow"] = ad.obs["imagerow"] / y_factor

    quality = "hires"
    scale = 1
    spot_diameter_fullres = 30

    if backup is not None:
        ad.uns[backup]["spatial"] = ad.uns["spatial"]
    ad.uns["spatial"] = {}
    ad.uns["spatial"][libraryId] = {}
    ad.uns["spatial"][libraryId]["images"] = {}
    ad.uns["spatial"][libraryId]["images"][quality] = image
    ad.uns["spatial"][libraryId]["use_quality"] = quality
    ad.uns["spatial"][libraryId]["scalefactors"] = {}
    ad.uns["spatial"][libraryId]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale
    ad.uns["spatial"][libraryId]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    ad.obsm["spatial"] = ad.obs[["imagecol", "imagerow"]].values
    ad.obs[["imagecol", "imagerow"]] = ad.obsm["spatial"] * scale


def getClusterScoreFromScDataByDestvi(
    ad_st: sc.AnnData,
    ad_sc: sc.AnnData,
    stLayer: str = "raw",
    scLayer: str = "raw",
    nFeatures: int = 3000,
    clusterKey: str = "leiden",
    batchSize: int = 256,
    condScviEpoch: int = 400,
    destviEpoch: int = 1000,
    minUmiCountsInStLayer: int = 10,
    resultKey: str = "proportions",
    threads: int = 24,
    mannualTraining: bool = False,
    dt_condScviConfigs: Dict = {},
):
    """
    Get cluster score from sc data by destvi.

    Parameters
    ----------
    ad_st : sc.AnnData
    ad_sc : sc.AnnData
    stLayer : str, optional
        by default "raw"
    scLayer : str, optional
        by default "raw"
    nFeatures : int, optional
        highly variable genes counts, by default 3000
    clusterKey : str, optional
        by default 'leiden'
    batchSize : int, optional
        by default 256
    condScviEpoch : int, optional
        by default 400
    destviEpoch : int, optional
        by default 1000
    minUmiCountsInStLayer : int, optional
        by default 10
    resultKey : str, optional
        by default 'proportions'

    Returns
    -------
    obsm of `ad_st` will be updated
    """
    import scvi
    from scvi.model import CondSCVI, DestVI

    scvi.settings.seed = 39
    scvi.settings.num_threads = threads

    ad_stOrg = ad_st
    ad_sc = ad_sc.copy()
    sc.pp.highly_variable_genes(
        ad_sc, n_top_genes=nFeatures, subset=True, layer=scLayer, flavor="seurat_v3"
    )
    intersect = np.intersect1d(ad_st.var_names, ad_sc.var_names)
    ad_st = ad_st[:, intersect].copy()
    ad_sc = ad_sc[:, intersect].copy()

    ad_st = ad_st[ad_st.to_df(scLayer).sum(1) > minUmiCountsInStLayer]
    logger.info(f"var number after filtering: {len(ad_st.var)}")
    logger.info(f"obs number in `ad_st` after filtering: {len(ad_st.obs)}")

    CondSCVI.setup_anndata(ad_sc, layer=scLayer, labels_key=clusterKey)
    model_sc = CondSCVI(ad_sc, weight_obs=True, **dt_condScviConfigs)
    setSeed(39)
    scvi.settings.seed = 39
    model_sc.train(max_epochs=condScviEpoch, batch_size=batchSize)
    model_sc.history["elbo_train"].plot()
    plt.yscale("log")
    plt.title("condVI")
    plt.show()
    if mannualTraining:
        while True:
            contineEpoch = int(input("Continue Epochs? (int)"))
            if contineEpoch > 0:
                setSeed(39)
                scvi.settings.seed = 39
                model_sc.train(max_epochs=contineEpoch, batch_size=batchSize)
                model_sc.history["elbo_train"].plot()
                plt.yscale("log")
                plt.title("condVI")
                plt.show()
            else:
                break

    DestVI.setup_anndata(ad_st, layer=stLayer)
    model_st = DestVI.from_rna_model(ad_st, model_sc)
    model_st.train(max_epochs=destviEpoch, batch_size=batchSize)
    model_st.history["elbo_train"].plot()
    plt.yscale("log")
    plt.title("destVI")
    plt.show()
    if mannualTraining:
        while True:
            contineEpoch = int(input("Continue Epochs? (int)"))
            if contineEpoch > 0:
                setSeed(39)
                scvi.settings.seed = 39
                model_st.train(max_epochs=contineEpoch, batch_size=batchSize)
                model_st.history["elbo_train"].plot()
                plt.yscale("log")
                plt.title("condVI")
                plt.show()
            else:
                break

    df_result = model_st.get_proportions()
    df_resultReindex = (
        df_result.reindex(ad_stOrg.obs.index)
        .fillna(0)
        .assign(keep=lambda df: df.index.isin(df_result.index.to_list()))
    )
    ad_stOrg.obsm[resultKey] = df_resultReindex


class SelectCellInteractive:
    def __init__(self, ad, colName, libraryName=None, scale=None):
        self.ad = ad
        self.colName = colName
        self.dt_selected = {}  # {`selectName`: [step, selector]}

        if libraryName is None:
            libraryName = list(ad.uns["spatial"].keys())[0]

        ## get scale factor
        if scale is None:
            scale = ad.uns["spatial"][libraryName]["scalefactors"][
                "tissue_" + "hires" + "_scalef"
            ]
        self.scale = scale
        self.libraryName = libraryName
        self.ar_image = ad.uns["spatial"][libraryName]["images"]["hires"].copy()

    def polySelect(self, selectName, step=1, figsize=(9, 4)):
        from ..otherTools import SelectByPolygon

        ar_image = self.ar_image[::step, ::step].copy()
        selector = SelectByPolygon(ar_image, figsize)
        self.dt_selected[selectName] = [step, selector]
        # ad_selected = self.ad[
        #     selector.path.contains_points(self.ad.obsm["spatial"] / step)
        # ]
        # self.dt_selected[selectName] = ad_selected

    def exportAd(self, ls_selectName: Optional[Union[List[str], str]] = None):
        if ls_selectName is None:
            ls_selectName = list(self.dt_selected.keys())
        elif isinstance(ls_selectName, str):
            ls_selectName = [ls_selectName]
        dt_ad = {}
        for selectName in ls_selectName:
            selector = self.dt_selected[selectName][1]
            step = self.dt_selected[selectName][0]
            ad_selected = self.ad[
                selector.path.contains_points(
                    self.ad.obsm["spatial"] * self.scale / step
                )
            ]
            dt_ad[selectName] = ad_selected
        ad_export = sc.concat(
            {selectName: dt_ad[selectName] for selectName in ls_selectName},
            label=self.colName,
            index_unique="-",
        ).copy()
        ad_export.uns["spatial"] = self.ad.uns["spatial"]
        return ad_export


# def selectCellInteractive(ad, libraryName, step=1, figsize=(8,4)) -> sc.AnnData:
#     from ..otherTools import SelectByPolygon
#     ar_image = ad.uns['spatial'][libraryName]['images']['hires'][::step, ::step].copy()
#     selector = SelectByPolygon(ar_image, figsize)
#     input("Press Enter to finish selecting cells")
#     ad_ = ad[selector.path.contains_points(ad.obsm['spatial'] * step)].copy() # add scale?
#     return ad_
