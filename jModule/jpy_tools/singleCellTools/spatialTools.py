from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import warnings

warnings.warn(
    "Maybe I will reformat these snippets into a module in the future", ImportWarning
)


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
    scale=True,
    ls_otherQuality: List[int] = [],
):
    """
    Add a figure as background to the AnnData object.
    """
    warnings.warn(
        "This function is highly experimental and may not work properly.", FutureWarning
    )
    if isinstance(fig, mpl.figure.Figure):
        canvas = fig.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        image = image[(image != 255).all(2).any(1)]
        image = image[:, (image != 255).all(2).any(0)]
        image = image / 255

        image = image[::-1, :, :]
    elif isinstance(fig, np.ndarray):
        image = fig

    if backup is not None:
        ad.uns[backup] = {}
        ad.uns[backup][colName] = ad.obs[colName].copy()
        ad.uns[backup][rowName] = ad.obs[rowName].copy()

    ad.obs["imagecol"] = ad.obs[colName]
    ad.obs["imagerow"] = ad.obs[rowName]
    if scale:
        ad.obs["imagecol"] = ad.obs[colName]
        ad.obs["imagerow"] = ad.obs[rowName]

        ad.obs["imagecol"] -= ad.obs["imagecol"].min()
        ad.obs["imagerow"] -= ad.obs["imagerow"].min()

        x_factor = ad.obs["imagecol"].max(0) / image.shape[1]
        y_factor = ad.obs["imagerow"].max(0) / image.shape[0]

        ad.obs["imagecol"] = ad.obs["imagecol"] / x_factor
        ad.obs["imagerow"] = ad.obs["imagerow"] / y_factor
        ad.obsm["spatial"] = ad.obs[["imagecol", "imagerow"]].values

    if backup is not None:
        ad.uns[backup]["spatial"] = ad.uns["spatial"]
    ad.uns["spatial"] = {}
    ad.uns["spatial"][libraryId] = {}
    ad.uns["spatial"][libraryId]["images"] = {}
    ad.uns["spatial"][libraryId]["scalefactors"] = {}
    ls_quality = ["hires", "lowres"]
    ls_scale = 1, 5
    spot_diameter_fullres = 30
    ad.uns["spatial"][libraryId]["use_quality"] = "lowres"
    ad.uns["spatial"][libraryId]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    for quality, scale in zip(ls_quality, ls_scale):
        ad.uns["spatial"][libraryId]["images"][quality] = image[::scale, ::scale]
        ad.uns["spatial"][libraryId]["scalefactors"][
            "tissue_" + quality + "_scalef"
        ] = (1 / scale)
    for scale in ls_otherQuality:
        ad.uns["spatial"][libraryId]["images"][str(scale)] = image[::scale, ::scale]
        ad.uns["spatial"][libraryId]["scalefactors"][
            "tissue_" + str(scale) + "_scalef"
        ] = (1 / scale)
    # quality = "hires"
    # scale = 1
    # spot_diameter_fullres = 30

    # if backup is not None:
    #     ad.uns[backup]["spatial"] = ad.uns["spatial"]
    # ad.uns["spatial"] = {}
    # ad.uns["spatial"][libraryId] = {}
    # ad.uns["spatial"][libraryId]["images"] = {}
    # ad.uns["spatial"][libraryId]["images"][quality] = image
    # ad.uns["spatial"][libraryId]["use_quality"] = quality
    # ad.uns["spatial"][libraryId]["scalefactors"] = {}
    # ad.uns["spatial"][libraryId]["scalefactors"][
    #     "tissue_" + quality + "_scalef"
    # ] = scale
    # ad.uns["spatial"][libraryId]["scalefactors"][
    #     "spot_diameter_fullres"
    # ] = spot_diameter_fullres
    # ad.obsm["spatial"] = ad.obs[["imagecol", "imagerow"]].values
    # ad.obs[["imagecol", "imagerow"]] = ad.obsm["spatial"] * scale


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
    hvgLabel: Optional[str] = None,
    hvgScDataOnly: bool = False,
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
    hvgLabel : Optional[str], optional
        high variance genes label, by default None
    hvgScDataOnly : bool
        which means find hvgs on sc data only or not
    Returns
    -------
    obsm of `ad_st` will be updated
    """
    import scvi
    from scvi.model import CondSCVI, DestVI

    scvi.settings.seed = 39
    scvi.settings.num_threads = threads

    ad_stOrg = ad_st
    intersect = np.intersect1d(ad_st.var_names, ad_sc.var_names)
    ad_sc = ad_sc[:, intersect].copy()
    ad_st = ad_st[:, intersect].copy()
    ad_st.X = ad_st.layers[stLayer].copy()
    ad_sc.X = ad_sc.layers[scLayer].copy()

    ad_merge = (
        ad_sc
        if hvgScDataOnly
        else sc.concat({"st": ad_st, "sc": ad_sc}, label="_category", index_unique="-")
    )
    sc.pp.highly_variable_genes(
        ad_merge,
        n_top_genes=nFeatures,
        subset=True,
        flavor="seurat_v3",
        batch_key=hvgLabel,
    )
    ls_hvg = ad_merge.var.index.to_list()

    # intersect = np.intersect1d(ad_st.var_names, ad_sc.var_names)
    ad_st = ad_st[:, ls_hvg].copy()
    ad_sc = ad_sc[:, ls_hvg].copy()

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
    setSeed(39)
    scvi.settings.seed = 39
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
    def __init__(
        self,
        ad,
        colName,
        libraryName=None,
        scale=None,
        mode: Literal["keep", "remove"] = "keep",
    ):
        self.ad = ad
        self.colName = colName
        self.dt_selected = {}  # {`selectName`: [step, selector, 'keep|remove']}

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
        self.mode = mode

    def polySelect(
        self, selectName, step=1, figsize=(9, 4), dt_lineprops={}, dt_markerprops={}
    ):
        from ..otherTools import SelectByPolygon

        ar_image = self.ar_image[::step, ::step].copy()
        selector = SelectByPolygon(ar_image, figsize, dt_lineprops, dt_markerprops)
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
        if self.mode == "keep":
            dt_ad = {}
            for selectName in ls_selectName:
                step, selector = self.dt_selected[selectName]
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
        elif self.mode == "remove":
            ls_removeObs = []
            for selectName in ls_selectName:
                step, selector = self.dt_selected[selectName]
                removeObs = self.ad[
                    selector.path.contains_points(
                        self.ad.obsm["spatial"] * self.scale / step
                    )
                ].obs.index.to_list()
                ls_removeObs.extend(removeObs)
            ad_export = self.ad[
                [x for x in self.ad.obs.index if x not in ls_removeObs]
            ].copy()
        ad_export.uns["spatial"] = self.ad.uns["spatial"]
        return ad_export


# def selectCellInteractive(ad, libraryName, step=1, figsize=(8,4)) -> sc.AnnData:
#     from ..otherTools import SelectByPolygon
#     ar_image = ad.uns['spatial'][libraryName]['images']['hires'][::step, ::step].copy()
#     selector = SelectByPolygon(ar_image, figsize)
#     input("Press Enter to finish selecting cells")
#     ad_ = ad[selector.path.contains_points(ad.obsm['spatial'] * step)].copy() # add scale?
#     return ad_


def trimBg(ad, libraryId=None):
    import copy

    warnings.warn(
        "This function is not compatible with stlearn anndata format now.",
        FutureWarning,
    )
    ad.uns["spatial"] = copy.deepcopy(ad.uns["spatial"])
    if libraryId is None:
        libraryId = list(ad.uns["spatial"].keys())[0]
    ls_allQuality = ad.uns["spatial"][libraryId]["images"].keys()
    for i, imageName in enumerate(ls_allQuality):
        image = ad.uns["spatial"][libraryId]["images"][imageName]
        scaleFactor = ad.uns["spatial"][libraryId]["scalefactors"][
            f"tissue_{imageName}_scalef"
        ]
        ax = sc.pl.spatial(ad, size=0.2, show=False, img_key=imageName)[0]
        plt.close()
        left, right, top, bottom = [*ax.get_xlim(), *ax.get_ylim()] | F(
            map, round
        )  # get position
        imageTrimmed = image[bottom:top, left:right]
        ad.uns["spatial"][libraryId]["images"][imageName] = imageTrimmed
        if i == 0:
            ad.obsm["spatial"][:, 0] = ad.obsm["spatial"][:, 0] - (left / scaleFactor)
            ad.obsm["spatial"][:, 1] = ad.obsm["spatial"][:, 1] - (bottom / scaleFactor)


def rotateBgAndObsm(ad, angle, libraryId, imgKey):
    '''This function takes in an angle, a libraryId, and an imageKey, and rotates the background and
    obstacle images in the library with the given libraryId by the given angle.
    
    Parameters
    ----------
    ad
        the ad object
    angle
        the angle to rotate the image by
    libraryId
        the id of the library that the image is in
    imgKey
        the key of the image to be rotated, other images will be deleted
    '''
    from PIL import Image
    import math

    def shear(angle, x, y, needRound=True):
        """
        |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) |
        |0      1     |  |sin(𝜃)   1|  |0      1     |
        """
        # shear 1
        tangent = math.tan(angle / 2)
        if needRound:
            new_x = round(x - y * tangent)
            new_y = y

            # shear 2
            new_y = round(
                new_x * math.sin(angle) + new_y
            )  # since there is no change in new_x according to the shear matrix

            # shear 3
            new_x = round(
                new_x - new_y * tangent
            )  # since there is no change in new_y according to the shear matrix
        else:
            new_x = x - y * tangent
            new_y = y

            # shear 2
            new_y = (
                new_x * math.sin(angle) + new_y
            )  # since there is no change in new_x according to the shear matrix

            # shear 3
            new_x = (
                new_x - new_y * tangent
            )  # since there is no change in new_y according to the shear matrix

        return new_y, new_x

    ad = ad.copy()
    ls_allQuality = list(ad.uns["spatial"][libraryId]["images"].keys())
    for i, imageName in enumerate(ls_allQuality):
        if imageName != imgKey:
            del ad.uns["spatial"][libraryId]["images"][imageName]
            del ad.uns["spatial"][libraryId]["scalefactors"][
                f"tissue_{imageName}_scalef"
            ]
        else:
            image = ad.uns["spatial"][libraryId]["images"][imageName]
            scalef = ad.uns["spatial"][libraryId]["scalefactors"][
                f"tissue_{imageName}_scalef"
            ]
    angle = math.radians(angle)  # converting degrees to radians
    cosine = math.cos(angle)
    sine = math.sin(angle)

    # rotate image
    height = image.shape[0]  # define the height of the image
    width = image.shape[1]

    new_height = round(abs(image.shape[0] * cosine) + abs(image.shape[1] * sine)) + 1
    new_width = round(abs(image.shape[1] * cosine) + abs(image.shape[0] * sine)) + 1

    output = np.zeros((new_height, new_width, image.shape[2]))
    image_copy = output.copy()

    original_centre_height = round(
        ((image.shape[0] + 1) / 2) - 1
    )  # with respect to the original image
    original_centre_width = round(
        ((image.shape[1] + 1) / 2) - 1
    )  # with respect to the original image

    new_centre_height = round(
        ((new_height + 1) / 2) - 1
    )  # with respect to the new image
    new_centre_width = round(((new_width + 1) / 2) - 1)  # with respect to the new image

    for i in range(height):
        for j in range(width):
            # co-ordinates of pixel with respect to the centre of original image
            y = image.shape[0] - 1 - i - original_centre_height
            x = image.shape[1] - 1 - j - original_centre_width
            # Applying shear Transformation
            new_y, new_x = shear(angle, x, y)
            """since image will be rotated the centre will change too, 
                so to adust to that we will need to change new_x and new_y with respect to the new centre"""
            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x
            # adding if check to prevent any errors in the processing
            if (
                0 <= new_x < new_width
                and 0 <= new_y < new_height
                and new_x >= 0
                and new_y >= 0
            ):
                output[new_y, new_x, :] = image[
                    i, j, :
                ]  # writing the pixels to the new destination in the output image
    output = (output).astype(np.uint8)
    ad.uns["spatial"][libraryId]["images"][imgKey] = output
    ls_newSpatial = []
    for (x, y) in ad.obsm["spatial"]:
        x = x * scalef
        y = y * scalef
        y = image.shape[0] - 1 - y - original_centre_height
        x = image.shape[1] - 1 - x - original_centre_width
        new_y, new_x = shear(angle, x, y, needRound=False)
        """since image will be rotated the centre will change too, 
            so to adust to that we will need to change new_x and new_y with respect to the new centre"""
        new_y = new_centre_height - new_y
        new_x = new_centre_width - new_x
        ls_newSpatial.append([new_x, new_y])
    ad.obsm["spatial"] = np.array(ls_newSpatial)
    ad.uns["spatial"][libraryId]["scalefactors"][f"tissue_{imgKey}_scalef"] = 1
    return ad


def normalieBySME(
    ad: sc.AnnData,
    layer: str = "normalize_log",
    dir_temp="/tmp/tiling/",
    weights: Literal[
        "weights_matrix_all",
        "weights_matrix_pd_gd",
        "weights_matrix_pd_md",
        "weights_matrix_gd_md",
        "gene_expression_correlation",
        "physical_distance",
        "morphological_distance",
    ] = "weights_matrix_all",
):
    """
    stlearn.SME.SME_normalize wrapper

    Parameters
    ----------
    ad : sc.AnnData
        requires :
        `.obs[['array_row', 'array_col']]`
        `.obsm['spatial']`
        `.uns['spatial']`
    layer : str
        normalize_log
    dir_temp : str
        directory for temporary files

    Returns
    -------
    'SME_normalized' matrix will be added into ad.layers
    """
    from pathlib import Path

    TILE_PATH = Path(dir_temp)
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    assert "spatial" in ad.uns.keys(), "ad.uns['spatial'] is required"
    assert "spatial" in ad.obsm.keys(), "ad.obsm['spatial'] is required"
    assert "array_row" in ad.obs.columns, "ad.obs['array_row'] is required"
    assert "array_col" in ad.obs.columns, "ad.obs['array_col'] is required"

    adOrg = ad
    ad = ad.copy()
    ad = st.convert_scanpy(ad)
    ad.X = ad.layers[layer].copy()
    st.pp.tiling(ad, TILE_PATH)
    st.pp.extract_feature(ad)

    st.em.run_pca(ad, n_comps=50)
    st.spatial.SME.SME_normalize(
        ad, use_data="raw", platform="Visium", weights=weights
    )  # raw means use `.X`
    adOrg.layers["SME_normalized"] = ad.obsm["raw_SME_normalized"].copy()
    tqdm.get_lock().locks = []  # release tqdm lock; it seems is a stlearn bug
