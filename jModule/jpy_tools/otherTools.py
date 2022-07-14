"""
@Date: 2020-06-05 22:08:50
LastEditors: liuzj
LastEditTime: 2021-01-29 13:20:18
@Description: 无法归类的工具
@Author: liuzj
FilePath: /jpy_tools/otherTools.py
"""
import os
import sh
import pandas as pd
import numpy as np
from loguru import logger
from io import StringIO
import sys
from threading import Thread
import matplotlib.pyplot as plt
import seaborn as sns
from cool import F
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import torch
from typing import (
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
    Dict,
)
from jpy_tools import settings


def setSeed(seed=0):
    import os
    import numpy as np
    import random
    import rpy2.robjects as ro

    R = ro.r

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    R("set.seed")(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class Capturing(list):
    "Capture std output"

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def mkdir(dirPath):
    try:
        sh.mkdir(dirPath)
    except:
        logger.warning(f"{dirPath} existed!!")


class Jinterval:
    """
    自己写的区间操作， 极其不完善
    """

    def __init__(self, lower, upper, overlapLimit=0.5):
        self.lower, self.upper = lower, upper
        self.interval = [lower, upper]
        self.overlapLimit = overlapLimit

    def __repr__(self):
        return f"Jinterval{self.interval}"

    def __str__(self):
        return f"Jinterval{self.interval}"

    def __and__(self, otherInterval):
        minn = max(self.lower, otherInterval.lower)
        maxn = min(self.upper, otherInterval.upper)
        if (maxn - minn) / (self.upper - self.lower) > self.overlapLimit:
            return [minn, maxn]
        else:
            return False

    def getOverlapRatio(self, otherInterval):
        minn = max(self.lower, otherInterval.lower)
        maxn = min(self.upper, otherInterval.upper)
        return max((maxn - minn) / (self.upper - self.lower), 0)


def creatUnexistedDir(directory):
    """
    @description: 目录不存在时创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def isOne(n, i):
    """
    @description: 判断n的从右向左第i位是否为1
    @param
        n:{int}
        i:{int}
    @return:
        bool
    """
    return (n & (1 << i)) != 0


def groupby(dtframe, key):
    """
    @description: 用于groupby操作
    """
    dtframe.sort_values(key, inplace=True, ignore_index=True)
    dtframeCol = dtframe[key].values
    i = 0
    j = 0
    forwardName = dtframeCol[0]
    for x in range(len(dtframeCol)):
        currentName = dtframeCol[x]
        if currentName == forwardName:
            j += 1
            pass
        else:
            yield dtframe[i:j]
            forwardName = currentName
            i = j
            j += 1
    yield dtframe[i:]


def myAsync(f):
    def wrapper(*args, **kwargs):
        logger.warning("async, you will don't get any results from this function")
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
        return thr

    return wrapper


def addColorLegendToAx(
    ax,
    title,
    colorDt: Mapping[str, str],
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(1.05, 1.0),
    ha="center",
    **legendParamsDt,
):
    from matplotlib.legend import Legend

    ax = ax.twinx()
    ax.get_yaxis().set_visible(False)
    artistLs = []
    for label, color in colorDt.items():
        artistLs.append(ax.bar(0, 0, color=color, label=label, linewidth=0))
    # leg = Legend(
    #     ax,
    #     artistLs,
    #     list(colorDt.keys()),
    #     title=title,
    #     loc=loc,
    #     ncol=ncol,
    #     bbox_to_anchor=bbox_to_anchor,
    #     **legendParamsDt,
    # )
    # leg._legend_box.align = "left"
    # ax.add_artist(leg)
    plt.sca(ax)
    leg = ax.legend(
        handles=artistLs,
        labels=list(colorDt.keys()),
        title=title,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        **legendParamsDt,
    )
    leg._legend_box.align = ha
    # leg = ax.legend(title=title, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)
    return ax.legend


def sankeyPlotByPyechart(
    df: pd.DataFrame,
    columns: Sequence[str],
    figsize=[5, 5],
    colorDictLs: Optional[List[Dict[str, str]]] = None,
    defaultJupyter: Literal["notebook", "lab"] = "notebook",
    needCounts: bool = False,
):
    """
    [summary]

    Parameters
    ----------
    df : pd.DataFrame
    columns : Sequence[str]
        the columns used for sankey plot
    figsize : list, optional
        by default [5,5]
    colorDict : Optional[List[Dict[str,str]]], optional
        Color of each label in dataframe. e.g. [{'a': '#000000', 'b': '#000001'}, {'a': '#000000', 'b': '#000001'}]. Length should consistent with columns
        by default None.

    Returns
    -------
    pyecharts.charts.basic_charts.sankey.Sankey
        Utilize Function render_notebook can get the final figure
    """
    from pyecharts.globals import CurrentConfig, NotebookType

    CurrentConfig.NOTEBOOK_TYPE = (
        NotebookType.JUPYTER_NOTEBOOK
        if defaultJupyter == "notebook"
        else NotebookType.JUPYTER_LAB
    )
    CurrentConfig.ONLINE_HOST
    from matplotlib.colors import rgb2hex
    from pyecharts import options as opts
    from pyecharts.charts import Sankey
    from scanpy.plotting import palettes

    def getSankeyFormatFromDf_Only2(
        df: pd.DataFrame,
        fromCol: str,
        toCol: str,
        fromColorDt: dict = None,
        toColorDt: dict = None,
        layerNum: int = 0,
        skNameLs: list = None,
    ):
        if not skNameLs:
            skNameLs = []

        skUseDt = (
            df[[fromCol, toCol]]
            .groupby([fromCol, toCol])
            .agg(lambda x: len(x))
            .to_dict()
        )

        if not fromColorDt:
            length = len(df[fromCol].unique())
            if length <= 20:
                palette = palettes.default_20
            elif length <= 28:
                palette = palettes.default_28
            elif length <= len(palettes.default_102):  # 103 colors
                palette = palettes.default_102
            else:
                palette = ["grey" for _ in range(length)]

            fromColorDt = {x: rgb2hex(y) for x, y in zip(df[fromCol].unique(), palette)}
        if not toColorDt:
            length = len(df[toCol].unique())
            if length <= 20:
                palette = palettes.default_20
            elif length <= 28:
                palette = palettes.default_28
            elif length <= len(palettes.default_102):  # 103 colors
                palette = palettes.default_102
            else:
                palette = ["grey" for _ in range(length)]

            toColorDt = {x: rgb2hex(y) for x, y in zip(df[toCol].unique(), palette)}

        fromColorDt = {f"{x}{' ' * layerNum}": y for x, y in fromColorDt.items()}
        toColorDt = {f"{x}{' ' * (layerNum+1)}": y for x, y in toColorDt.items()}

        skNodeLs = []
        skLinkLs = []
        for (source, target), counts in skUseDt.items():
            source = source + " " * layerNum
            target = target + " " * (layerNum + 1)
            if source not in skNameLs:
                skNameLs.append(source)
                sourceColor = fromColorDt[source]
                skNodeLs.append({"name": source, "itemStyle": {"color": sourceColor}})
            if target not in skNameLs:
                skNameLs.append(target)
                targetColor = toColorDt[target]
                skNodeLs.append({"name": target, "itemStyle": {"color": targetColor}})

            skLinkLs.append({"source": source, "target": target, "value": counts})

        return skNodeLs, skLinkLs, skNameLs

    skNodeLs = []
    skLinkLs = []
    skNameLs = []
    if not colorDictLs:
        colorDictLs = [None] * len(columns)
    df = df.copy()
    if needCounts:
        for i in range(len(columns)):
            dt_newName = df.loc[:, columns[i]].value_counts().to_dict() | F(
                lambda z: {x: f"{x}\nN = {y}" for x, y in z.items()}
            )
            df.loc[:, columns[i]] = df.loc[:, columns[i]].map(dt_newName)
            colorDictLs[i] = {dt_newName[x]: y for x, y in colorDictLs[i].items()}

    for i in range(len(columns) - 1):
        partSkNodeLs, partSkLinkLs, skNameLs = getSankeyFormatFromDf_Only2(
            df,
            columns[i],
            columns[i + 1],
            colorDictLs[i],
            colorDictLs[i + 1],
            i,
            skNameLs,
        )
        skNodeLs.extend(partSkNodeLs)
        skLinkLs.extend(partSkLinkLs)

    sankey = Sankey(
        init_opts=opts.InitOpts(
            width=f"{figsize[0]*100}px",
            height=f"{figsize[1]*100}px",
        )
    )

    sankey.add(
        "",
        skNodeLs,
        skLinkLs,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right"),
    ).set_global_opts(
        title_opts=opts.TitleOpts(title=""),
    )

    return sankey


def copyFromIpf(ipfPath) -> str:
    import sh

    tmpPath = "/scem/work/liuzj/tmp/1"
    sh.scp(f"172.18.6.205:{ipfPath}", tmpPath)
    return tmpPath


def copyToIpf(inPath, ipfPath) -> str:
    import sh

    sh.scp(inPath, f"172.18.6.205:{ipfPath}")


def toPkl(
    obj, name, server, config=None, writeFc=None, arg_path=None, dir_path=None, **dt_arg
):
    """

    Parameters
    ----------
    obj
    name :
        data file name
    server :
        ipf|scem
    config :
        will overwrite writeFc, arg_path and dt_arg
        support:
            scvi_model|mudata
    writeFc :
        write function
    arg_path : [type], optional
        argument of save path
    """
    import pickle
    import os

    dt_config = {
        "scvi.model.base._base_model.BaseModelClass": {
            "writeFc": lambda x, **dt: x.save(**dt),
            "arg_path": "dir_path",
            "dt_arg": {"overwrite": True},
            "readFc": "lambda **dt:scvi.model.SCVI.load(**dt), arg_path='dir_path', adata=ad",
        },
        "mudata._core.mudata.MuData": {
            "writeFc": lambda x, **dt: x.write_h5mu(**dt),
            "arg_path": "filename",
            "dt_arg": {},
            "readFc": "lambda **dt:mu.read_h5mu(**dt), arg_path='filename'",
        },
        "anndata._core.anndata.AnnData": {
            "writeFc": lambda x, **dt: x.write_h5ad(**dt),
            "arg_path": "filename",
            "dt_arg": {},
            "readFc": "lambda **dt:sc.read_h5ad(**dt), arg_path='filename'",
        },
    }

    dt_dirPkl = settings.dt_dirPkl
    # dt_dirPkl = {
    #     "ipf": "/public/home/liuzj/tmp/python_pkl/",
    #     "scem": "/scem/work/liuzj/tmp/python_pkl/",
    # }
    dt_ip = {"ipf": "172.18.6.205", "scem": "172.18.5.205"}
    dt_scpConfig = {"ipf": "", "scem": "-P 2323"}

    dt_currentServer = {x: os.path.exists(y) for x, y in dt_dirPkl.items()}
    ls_currentServer = [x for x, y in dt_currentServer.items() if y]
    assert len(ls_currentServer) == 1, "Unknown current server"
    currentServer = ls_currentServer[0]

    dir_currentPkl = dt_dirPkl[currentServer]
    if dir_path:
        dir_currentPkl = dir_path
    if config is None:
        import scvi, mudata, anndata

        for className in dt_config.keys():
            class_obj = eval(className)
            if isinstance(obj, class_obj):
                config = className
                break
    if config is not None:
        config = dt_config[config]
        writeFc = config["writeFc"]
        arg_path = config["arg_path"]
        dt_arg = config["dt_arg"]
        logger.info(f"please run `loadPkl('{name}', {config['readFc']})` to get object")

    if not writeFc:
        with open(f"{dir_currentPkl}/{name}", "wb") as fh:
            pickle.dump(obj, fh)
    else:
        dt_arg.update({arg_path: f"{dir_currentPkl}/{name}"})
        writeFc(obj, **dt_arg)

    if server != currentServer:
        dir_pkl = dt_dirPkl[server]
        ip_target = dt_ip[server]
        config_scp = dt_scpConfig[server]
        print(
            os.system(
                f"scp -r {config_scp} {dir_currentPkl}/{name} {ip_target}:{dir_pkl}/"
            )
        )


def loadPkl(name: str, readFc=None, arg_path=None, dir_path=None, **dt_arg):
    """

    Parameters
    ----------
    obj
    name :
        data file name
    server :
        ipf|scem
    readFc :
        read function
    arg_path : [type], optional
        argument of save path
    """
    import pickle

    if name.startswith("/"):
        dir_currentPkl = ""
    else:
        dt_dirPkl = settings.dt_dirPkl
        # dt_dirPkl = {
        #     "ipf": "/public/home/liuzj/tmp/python_pkl/",
        #     "scem": "/scem/work/liuzj/tmp/python_pkl/",
        # }

        dt_currentServer = {x: os.path.exists(y) for x, y in dt_dirPkl.items()}
        ls_currentServer = [x for x, y in dt_currentServer.items() if y]
        assert len(ls_currentServer) == 1, "Unknown current server"
        currentServer = ls_currentServer[0]
        dir_currentPkl = dt_dirPkl[currentServer]
    if dir_path:
        dir_currentPkl = dir_path

    if not readFc:
        with open(f"{dir_currentPkl}/{name}", "rb") as fh:
            obj = pickle.load(fh)
    else:
        dt_arg.update({arg_path: f"{dir_currentPkl}/{name}"})
        obj = readFc(**dt_arg)

    return obj


def getGoDesc(goTerm: Union[str, List[str]], retry=5) -> pd.DataFrame:
    """
    query GO term description from QuickGO

    Parameters
    ----------
    goTerm :
        go term
    retry :
        retry times

    Returns
    -------
    pd.DataFrame
    """
    import requests, sys, json
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm

    def _getGOcomment(goTerm, retry=5):
        _goTerm = goTerm.split(":")[-1]
        requestURL = (
            f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A{_goTerm}"
        )
        for i in range(retry):
            try:
                r = requests.get(requestURL, headers={"Accept": "application/json"})
                break
            except:
                pass

        if not r.ok:
            r.raise_for_status()
            sys.exit()

        responseBody = r.text
        return goTerm, responseBody

    if isinstance(goTerm, str):
        goTerm = [goTerm]
    for x in goTerm:
        assert x.startswith("GO:"), f"Wrong format: {x}"

    ls_goTerm = Parallel(128, "threading")(
        delayed(_getGOcomment)(x, retry) for x in tqdm(goTerm, position=0)
    )

    dt_go = {}
    for name, dt_singleGo in ls_goTerm:
        dt_singleGo = json.loads(dt_singleGo)
        dt_singleGoFirstHit = dt_singleGo["results"][0]
        dt_go[name] = {
            "hitGO": dt_singleGoFirstHit["id"],
            "hitName": dt_singleGoFirstHit["name"] + f" ({name})",
            "hitDefinition": dt_singleGoFirstHit["definition"]["text"],
            "hitCounts": dt_singleGo["numberOfHits"],
            "aspect": dt_singleGoFirstHit["aspect"],
        }
        if name != dt_go[name]["hitGO"]:
            logger.warning(f"query : {name}, target : {dt_go[name]['hitGO']}")
    df_go = pd.DataFrame.from_dict(dt_go, "index")
    return df_go


from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt


class SelectByPolygon:
    def __init__(self, ar_image, figsize=(10, 4), dt_lineprops={}, dt_markerprops={}):
        self.ar_image = ar_image
        self.imShape = ar_image.shape[:2]
        self.xy = np.array(
            [
                [x, y]
                for x in range(self.ar_image.shape[1])
                for y in range(self.ar_image.shape[0])
            ]
        )
        if len(ar_image.shape) == 2:
            self.empty = 0
        elif len(ar_image[0, 0]) == 3:
            self.empty = np.array([0, 0, 0])
        elif len(ar_image[0, 0]) == 4:
            self.empty = np.array([0, 0, 0, 0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.imshow(self.ar_image)

        self.selectedImage = np.zeros_like(self.ar_image)
        self.ax2.imshow(self.selectedImage)
        plt.subplots_adjust()
        self.poly = PolygonSelector(
            self.ax1, self.onselect, lineprops=dt_lineprops, markerprops=dt_markerprops
        )

        logger.warning(
            "This function is only available in jupyter environment and you should run `%matplotlib widget` before execute this function"
        )
        print("Select points in the figure by enclosing them within a polygon.")
        print("Press the 'esc' key to start a new polygon.")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")

    def onselect(self, verts):
        path = Path(verts)
        self.path = path
        self.ind = path.contains_points(self.xy).reshape(self.imShape, order="F")
        self.selectedImage = self.ar_image.copy()
        self.selectedImage[~self.ind] = self.empty
        self.ax2.clear()
        self.ax2.imshow(self.selectedImage)
        plt.subplots_adjust()

    def disconnect(self):
        self.poly.disconnect_events()
        plt.close()


def pwStack(ls_ax, ncols=5):
    import patchworklib as pw
    from more_itertools import chunked
    from cool import F

    ls_ax = chunked(ls_ax, ncols) | F(list)
    if len(ls_ax) == 1:
        axs = pw.stack(ls_ax[0])
    elif len(ls_ax) % ncols == 0:
        axs = pw.stack([pw.stack(x) for x in ls_ax], operator="/")
    else:
        axs = pw.stack([pw.stack(x) for x in ls_ax[:-1]], operator="/")
        ls_name = list(axs.bricks_dict.keys())
        for i, ax in enumerate(ls_ax[-1]):
            axs = axs[ls_name[i]] / ax
    return axs


def pwRecoverSeaborn():
    ls_seabornParams = [
        # sns.axisgrid.Grid._figure,
        sns.axisgrid.Grid.add_legend,
        sns.axisgrid.FacetGrid.__init__,
        sns.axisgrid.FacetGrid.despine,
        sns.axisgrid.PairGrid.__init__,
        sns.axisgrid.JointGrid.__init__,
        sns.matrix.ClusterGrid.__setattr__,
    ]

    def _recoverSeaborn():
        (
            # sns.axisgrid.Grid._figure,
            sns.axisgrid.Grid.add_legend,
            sns.axisgrid.FacetGrid.__init__,
            sns.axisgrid.FacetGrid.despine,
            sns.axisgrid.PairGrid.__init__,
            sns.axisgrid.JointGrid.__init__,
            sns.matrix.ClusterGrid.__setattr__,
        ) = ls_seabornParams
        del sns.axisgrid.Grid._figure

    return _recoverSeaborn


def mergePdf(dir_inputPath, path_mergedPdf):
    import glob
    from PyPDF2 import PdfFileMerger

    ls_pdfPath = glob.glob(f"{dir_inputPath}/*.pdf")
    pdfMerger = PdfFileMerger()
    [pdfMerger.append(x) for x in ls_pdfPath]
    pdfMerger.write(path_mergedPdf)
    pdfMerger.close()


def clusterWithKmeans(
    df_mat, nClusters, minDistance = 1, kwargs_to_kmeans={}, kwargs_to_clustermap={}
) -> Tuple[np.ndarray, List[str], Mapping[str, List[str]]]:
    """It takes a dataframe of features and their values, and clusters the features using kmeans. It then
    uses hierarchical clustering to cluster the features within each kmeans cluster. It then combines
    the kmeans and hierarchical clustering dendrograms into one dendrogram

    Parameters
    ----------
    df_mat
        the dataframe of the data you want to cluster
    nClusters
        number of clusters to use for kmeans
    kwargs_to_kmeans
        parameters to pass to KMeans
    kwargs_to_clustermap
        parameters to pass to sns.clustermap

    Returns
    ---------
    hierachicalWithKmeans (namedlist):
        dendrogram linkage
            the linkage matrix of the dendrogram
        ls_featureOrder
            the order of the features in the dendrogram
        dt_kMeansCluster
            the kmeans clustering results
        kMeans
            the kmeans result
    """
    from sklearn.cluster import KMeans
    from namedlist import namedlist

    def _getNewFeatureIndex(i, dt_featureCounts, totalFeatureCounts):
        if dt_featureCounts[i] > 1:
            featureIndex = (
                dt_featureCounts[i] - 1 - 1 + totalFeatureCounts
            )  # 0-index and newFeatureCounts = total Feature Counts - 1
            #     print(featureIndex)
            for j in range(i):
                featureIndex += dt_featureCounts[j] - 1
        #         print(featureIndex)
        else:
            featureIndex = 0
            for j in range(i):
                featureIndex += dt_featureCounts[j]
        return featureIndex

    kmeans = KMeans(n_clusters=nClusters, random_state=0, **kwargs_to_kmeans).fit(
        df_mat.T
    )
    df_distance = pd.DataFrame(
        kmeans.transform(df_mat.values.T), index=df_mat.columns
    ).assign(
        minDistance=lambda df: df.min(1),
        module=lambda df: df.idxmin(1)
    ).query("minDistance <= @minDistance")
    ls_usedFeature = df_distance.index.tolist()

    df_clusterFeature = pd.DataFrame(
        [kmeans.labels_, df_mat.columns], index=["cluster", "feature"]
    ).T.query("feature in @ls_usedFeature")
    dt_clusterFeature = (
        df_clusterFeature.groupby("cluster")["feature"].agg(list).to_dict()
    )
    ls_clusterOrder = sorted(list(dt_clusterFeature.keys()), key=lambda x: len(dt_clusterFeature[x]), reverse=True)
    dt_new2OldCluster = {x:y for x,y in enumerate(ls_clusterOrder)}
    dt_clusterFeature = {x:dt_clusterFeature[dt_new2OldCluster[x]] for x in dt_new2OldCluster.keys()}

    dt_clusterFeature = {
        i: dt_clusterFeature[x]
        for i, x in enumerate(
            sorted(
                dt_clusterFeature, key=lambda x: len(dt_clusterFeature[x]), reverse=True
            )
        )
    }
    assert (
        len(dt_clusterFeature[0]) > 1
    ), "expected more than 2 features in the biggest cluster"

    df_mat = df_mat[sum(list(dt_clusterFeature.values()), [])]

    dt_dendrogram = {}
    dt_featureCounts = {}
    dt_dendrogramCounts = {}
    for cluster, ls_features in dt_clusterFeature.items():
        dt_featureCounts[cluster] = len(ls_features)
        if len(ls_features) > 1:
            g = sns.clustermap(
                df_mat[ls_features], row_cluster=False, **kwargs_to_clustermap
            )
            plt.close()
            dt_dendrogram[cluster] = g.dendrogram_col.calculated_linkage
            dt_dendrogramCounts[cluster] = dt_dendrogram[cluster].shape[0]
    totalFeatureCounts = df_mat.shape[1]

    currentCounts = 0
    currentGroup = 0
    for cluster, ar_dendrogram in dt_dendrogram.items():
        if dt_featureCounts[cluster] > 1:
            ar_dendrogram[:, 0:2] = np.where(
                ar_dendrogram[:, 0:2] < dt_featureCounts[cluster],
                ar_dendrogram[:, 0:2] + currentCounts,
                ar_dendrogram[:, 0:2]
                - dt_featureCounts[cluster]
                + totalFeatureCounts
                + currentCounts
                - currentGroup,
            )
        currentCounts += dt_featureCounts[cluster]
        currentGroup += 1

    maxDistance = np.concatenate(list(dt_dendrogram.values()))[:, 2].max() * 1.1

    ls_kmeansDend = []
    for i in range(len(dt_featureCounts) - 1):
        if i == 0:
            ls_kmeansDend.append(
                [
                    _getNewFeatureIndex(i + 1, dt_featureCounts, totalFeatureCounts),
                    _getNewFeatureIndex(i, dt_featureCounts, totalFeatureCounts),
                    maxDistance,
                    dt_featureCounts[i] + dt_featureCounts[i + 1],
                ]
            )
        else:
            ls_kmeansDend.append(
                [
                    _getNewFeatureIndex(i + 1, dt_featureCounts, totalFeatureCounts),
                    totalFeatureCounts
                    + totalFeatureCounts
                    - 1
                    - len(dt_featureCounts)
                    + i,
                    maxDistance,
                    sum([dt_featureCounts[j] for j in range(i + 2)]),
                ]
            )
    HierachicalWithKmeans = namedlist(
        "HierachicalWithKmeans",
        ["linkage", "featureOrder", "kMeansCluster", "kMeans", "distance", "clusterRenameInfo"],
    )
    df_distance = pd.DataFrame(
        kmeans.transform(df_mat.values.T), index=df_mat.columns
    ).rename(columns = {y:x for x,y in dt_new2OldCluster.items()}).sort_index(axis=1).assign(
        minDistance=lambda df: df.min(1),
        module=lambda df: df.idxmin(1)
    ).query('minDistance <= @minDistance')
    hierachicalWithKmeans = HierachicalWithKmeans(
        np.concatenate((*list(dt_dendrogram.values()), np.array(ls_kmeansDend))),
        df_mat.columns.to_list(),
        dt_clusterFeature,
        kmeans, df_distance, dt_new2OldCluster
    )
    return hierachicalWithKmeans