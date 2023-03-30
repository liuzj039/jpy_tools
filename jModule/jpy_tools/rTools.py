##################################################################################
#                              Author: Author Name                               #
# File Name: /public/home/liuzj/softwares/python_scripts/jpy_modules/jpy_tools/rTools.py#
#                     Creation Date: March 23, 2021 10:42 AM                     #
#                     Last Updated: March 23, 2021 10:46 AM                      #
#                            Source Language: python                             #
#           Repository: https://github.com/liuzj039/myPythonTools.git            #
#                                                                                #
#                            --- Code Description ---                            #
#                    use R in python. forked from gokceneraslan                  #
##################################################################################

# from gokceneraslan
import numpy as np
import pandas as pd
from jpy_tools import settings
import functools
import scipy.sparse as sp
from contextlib import contextmanager
from rpy2.robjects.lib import grdevices
from IPython.display import Image, display
import scanpy as sc
import muon as mu
import h5py
from tempfile import TemporaryDirectory
import sys
import inspect
import re
from rpy2.rinterface import evaluation_context
import rpy2.robjects as ro
from rpy2 import rinterface_lib
from rpy2.robjects import rl
from rpy2.robjects.packages import importr
from tempfile import TemporaryDirectory
import pickle
import sys
import os
from loguru import logger
from .otherTools import Capturing, F
# import pyarrow

# pyarrow.Table.from_pandas(pd.DataFrame([1,2,3])) # this package should be imported before arrow(R)

R = ro.r
# seo = importr("SeuratObject")
rBase = importr("base")
rUtils = importr("utils")
# importr("magrittr")


def rcontext(func):
    """
    A decorator to run a function in an R context.
    `rEnv` parameter will be auto updated
    """

    @functools.wraps(func)
    def wrapper(*args, **kargs):
        dt_parsedKargs = inspect.signature(func).bind_partial(*args, **kargs).arguments
        if not "rEnv" in dt_parsedKargs:
            kargs["rEnv"] = None

        rEnv = kargs["rEnv"]
        if rEnv is None:
            clearEnv = True
            rEnv = ro.Environment()
        else:
            clearEnv = False
        kargs["rEnv"] = rEnv

        if not "rEnv" in inspect.signature(func).parameters:
            kargs.pop("rEnv")
        try:
            with ro.local_context(rEnv) as rlc:
                result = func(*args, **kargs)
        except rinterface_lib.embedded.RRuntimeError as e:
            ro.r.traceback()
            raise e
        if clearEnv:
            rEnv.clear()
        ro.r.gc()
        return result

    return wrapper


def rpy2_check(func):
    """Decorator to check whether rpy2 is installed at runtime"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import rpy2
        except ImportError:
            raise ImportError("Please install rpy2 package.")
        return func(*args, **kwargs)

    return wrapper


def anndata2ri_check(func):
    """Decorator to check whether anndata2ri is installed at runtime"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import anndata2ri
        except ImportError:
            raise ImportError("Please install anndata2ri package.")
        return func(*args, **kwargs)

    return wrapper


@rpy2_check
def r_is_installed(package_name):
    """Checks whether a given R package is installed"""
    from rpy2.robjects.packages import isinstalled

    if not isinstalled(package_name):
        raise ImportError(f"Please install {package_name} R package.")


@rpy2_check
def r_set_seed(seed):
    """Set the seed of R random number generator"""
    from rpy2.robjects import r

    set_seed = r("set.seed")
    set_seed(seed)


@rpy2_check
def r_set_logger_level(level):
    """Set the logger level of rpy2"""
    import rpy2.rinterface_lib.callbacks

    rpy2.rinterface_lib.callbacks.logger.setLevel(level)


@rpy2_check
@anndata2ri_check
def py2r(x, name=None, on_disk=None, verbose=0):
    """Convert a Python object to an R object using rpy2"""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    import anndata2ri
    import time

    if not name:
        name = ""
    objType = type(x)

    if on_disk == None:
        on_disk = True if py2r_disk(x, check=True) else False
    if verbose:
        print(
            f"on disk mode: {on_disk}, transfer `{objType}` to R: {name} start.", end=""
        )
    timeStart = time.time()

    if on_disk:
        x = py2r_disk(x)

    else:
        if sp.issparse(x):
            # workaround for: https://github.com/theislab/anndata2ri/issues/47
            x = anndata2ri.scipy2ri.py2rpy(x)

        with localconverter(
            ro.default_converter
            + numpy2ri.converter
            + pandas2ri.converter
            + anndata2ri.converter
        ):
            x = ro.conversion.py2rpy(x)

    timeEnd = time.time()
    timePass = timeEnd - timeStart
    if verbose:
        print(
            "\r"
            + f"on disk mode: {on_disk}, transfer `{objType}` to R: {name} End. Elapsed time: {timePass:.0f}",
            flush=True,
        )
    return x


def py2r_disk(obj, check=False, *args, **kwargs):
    """Convert a Python object to R on disk"""
    from tempfile import NamedTemporaryFile
    import scanpy as sc

    def _adata(obj, X_layer="X"):
        zellkonverter = importr("zellkonverter")
        sce = importr("SingleCellExperiment")
        tpFile = NamedTemporaryFile(suffix=".h5ad")
        obj.var["temp_featureName"] = obj.var.index
        obj.obs["temp_barcodeName"] = obj.obs.index
        obj.write_h5ad(tpFile.name)
        objR = zellkonverter.readH5AD(tpFile.name, X_layer, reader="R")
        dfR_obs = py2r(obj.obs)
        dfR_var = py2r(obj.var)
        with ro.local_context() as rlc:
            rlc["objR"] = objR
            rlc["dfR_obs"] = dfR_obs
            rlc["dfR_var"] = dfR_var
            R(
                """
            objR@rowRanges@partitioning@NAMES <- rowData(objR)$temp_featureName
            objR@colData@rownames <- colData(objR)$temp_barcodeName
            objR@colData <- dfR_obs %>% DataFrame
            objR@rowRanges@elementMetadata <- dfR_var %>% DataFrame
            """
            )
            objR = R("objR")

        tpFile.close()
        return objR

    def _dataframe(obj):
        arrow = importr("arrow")
        tpFile = NamedTemporaryFile(suffix=".feather")
        obj = obj.rename(columns=str)

        #  bypass error: `Object of type bool_ is not JSON serializable`
        for colName, colType in obj.dtypes.items():
            if isinstance(colType, pd.CategoricalDtype):
                ls_category = colType.categories
                obj[colName] = obj[colName].astype('object').astype('category').cat.set_categories(ls_category)
        
        if (obj.index == obj.reset_index().index).all():
            obj.to_feather(tpFile.name)
            needSetIndex = False
        else:
            obj.rename_axis("_index_py2r_").reset_index().to_feather(tpFile.name)
            needSetIndex = True

        dfR = arrow.read_feather(tpFile.name, as_data_frame=True)
        dfR = rBase.as_data_frame(dfR)
        if needSetIndex:
            with ro.local_context() as rlc:
                rlc["dfR"] = dfR
                R(
                    """
                srR_index <- dfR$`_index_py2r_`
                dfR$`_index_py2r_` <- NULL
                rownames(dfR) <- srR_index
                """
                )
                dfR = rlc["dfR"]
        return dfR

    def _array(obj):
        obj = pd.DataFrame(obj)
        obj = obj.rename(columns=str)
        dfR = py2r(obj)
        arR = rBase.as_matrix(dfR)
        return arR

    dt_config = {sc.AnnData: _adata, pd.DataFrame: _dataframe, np.ndarray: _array}
    if check:
        for _class in dt_config.keys():
            if isinstance(obj, _class):
                if _class == np.ndarray:  # _array only worked for 2D arrays
                    if len(obj.shape) == 2:
                        return True
                    else:
                        return False
                else:
                    return True
        else:
            return False
        # if type(obj) in dt_config:
        #     if type(obj) == np.asaarray:
        #         if len(obj.shape) == 2: # _array only worked for 2D arrays
        #             return True
        #         else:
        #             return False
        #     else:
        #         return True
        # else:
        #     return False
    for _class in dt_config.keys():
        if isinstance(obj, _class):
            _type = _class
            break
    func = dt_config[_type]
    objR = func(obj, *args, **kwargs)
    return objR


@rcontext
def ad2so(
    ad,
    layer="raw",
    dataLayer=None,
    scaleLayer=None,
    scaleLayerInObsm=False,
    assay="RNA",
    rEnv=None,
    verbose=0,
    **kwargs,
):
    '''`ad2so` converts an AnnData object to a SeuratObject object
    
    Parameters
    ----------
    ad
        AnnData object
    layer, optional
        the layer to use for the count slot.
    dataLayer
        the name of the layer to use for the data slot.
    scaleLayer
        the name of the layer to use for scaleData slot.
    scaleLayerInObsm, optional
        if True, then the scaleLayer is assumed to be in the obsm of the adata object.
    assay, optional
        The assay to use.
    rEnv
        R environment to use. If None, then a new one is created.
    verbose, optional
        0, 1, 2, 3, 4
    '''
    import scipy.sparse as ss

    importr("Seurat")
    R = ro.r
    mt_count = ad.layers[layer]
    if ad.var.empty:
        ad.var["project_ad2so"] = "temp"
    if ad.obs.empty:
        ad.obs["project_ad2so"] = "temp"
    rEnv["mtR_count"] = py2r(mt_count.T)
    rEnv["arR_obsName"] = R.unlist(R.c(ad.obs.index.to_list()))
    rEnv["arR_varName"] = R.unlist(R.c(ad.var.index.to_list()))
    rEnv["assay"] = assay

    R(
        """
    colnames(mtR_count) <- arR_obsName
    rownames(mtR_count) <- arR_varName
    so <- CreateSeuratObject(mtR_count, assay=assay)
    """
    )
    if "highly_variable" in ad.var.columns:
        ls_hvgGene = (
            ad.var["highly_variable"]
            .loc[lambda x: x]
            .index.str.replace("_", "-")
            .to_list()
        )
        rEnv["arR_hvgGene"] = R.unlist(R.c(ls_hvgGene))
        R(
            """
        VariableFeatures(so) <- arR_hvgGene
        """
        )
    rEnv["dfR_obs"] = py2r(ad.obs, verbose=verbose)
    rEnv["dfR_var"] = py2r(ad.var, verbose=verbose)
    R(
        """
    so <- AddMetaData(so, dfR_obs)
    so[['RNA']] <- AddMetaData(so[[assay]], dfR_var)
    """
    )

    if dataLayer is None:
        R(
            "NormalizeData(so, normalization.method = 'LogNormalize', scale.factor = 10000, verbose = F)"
        )
    else:
        mt_data = ad.layers[dataLayer]
        rEnv["mtR_data"] = py2r(mt_data.T, verbose=verbose)
        R(
            """
        colnames(mtR_data) <- arR_obsName
        rownames(mtR_data) <- arR_varName
        so <- SetAssayData(so, slot = "data",mtR_data, assay = assay)
        """
        )

    if scaleLayer is None:
        pass
    else:
        if not scaleLayerInObsm:
            mt_scaleData = ad.layers[scaleLayer]
            rEnv["mtR_scaleData"] = py2r(mt_scaleData.T, verbose=verbose)
            R(
                """
            colnames(mtR_scaleData) <- arR_obsName
            rownames(mtR_scaleData) <- arR_varName
            so <- SetAssayData(so, slot = "scale.data", mtR_scaleData, assay = assay)
            """
            )
        else:
            rEnv["dfR_scaleData"] = py2r(
                ad.obsm[scaleLayer].loc[:, lambda df: df.columns.isin(ad.var.index)].T,
                verbose=verbose,
            )
            R(
                """
            mtR_scaleData <- dfR_scaleData %>% as.matrix
            so <- SetAssayData(so, slot = "scale.data", mtR_scaleData, assay = assay)
            """
            )
    ls_obsm = [x for x in ad.obsm.keys() if x.startswith("X_")]
    for obsm in ls_obsm:
        obsm_ = obsm.split("X_", 1)[1]
        df_obsm = pd.DataFrame(
            ad.obsm[obsm],
            index=ad.obs.index,
            columns=[f"{obsm_}_{x}" for x in range(1, 1 + ad.obsm[obsm].shape[1])],
        )
        rEnv["dfR_obsm"] = py2r(df_obsm, verbose=verbose)
        rEnv["obsm"] = obsm_
        R(
            """
        mtR_obsm <- dfR_obsm %>% as.matrix
        so[[obsm]] <- CreateDimReducObject(mtR_obsm, assay=assay, key=paste0(obsm, '_'))
        """
        )

    for obsp in ad.obsp.keys():
        rEnv["mtR_obsp"] = py2r(ss.csc_matrix(ad.obsp[obsp]), verbose=verbose)
        rEnv["obsp"] = obsp
        R(
            """
        colnames(mtR_obsp) <- arR_obsName
        rownames(mtR_obsp) <- arR_obsName
        so[[obsp]] <- as.Graph(x = mtR_obsp)
        """
        )
    return rEnv["so"]


@rpy2_check
@anndata2ri_check
def r2py(x, name=None, verbose=0):
    """Convert an rpy2 (R)  object to a Python object"""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    import anndata2ri
    import time
    from tempfile import NamedTemporaryFile

    def _dataframe(objR):
        tpFile = NamedTemporaryFile(suffix=".feather")
        with ro.local_context() as rlc:
            rlc["objR"] = objR
            rlc["filePath"] = tpFile.name
            R(
                """
            library('arrow')
            objR$`index_r2py` <- rownames(objR)
            rownames(objR) <- NULL
            write_feather(objR, filePath)
            """
            )
        obj = pd.read_feather(tpFile.name)
        obj = obj.set_index("index_r2py").rename_axis(None)
        return obj

    if not name:
        name = ""

    try:
        objType = list(x.rclass)[0]
    except:
        objType = "unknown type"
    if verbose:
        print(f"transfer `{objType}` to python: {name} start", end="")
    timeStart = time.time()
    if ro.r("class")(x)[0] == "data.frame":
        x = _dataframe(x)
    else:
        try:
            with localconverter(
                ro.default_converter
                + numpy2ri.converter
                + pandas2ri.converter
                + anndata2ri.scipy2ri.converter
                + anndata2ri.converter
            ):
                x = ro.conversion.rpy2py(x)

        except TypeError:
            # workaround for: https://github.com/theislab/anndata2ri/issues/47
            x = anndata2ri.scipy2ri.rpy2py(x)
    timeEnd = time.time()
    timePass = timeEnd - timeStart
    if verbose:
        print(
            "\r"
            + f"transfer `{objType}` to python: {name} End. Elapsed time: {timePass:.0f}",
            flush=True,
        )
    return x


@rcontext
def so2ad(so, assay=None, verbose=0, rEnv=None, skipScaleMtx=False, **kwargs):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    import scipy.sparse as ss

    importr("Seurat")
    R = ro.r
    rEnv["so"] = so
    if assay is None:
        assay = R("DefaultAssay(so)")[0]
    R(f"dfR_var <- so${assay}[[]] %>% as.data.frame")
    R("dfR_obs <- so[[]] %>% as.data.frame")
    df_obs = r2py(R("dfR_obs"), verbose=verbose)
    df_var = r2py(R("dfR_var"), verbose=verbose)
    ad = sc.AnnData(
        ss.csc_matrix((df_obs.shape[0], df_var.shape[0])), obs=df_obs, var=df_var
    )

    for slot in ["counts", "data"]:
        ar_mtx = r2py(
            R(f"""GetAssayData(object=so, assay='{assay}', slot='{slot}')"""),
            verbose=verbose,
        ).T
        if ar_mtx.shape == (0, 0):
            pass
        else:
            ad.layers[f"{assay}_{slot}"] = ar_mtx
        # ad.layers[f"{assay}_{slot}"] = r2py(
        #     R(f"""GetAssayData(object=so, assay='{assay}', slot='{slot}')"""), verbose=verbose
        # ).T
    if not skipScaleMtx:
        df_scale = r2py(
            R(
                f"""GetAssayData(object=so, assay='{assay}', slot='scale.data') %>% as.data.frame"""
            ),
            verbose=verbose,
        ).T
        if df_scale.empty:
            pass
        elif (df_scale.shape == (1, 1)) & (
            (df_scale.iat[0, 0] == -2147483648) | (df_scale.iat[0, 0] is None)
        ):
            pass
        else:
            ad.obsm[f"{assay}_scale.data"] = df_scale

    if R("names(so@reductions)") is R("NULL"):
        pass
    else:
        for obsm in R("names(so@reductions)"):
            usedAssay = R(f"so@reductions${obsm}@assay.used")[0]
            ad.obsm[f"X_{obsm}_{usedAssay}"] = r2py(
                R(f'Embeddings(object = so, reduction = "{obsm}")'), verbose=verbose
            )
            if usedAssay == assay:
                ad.obsm[f"X_{obsm}"] = r2py(
                    R(f'Embeddings(object = so, reduction = "{obsm}")'), verbose=verbose
                )
    if R("names(so@graphs)") is R("NULL"):
        pass
    else:
        for obsp in R("names(so@graphs)"):
            ad.obsp[obsp] = r2py(R(f"so@graphs${obsp} %>% as.sparse"), verbose=verbose)
    
    # add hvg info
    haveVarGene = R("VariableFeatures")(so, assay=assay) >> F(R("length")) >> F(R("\(x) {x>0}")) >> F(lambda x:x[0]) 
    if haveVarGene:
        ls_var =  R("VariableFeatures")(so, assay=assay) >> F(list)
        logger.info(f"assay {assay}: Variable features are already calculated. {len(ls_var)} features are selected.")
        dt_var = {ls_var[i]:i for i in range(len(ls_var))}
        ad.var['highly_variable'] = ad.var.index.isin(dt_var)
        ad.var['highly_variable_rank'] = ad.var.index.map(lambda x:dt_var.get(x, np.nan))
    else:
        logger.info(f"assay {assay}: No variable features are calculated. Pass")
    return ad


@rcontext
def so2md(so, rEnv=None, verbose=0, **kwargs):
    import scipy.sparse as ss
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    importr("Seurat")
    R = ro.r
    rEnv["so"] = so
    assays = list(R("names(so@assays)"))
    dtAd = {}
    for assay in assays:
        ad = so2ad(so, assay=assay, rEnv=rEnv, verbose=verbose)
        dtAd[assay] = ad
        if f"{assay}_scale.data" in ad.obsm:
            df_layer = ad.obsm[f"{assay}_scale.data"]
            del ad.obsm[f"{assay}_scale.data"]
            ad_scale = sc.AnnData(df_layer)
            ad_scale.layers[f"{assay}_scale.data"] = ad_scale.X
            ad_scale.X = ss.csc_matrix(ad_scale.shape)
            if "highly_variable" in ad.var:
                ad_scale.var["highly_variable"] = ad.var["highly_variable"]
                ad_scale.var["highly_variable_rank"] = ad.var["highly_variable_rank"]
            dtAd[f"{assay}_scale.data"] = ad_scale
    md = mu.MuData(dtAd, axis=-1)
    return md


@contextmanager
def r_inline_plot(width=None, height=None, res=None):
    dt_params = dict(width=width, height=height, res=res)
    dt_params = {x: y for x, y in dt_params.items() if y}
    with grdevices.render_to_bytesio(grdevices.png, **dt_params) as b:
        yield
    data = b.getvalue()
    display(Image(data=data, format="png", embed=True))


def rHelp(x: str):
    import rpy2.robjects as ro

    R = ro.r
    with Capturing() as output:
        str(R.help(x))
    print("\n".join(output[1::2]))


def trl(objR, name=None, prefix="trl", verbose=1):
    "return an un-evaluated R object. More details in https://github.com/rpy2/rpy2/issues/815"
    rEnv = evaluation_context.get()
    if name is None:
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r"\btrl\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
            if m:
                name = m.group(1)
                break
        else:
            assert False, "Can not identify argument"
    name = f"{prefix}_{name}"
    rEnv[name] = objR
    ro.r.gc()
    if verbose > 0:
        print(name)
    return rl(name)


class Trl:
    def __init__(self, objName="T", verbose=1):
        self.objName = objName
        self.verbose = verbose

    def __ror__(self, objR):
        objName = self.objName
        objName = objName[::-1]
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            line = line[::-1]
            m = re.search(rf"\b{objName}\s*\|\s*(\w+)[\s\W]*", line)
            if m:
                name = m.group(1)[::-1]
                break
        else:
            assert False, "Can not identify argument"
        return trl(objR, name, verbose=self.verbose)


def rGet(objR, *attrs):
    import rpy2.robjects as ro
    _ = objR
    for attr in attrs:
        if attr[0] in ["@", "$"]:
            cat = attr[0]
            attr = attr[1:]
            cat = {"@": "slots", "$": "rx2"}[cat]
            _ = eval(f"_.{cat}['{attr}']")
        else:
            if isinstance(_, ro.methods.RS4):
                _ = _.slots[attr]
            else:
                _ = _.rx2[attr]
    return _


def rSet(objR, targetObjR, *attrs):
    import rpy2.robjects as ro

    _ = objR
    for attr in attrs[:-1]:
        if attr[0] in ["@", "$"]:
            cat = attr[0]
            attr = attr[1:]
            cat = {"@": "slots", "$": "rx2"}[cat]
            _ = eval(f"_.{cat}['{attr}']")
        else:
            if isinstance(_, ro.methods.RS4):
                _ = _.slots[attr]
            else:
                _ = _.rx2[attr]

    attr = attrs[-1]
    if attr[0] in ["@", "$"]:
        cat = attr[0]
        attr = attr[1:]
        cat = {"@": "slots", "$": "rx2"}[cat]
        if cat == "slots":
            _.slots[attr] = targetObjR
        elif cat == "rx2":
            _.rx2[attr] = targetObjR
    else:
        if isinstance(_, ro.methods.RS4):
            _.slots[attr] = targetObjR
        else:
            _.rx2[attr] = targetObjR


def py2r_re(obj):
    R = ro.r
    with TemporaryDirectory() as dir_tmp:
        fileName = dir_tmp + "/tmp"
        with open(fileName, "wb") as fh:
            pickle.dump(obj, fh)
        objR = R(f"reticulate::py_load_object('{fileName}')")
    return objR


def r2py_re(objR):
    R = ro.r
    rEnv = evaluation_context.get()
    rEnv["temp_r2py"] = objR
    R("temp_r2py <- reticulate::r_to_py(temp_r2py)")
    with TemporaryDirectory() as dir_tmp:
        fileName = dir_tmp + "/tmp"
        R(f"reticulate::py_save_object(temp_r2py, '{fileName}')")
        with open(fileName, "rb") as fh:
            obj = pickle.load(fh)
    del rEnv["temp_r2py"]
    return obj

def RR(script):
    R = ro.r
    return R(f"\(x) {script}")

def FR(script):
    R = ro.r
    return F(R(script))
