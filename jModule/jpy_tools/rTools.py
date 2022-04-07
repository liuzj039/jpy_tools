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
from .otherTools import Capturing
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scanpy as sc
import h5py
from tempfile import TemporaryDirectory
import sys

R = ro.r
seo = importr("SeuratObject")
rBase = importr("base")
rUtils = importr("utils")
# arrow = importr('arrow') # this package should be imported before pyarrow

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
def py2r(x, name=None, on_disk=None):
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

    print(f"on disk mode: {on_disk}, transfer `{objType}` to R: {name} start.", end="")
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
        with ro.local_context() as rlc:
            rlc["objR"] = objR
            R(
                """
            objR@rowRanges@partitioning@NAMES <- rowData(objR)$temp_featureName
            objR@colData@rownames <- colData(objR)$temp_barcodeName
            """
            )
            objR = R("objR")

        tpFile.close()
        return objR
    
    def _dataframe(obj):
        arrow = importr('arrow')
        tpFile = NamedTemporaryFile(suffix=".feather")
        obj = obj.rename(columns=str)
        if (obj.index == obj.reset_index().index).all():
            obj.to_feather(tpFile.name)
            needSetIndex = False
        else:
            obj.rename_axis('_index_py2r_').reset_index().to_feather(tpFile.name)
            needSetIndex = True

        dfR = arrow.read_feather(tpFile.name, as_data_frame = True)
        dfR = rBase.as_data_frame(dfR)
        if needSetIndex:
            with ro.local_context() as rlc:
                rlc["dfR"] = dfR
                R("""
                srR_index <- dfR$`_index_py2r_`
                dfR$`_index_py2r_` <- NULL
                rownames(dfR) <- srR_index
                """)
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


def ad2so(
    ad,
    layer="raw",
    ls_obs=None,
    ls_var=None,
    ls_obsm=None,
    dir_tmp=None,
    dataLayer=None,
    scaleLayer=None,
    lightMode=False,
    renv = None,
    path_R=None,
    libPath_R =None,
    verbose=0
):
    """
    anndata to seuratObject.

    obsm:
        matrix which's name does not start with 'X_' will be discarded.
    uns:
        discarded
    obsp:
        only keep 'distances'
    varp:
        discarded
    lightMode:
        if True, obsm/varm/obsp info will be discarded.

    layer must be raw.
    """
    import sh
    import scipy.sparse as ss
    if not path_R:
        path_R = settings.seuratDisk_rPath
    if not libPath_R:
        libPath_R = settings.seuratDisk_rLibPath

    # ad = ad.copy() # workaround `memoory not mapped` error
    # R('.libPaths')(libPath_R)
    # seuratDisk = importr("SeuratDisk")
    if renv is None:
        renv = ro.Environment()

    if not dir_tmp:
        dir_tmp_ = TemporaryDirectory()
        dir_tmp = dir_tmp_.name
    path_h5ad = f"{dir_tmp}/temp.h5ad"
    path_h5so = f"{dir_tmp}/temp.h5seurat"
    path_rds = f"{dir_tmp}/temp.rds"
    if lightMode:
        ad_partial = sc.AnnData(
            ad.layers[layer].copy(),
            obs=ad.obs,
            var=ad.var,
        )
    else:
        ad_partial = sc.AnnData(
            ad.layers[layer].copy(),
            obs=ad.obs,
            var=ad.var,
            obsm=ad.obsm,
            varm=ad.varm,
            obsp=ad.obsp,
        )
    if not ls_obs is None:
        if isinstance(ls_obs, str):
            ls_obs = [ls_obs]
        ad_partial.obs = ad_partial.obs[ls_obs]
    if not ls_var is None:
        if isinstance(ls_var, str):
            ls_var = [ls_var]
        ad_partial.var = ad_partial.var[ls_var]
    if not ls_obsm is None:
        if isinstance(ls_obsm, str):
            ls_obsm = [ls_obsm]
        ls_rm = []
        for _obsm in ad_partial.obsm.keys():
            if not _obsm in ls_obsm:
                ls_rm.append(_obsm)
        for _obsm in ls_rm:
            del(ad_partial.obsm[_obsm])

    ad_partial.X = ss.csr_matrix(
        ad_partial.X
    )  # workaround https://github.com/satijalab/seurat/issues/2374
    _ls = []
    for key in ad_partial.obsm:
        if not key.startswith("X_"):
            _ls.append(key)
    for key in _ls:
        del ad_partial.obsm[key]
    
    # workaround `memoory not mapped` error
    df_var = ad_partial.var
    ad_partial.var = ad_partial.var[[]]
    ad_partial.raw = ad_partial
    ad_partial.var = df_var
    

    sc.pp.normalize_total(ad_partial, 1e4)
    sc.pp.log1p(ad_partial)

    ad_partial.write(path_h5ad)

    h5 = h5py.File(path_h5ad, "r+")
    if "obsp/distances" in h5:
        h5["/uns/neighbors/distances"] = h5["/obsp/distances"]
        h5["/uns/neighbors/params/method"] = "nn"

    h5.close()

    # seuratDisk.Convert(path_h5ad, dest="h5Seurat", overwrite=True)
    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); Convert('{path_h5ad}', dest='h5Seurat', overwrite=T); so <- LoadH5Seurat('{path_h5so}'); saveRDS(so, '{path_rds}')",
    ]
    if verbose:
        cmd = sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        cmd = sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    # for x in sh.Command(path_R)(*ls_cmd, _err_to_out=True, _iter=True):
    #     print(x.rstrip())
    so = R.readRDS(path_rds)

    if dataLayer:
        with ro.local_context(renv) as rlc:
            _ad = sc.AnnData(X=ss.csr_matrix(ad.shape), obs=ad.obs[[]], var=ad.var[[]])
            _ad.layers["data"] = ad.layers[dataLayer].copy()
            _se = py2r(_ad)
            rlc["se"] = _se
            rlc["so"] = so
            R(
                """so <- SetAssayData(so, slot = 'data', new.data = assay(se, 'data')) """
            )
            so = rlc["so"]
    if scaleLayer:
        with ro.local_context(renv) as rlc:
            _ad = sc.AnnData(X=ss.csr_matrix(ad.shape), obs=ad.obs[[]], var=ad.var[[]])
            _ad.layers["scale.data"] = ad.layers[scaleLayer].copy()
            _se = py2r(_ad)
            rlc["se"] = _se
            rlc["so"] = so
            R(
                """so <- SetAssayData(so, slot = 'scale.data', new.data = assay(se, 'scale.data')) """
            )
            so = rlc["so"]

    return so


@rpy2_check
@anndata2ri_check
def r2py(x, name=None):
    """Convert an rpy2 (R)  object to a Python object"""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    import anndata2ri
    import time

    if not name:
        name = ""

    try:
        objType = list(x.rclass)[0]
    except:
        objType = "unknown type"

    print(f"transfer `{objType}` to python: {name} start", end="")
    timeStart = time.time()
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
    print(
        "\r"
        + f"transfer `{objType}` to python: {name} End. Elapsed time: {timePass:.0f}",
        flush=True,
    )
    return x


def so2ad(so, dir_tmp=None, libPath_R = None, path_R=None, obsReParse = True, verbose = True) -> sc.AnnData:
    import sh
    if not libPath_R:
        libPath_R = settings.seuratDisk_rLibPath
    if not path_R:
        path_R = settings.seuratDisk_rPath
    # R('.libPaths')(libPath_R)
    # seuratDisk = importr("SeuratDisk")
    if not dir_tmp:
        dir_tmp_ = TemporaryDirectory()
        dir_tmp = dir_tmp_.name
    path_h5ad = f"{dir_tmp}/temp.h5ad"
    path_h5so = f"{dir_tmp}/temp.h5seurat"
    path_rds = f"{dir_tmp}/temp.rds"

    R.saveRDS(so, path_rds)

    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); so <- readRDS('{path_rds}'); SaveH5Seurat(so, '{path_h5so}')",
    ]
    if verbose:
        sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    # seuratDisk.SaveH5Seurat(so, path_h5so, overwrite=True)

    h5so = h5py.File(path_h5so, "r+")
    ls_assays = h5so["/assays"].keys()
    for assay in ls_assays:
        ls_keys = h5so[f"/assays/{assay}"].keys()
        ls_slots = [x for x in ls_keys if x in ["counts", "data", "scale.data"]]
        ls_slots = [x for x in h5so[f"/assays/{assay}"] if x in ls_slots]
        for slot in ls_slots:
            if slot != "scale.data":
                h5so[f"/assays/{assay}_{slot}/data"] = h5so[f"/assays/{assay}/{slot}"]
                h5so[f"/assays/{assay}_{slot}/features"] = h5so[
                    f"/assays/{assay}/features"
                ]
                # h5so[f"/assays/{assay}_{slot}/misc"] = h5so[f"/assays/{assay}/misc"]
            else:
                h5so[f"/assays/{assay}_{slot}/scale.data"] = h5so[
                    f"/assays/{assay}/{slot}"
                ]
                h5so[f"/assays/{assay}_{slot}/data"] = h5so[f"/assays/{assay}/{slot}"]
                h5so[f"/assays/{assay}_{slot}/features"] = h5so[
                    f"/assays/{assay}/features"
                ]
                # h5so[f"/assays/{assay}_{slot}/misc"] = h5so[f"/assays/{assay}/misc"]
                h5so[f"/assays/{assay}_{slot}/scaled.features"] = h5so[
                    f"/assays/{assay}/scaled.features"
                ]
    h5so.close()

    # seuratDisk.Convert(path_h5so, dest="h5ad", overwrite=True)
    ls_cmd = [
        "-q",
        "-e",
        f".libPaths('{libPath_R}'); library(SeuratDisk); Convert('{path_h5so}', dest='h5ad', overwrite=T)",
    ]
    if verbose:
        sh.Command(path_R)(*ls_cmd, _err=sys.stderr, _out=sys.stdout)
    else:
        sh.Command(path_R)(*ls_cmd, _err_to_out=True)
    with h5py.File(path_h5ad, "a") as h5ad:
        if "raw" in h5ad.keys():
            del h5ad["raw"]
    ad = sc.read_h5ad(path_h5ad)
    if obsReParse:
        df_obs = r2py(so.slots['meta.data'])
        df_obs = df_obs.combine_first(ad.obs)
        ad.obs = df_obs.copy()
    return ad


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


def trl(objR):
    "transfer objR to an unevaluated R language object"
    from rpy2.robjects import rl
    import rpy2.robjects as ro
    import random
    import string

    def ranstr(num):
        salt = "".join(random.sample(string.ascii_letters, num)) + "".join(
            random.sample(string.digits, 5)
        )
        return salt

    tempName = ranstr(40)
    ro.globalenv[tempName] = objR
    return rl(tempName)


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
