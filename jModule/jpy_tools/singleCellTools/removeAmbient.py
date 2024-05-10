from logging import log
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import patchworklib as pw
import matplotlib as mpl
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from io import StringIO
from concurrent.futures import ProcessPoolExecutor as Mtp
from concurrent.futures import ThreadPoolExecutor as MtT
from joblib import Parallel, delayed
import sh
import h5py
from tqdm import tqdm
import scvi
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
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from . import basic, diffxpy
from ..rTools import rcontext, r_inline_plot, py2r, r2py
from ..otherTools import F, pwStack, pwShow, MuteInfo
    
rBase = importr("base")
rUtils = importr('utils')

@rcontext
def removeAmbientBySoupx(ad:sc.AnnData, ad_raw:sc.AnnData, layerAd:str='raw', layerRaw:str='raw', res=1, correctedLayerName:str='soupX_corrected', forceAccept=True, rEnv=None):
    '''`removeAmbientBySoupx` removes the ambient signal from the data by using the soupX algorithm
    
    Parameters
    ----------
    ad : sc.AnnData
        the AnnData object that contains the data you want to correct
    ad_raw : sc.AnnData
        the raw data, which is used to calculate the ambient
    layerAd : str, optional
        the layer in ad that you want to remove the ambient from
    layerRaw : str, optional
        the layer in ad_raw that contains the raw counts
    correctedLayerName : str, optional
        the name of the layer that will be created in the ad object.
    rEnv
        R environment to use. If None, will create a new one.
    
    '''
    SoupX = importr('SoupX')
    R = ro.r
    assert (ad.var.index == ad_raw.var.index).all(), 'ad.var.index != ad_raw.var.index'
    
    ad_pp = ad.copy()
    ad_pp.X = ad_pp.layers[layerAd]
    sc.pp.highly_variable_genes(ad_pp, n_top_genes=2000, flavor='seurat_v3')
    sc.pp.normalize_per_cell(ad_pp)
    sc.pp.log1p(ad_pp)
    sc.pp.pca(ad_pp)
    sc.pp.neighbors(ad_pp)
    sc.tl.leiden(ad_pp, key_added="soupx_groups", resolution=res)
    soupx_groups = ad_pp.obs["soupx_groups"]
    del ad_pp

    cells = ad.obs_names
    genes = ad.var_names
    data = ad.layers[layerAd].T
    data_tod = ad_raw.layers[layerRaw].T

    data = py2r(data)
    data_tod = py2r(data_tod)
    cells = py2r(cells)
    genes = py2r(genes)

    rEnv['data'] = data
    rEnv['data_tod'] = data_tod
    rEnv['cells'] = cells
    rEnv['genes'] = genes
    rEnv['soupx_groups'] = R.c(**soupx_groups)
    rEnv['forceAccept'] = forceAccept

    R("""
    rownames(data) = genes
    colnames(data) = cells
    # ensure correct sparse format for table of counts and table of droplets
    data <- as(data, "sparseMatrix")
    data_tod <- as(data_tod, "sparseMatrix")

    # Generate SoupChannel Object for SoupX 
    sc = SoupChannel(data_tod, data, calcSoupProfile = FALSE)

    # Add extra meta data to the SoupChannel object
    soupProf = data.frame(row.names = rownames(data), est = rowSums(data)/sum(data), counts = rowSums(data))
    sc = setSoupProfile(sc, soupProf)
    # Set cluster information in SoupChannel
    sc = setClusters(sc, soupx_groups)
    """)

    with r_inline_plot():
        R("""
        sc  = autoEstCont(sc, doPlot=T, forceAccept=forceAccept)
        out = adjustCounts(sc, roundToInt = TRUE)
        """)

    ad.layers[correctedLayerName] = r2py(rEnv['out']).T

    ad.obs['ambientRnaFractionEstimatedBySoupx'] = 1 - (ad.layers[correctedLayerName].sum(1).A.reshape(-1) / ad.layers['raw'].sum(1).A.reshape(-1))