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


def starsolo_transferMtxToH5ad(starsoloMtxDir, force=False) -> sc.AnnData:
    import scipy.sparse as ss
    import glob
    import os

    starsoloMtxDir = starsoloMtxDir.rstrip("/") + "/"
    path_out = f"{starsoloMtxDir}adata.h5ad"
    if (os.path.exists(path_out)) and (not force):
        adata = sc.read_h5ad(path_out)
    else:
        ls_allMtx = glob.glob(f"{starsoloMtxDir}*.mtx")
        ls_mtxName = [x.split("/")[-1].split(".")[0] for x in ls_allMtx]
        path_barcode = f"{starsoloMtxDir}barcodes.tsv"
        path_feature = f"{starsoloMtxDir}features.tsv"
        df_barcode = pd.read_table(path_barcode, names=["barcodes"]).set_index(
            "barcodes"
        )
        df_feature = pd.read_table(
            path_feature, names=["geneid", "symbol", "category"]
        ).set_index("geneid")
        
        adata = sc.AnnData(
            X=ss.csr_matrix((len(df_barcode), len(df_feature))),
            obs=df_barcode,
            var=df_feature,
        )
        for mtxName, mtxPath in zip(ls_mtxName, ls_allMtx):
            adata.layers[mtxName] = sc.read_mtx(mtxPath).X.T
            logger.info(f"read {mtxName} done")
        adata.write_h5ad(path_out)
    return adata