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

def loadBGI(path_gem, binSize) -> sc.AnnData:
    df_gem = pd.read_table(path_gem, comment='#')
    df_gem = df_gem.pipe(
        lambda df: df.assign(
            xBin=((df["x"] // binSize) * binSize + binSize / 2).astype(int),
            yBin=((df["y"] // binSize) * binSize + binSize / 2).astype(int),
        )
    ).assign(nameBin=lambda df: df["xBin"].astype(str) + "_" + df["yBin"].astype(str))
    df_mtxLong = df_gem.groupby(["nameBin", "geneID"])["MIDCount"].agg("sum").reset_index()
    ar_mtx = np.zeros(
        (df_mtxLong["nameBin"].unique().shape[0], df_mtxLong["geneID"].unique().shape[0])
    )
    ls_obs = df_mtxLong['nameBin'].unique().tolist()
    _dt = {x:i for i,x in enumerate(ls_obs)}

    df_mtxLong['nameBin_index'] = df_mtxLong['nameBin'].map(_dt)

    ls_var = df_mtxLong['geneID'].unique().tolist()
    _dt = {x:i for i,x in enumerate(ls_var)}

    df_mtxLong['geneID_index'] = df_mtxLong['geneID'].map(_dt)
    for tp_line in tqdm(df_mtxLong.itertuples(), total=len(df_mtxLong)):
        ar_mtx[tp_line.nameBin_index, tp_line.geneID_index] += tp_line.MIDCount
    ad = sc.AnnData(ar_mtx, obs=pd.DataFrame(index=ls_obs), var=pd.DataFrame(index=ls_var))
    df_spatial = pd.DataFrame(
        [ad.obs.index.str.split("_").str[0], ad.obs.index.str.split("_").str[1]],
        columns=ad.obs.index,
        index=["imagecol", "imagerow"],
    ).T.astype(int)
    ad = st.create_stlearn(ad.to_df(), df_spatial, 'bgi')
    ad.X = ss.csr_matrix(ad.X)
    return ad