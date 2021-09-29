"""
recipe
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
from . import basic
from . import normalize

def sct(ad):
    """
    sct + pca + neighbors + umap

    `.X` must be raw data

    ----------
    `ad` will be updated.
    ad.X will be `normalize_log` layer
    """
    basic.testAllCountIsInt(ad, None)
    ad.layers['raw'] = ad.X.copy()
    ad.layers['normalize_log'] = ad.layers['raw'].copy()
    sc.pp.normalize_total(ad, 1e4, layer='normalize_log')
    sc.pp.log1p(ad, layer='normalize_log')
    normalize.normalizeBySCT(ad, layer='raw', min_cells=10, log_scale_correct=True)
    ad.X = ad.layers['sct_residuals'].copy()
    sc.tl.pca(ad)
    sc.pp.neighbors(ad, n_pcs=50)
    sc.tl.umap(ad)
    ad.X = ad.layers['normalize_log'].copy()
    sc.pl.umap(ad, color=['n_genes', 'n_counts'], cmap='Reds')

