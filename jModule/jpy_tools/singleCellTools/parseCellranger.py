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

def extractReadCountsByUmiFromTenX(
    molInfoPath: str, kitVersion: Literal["v2", "v3"] = "v2", filtered: bool = True
) -> pd.DataFrame:
    """
    parse molInfo_info.h5

    Parameters
    ----------
    molInfoPath : strkitVersion, optional
        molecule_info.h5, by default "v2"
    Kitversion : 'v2' or 'v3'
        v2: umi 10bp; v3: umi 12bp
    filtered : bool, optional
        only use these cells which pass the filter, by default True

    Returns
    -------
    pd.DataFrame
        columns : ["barcodeUmi", "featureName", "readCount"]
    """
    umiLength = {"v2": 10, "v3": 12}[kitVersion]

    def NumToSeq():
        nonlocal umiLength

        numToBase = {"00": "A", "01": "C", "10": "G", "11": "T"}

        def _numToSeq(num):
            num = int(num)
            numStr = f"{num:032b}"[-umiLength * 2 :]
            return "".join(
                [numToBase[numStr[2 * x : 2 * x + 2]] for x in range(umiLength)]
            )

        return _numToSeq

    logger.warning(
        "It consumes a lot of memory when processing files exceeding 1Gb."
    )
    numToSeq = NumToSeq()
    molInfo = h5py.File(molInfoPath, "r")
    allBarcodeAr = molInfo["barcodes"][()].astype(str)
    useBarcodeAr = allBarcodeAr[molInfo["barcode_idx"][()]]
    useFeatureAr = molInfo["features/id"][()][molInfo["feature_idx"][()]]
    umiAr = molInfo["umi"][()]
    countAr = molInfo["count"][()]

    if filtered:
        useCellLs = allBarcodeAr[molInfo["barcode_info/pass_filter"][()][:, 0]]
        useCellBoolLs = np.isin(useBarcodeAr, useCellLs)
        useBarcodeAr = useBarcodeAr[useCellBoolLs]
        useFeatureAr = useFeatureAr[useCellBoolLs]
        umiAr = umiAr[useCellBoolLs]
        countAr = countAr[useCellBoolLs]

    allUmiCount = pd.DataFrame(
        np.c_[
            umiAr,
            useBarcodeAr,
            countAr,
            useFeatureAr,
        ]
    )
    allUmiCount[0] = allUmiCount[0].map(numToSeq)
    allUmiCount["barcodeUmi"] = allUmiCount[1] + "_" + allUmiCount[0]
    allUmiCount = allUmiCount.reindex(["barcodeUmi", 3, 2], axis=1, copy=False)
    allUmiCount.columns = ["barcodeUmi", "featureName", "readCount"]
    return allUmiCount

