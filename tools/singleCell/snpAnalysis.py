"""
Author: liuzj
Date: 2021-01-19 14:33:01
LastEditors: liuzj
LastEditTime: 2021-01-20 12:06:05
Description: 从read的Snp信息中获得UMI的Snp信息
FilePath: /singleCell/snpAnalysis.py
"""
import pysam
import gc
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from loguru import logger
from itertools import count, repeat
import scanpy as sc
import click
import jpy_tools.singleCellTools
import jpy_tools.otherTools


def extractSingleReadBarcodeUmi(read, useBarcodeUmiSt):
    if (read.has_tag("UB")) & (read.has_tag("CB")):
        readDt = {}
        readDt["readBarcode"] = read.get_tag("CB").split("-")[0]
        readDt["readUmi"] = read.get_tag("UB")
        readDt["readBarcodeUmi"] = readDt["readBarcode"] + "_" + readDt["readUmi"]
        if readDt["readBarcodeUmi"] in useBarcodeUmiSt:
            alignInfo = read.get_tag("xf")
            if (jpy_tools.otherTools.isOne(alignInfo, 0)) and (
                not jpy_tools.otherTools.isOne(alignInfo, 1)
            ):  # ones place: confidence mapping; tens place: majority feature
                readDt["name"] = read.qname
                return readDt
            else:
                return False
        else:
            return False
    else:
        return False


def extractBamReadId(bamPath):
    readIdLs = []
    with pysam.AlignmentFile(bamPath) as bam:
        for read in bam:
            readIdLs.append(read.qname)
    return readIdLs


def getUmiSnpInfo(umiWithReadSnpSr):
    """
    从read的Snp信息中获得UMI的Snp信息
    """
    umi, readSnpSr = umiWithReadSnpSr
    umiSnpDt = readSnpSr.value_counts().to_dict()
    umiSnpDt["barcodeUmi"] = umi
    return umiSnpDt


@click.command()
@click.option("--ad", "adataPath")
@click.option("--ml", "molPath")
@click.option("--cr-bam", "crBamPath")
@click.option("--one-bam", "colBamPath")
@click.option("--two-bam", "lerBamPath")
@click.option("-o", "outDir", required=True)
@click.option("-t", "threads", type=int, required=True)
@click.option("--kit", "kitVersion", default="v2")
@click.option("--ecoType", "ecoType", nargs=2, default=("COL", "LER"), required=True)
@click.option("--read-BU", "readBarcodeUmiInfoPath")
def main(
    adataPath,
    molPath,
    crBamPath,
    colBamPath,
    lerBamPath,
    outDir,
    threads,
    kitVersion,
    ecoType,
    readBarcodeUmiInfoPath
):
    """
    extract umi snp info from the bam file

    \b
    --ad: adataPath, only those barcodes listed in this file will be used; or filtered_feature_bc_matrix.h5 file
    --ml: cellRanger molecular path
    --cr-bam: cellRanger bam path
    --one-bam: split_diploid_hybrid_bam_to_each_parental_bam.pl result; should be quoted. e.g.:"/bamSplitDir/*COL.bam"
    --two-bam: split_diploid_hybrid_bam_to_each_parental_bam.pl result; should be quoted. e.g.:"/bamSplitDir/*LER.bam"
    -o: outDir
    -t: threads
    --kit: kitVersion, v2|v3
    --ecoType: label. e.g.:  COL LER
    --read-BU: output readBarcodeUmiInfoDf; if this file provided, the option ad, ml, cr-bam is not required
    """
    if not readBarcodeUmiInfoPath:
        # get used barcode
        if adataPath.endswith("h5ad"):
            mixedAd = sc.read_h5ad(adataPath)
        elif adataPath.endswith("h5"):
            mixedAd = sc.read_10x_h5(adataPath)
        else:
            0/0

        barcodeLs = mixedAd.obs.index.str.split("-").str[0]
        logger.info(f"barcode extraction finished")

        # get used UMI
        umiInfoDf = jpy_tools.singleCellTools.extractReadCountsByUmiFromTenX(
            molPath, kitVersion, False
        )
        umiInfoDfBarcodeUmiSplited = umiInfoDf["barcodeUmi"].str.split("_")
        umiInfoDf = umiInfoDf.pipe(
            lambda df: df.assign(
                barcode=umiInfoDfBarcodeUmiSplited.str[0],
                umi=umiInfoDfBarcodeUmiSplited.str[1],
            )
        ).query("barcode in @barcodeLs")
        useBarcodeUmiSt = set(umiInfoDf["barcodeUmi"])
        logger.info(f"UMI extraction finished")

        # get used read name
        crBam = pysam.AlignmentFile(crBamPath)
        readBarcodeUmiInfoDtLs = []
        for i, read in enumerate(crBam):
            readBarcodeUmiDt = extractSingleReadBarcodeUmi(read, useBarcodeUmiSt)
            if readBarcodeUmiDt:
                readBarcodeUmiInfoDtLs.append(readBarcodeUmiDt)
            if i % 1e6 == 0:
                logger.info(f"{i} reads processed")
        readBarcodeUmiInfoDf = pd.DataFrame(readBarcodeUmiInfoDtLs)
        logger.info(f"read extraction finished")
        del readBarcodeUmiInfoDtLs
        gc.collect()
        readBarcodeUmiInfoDf.to_feather(f"{outDir}readIdMapBarcodeUmi.fea")
    else:
        readBarcodeUmiInfoDf = pd.read_feather(readBarcodeUmiInfoPath)

    colBamPathLs = glob.glob(colBamPath)
    lerBamPathLs = glob.glob(lerBamPath)

    colReadLs = [y for x in colBamPathLs for y in extractBamReadId(x)]
    lerReadLs = [y for x in lerBamPathLs for y in extractBamReadId(x)]

    colReadSt = set(colReadLs)
    lerReadSt = set(lerReadLs)

    del colReadLs
    del lerReadLs
    gc.collect()

    readBarcodeUmiInfoDf = readBarcodeUmiInfoDf.pipe(
        lambda df: df.assign(
            ecoType=np.select(
                [
                    df["name"].map(lambda x: x in colReadSt),
                    df["name"].map(lambda x: x in lerReadSt),
                ],
                ecoType,
                "Unknown",
            )
        )
    )

    with ProcessPoolExecutor(threads) as mtP:
        mtpResultLs = mtP.map(
            getUmiSnpInfo,
            readBarcodeUmiInfoDf.groupby("readBarcodeUmi")["ecoType"],
            chunksize=2 ** 10,
        )
    mtpResultLs = list(mtpResultLs)

    umiSnpDf = (
        pd.DataFrame(mtpResultLs)
        .fillna(0)
        .reindex(["barcodeUmi", ecoType[0], ecoType[1], "Unknown"], axis=1)
    )

    umiSnpDf = umiSnpDf.assign(
        ecoType=lambda df: np.select(
            [
                df.eval(f"`{ecoType[0]}`>`{ecoType[1]}`"),
                df.eval(f"`{ecoType[1]}`>`{ecoType[0]}`"),
            ],
            ecoType,
            "Unknown",
        ),
        barcode=lambda df: df["barcodeUmi"].str.split("_").str[0],
    )

    umiSnpDf.to_feather(f"{outDir}umiSnpInfo.fea")


main()