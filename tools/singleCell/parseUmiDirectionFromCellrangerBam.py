import os
from numpy import e
import pysam
import pyranges as pr
import pandas as pd
import numpy as np
import anndata
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import count, repeat
import click




def addTagToBarcodeFile(barcodePath):
    barcodeProcessedPath = f"{barcodePath}.addTag.tsv"
    os.system(f"zcat {barcodePath}| awk '{{print \"CB:Z:\"$1}}' > {barcodeProcessedPath}")
    return list(pd.read_table(barcodePath, compression='gzip', names=['barcode'])['barcode'])

def sortBam(cellRangerBamPath, threads, samtoolsPath, barcodePath):
    sortedByCBSamPath = f"{cellRangerBamPath}.sortedByCB.sam"
    samHeaderPath = f"{sortedByCBSamPath}.header"
    sortedOnlyCBSamPath = f"{sortedByCBSamPath}.onlyCB.sam"
    sortedOnlyCBBamPath = f"{sortedByCBSamPath}.onlyCB.bam"
    if barcodePath:
        barcodeProcessedPath = f"{barcodePath}.addTag.tsv"

    if not os.path.exists(sortedOnlyCBBamPath):
        logger.info("start to sort bam by CB tag")
        os.system(
            f"{samtoolsPath} sort -t CB -m 1536M -@ {threads} {cellRangerBamPath} -O SAM -o {sortedByCBSamPath}"
        )
        logger.info("start to filter reads which not have CB tag")
        os.system(f"{samtoolsPath} view -SH {sortedByCBSamPath} > {samHeaderPath}")
        if barcodePath:
            os.system(f"LC_ALL=C grep -F 'CB:Z' {sortedByCBSamPath} | LC_ALL=C grep -F -f {barcodeProcessedPath} > {sortedOnlyCBSamPath}")
        else:
            os.system(f"LC_ALL=C grep -F 'CB:Z' {sortedByCBSamPath} > {sortedOnlyCBSamPath}")
        logger.info("start to transform sam to bam")
        os.system(
            f"cat {samHeaderPath} {sortedOnlyCBSamPath} | {samtoolsPath} view -F 4 -Sb -@ {threads} - -o {sortedOnlyCBBamPath}"
        )
        os.system(f"rm {sortedByCBSamPath} {samHeaderPath} {sortedOnlyCBSamPath}")

    if barcodePath:
        os.system(f"rm {barcodeProcessedPath}")
    return sortedOnlyCBBamPath


def getOneUmiReadIt(bcList):
    def __getOneUmiReadIt(bcList):
        readList = []
        lastReadUmi = ""
        for read in bcList:
            currentReadUmi = read["tags"]["UB"]
            if currentReadUmi != lastReadUmi:
                yield lastReadUmi, readList
                readList = []
            lastReadUmi = currentReadUmi
            readList.append(read)
        yield lastReadUmi, readList

    readIt = __getOneUmiReadIt(bcList)
    next(readIt)
    for x in readIt:
        yield x


def getSameCbReadIt(bamFilePath):
    def __getSameCbReadIt(bamFilePath):
        bam = pysam.AlignmentFile(bamFilePath)
        bcList = []
        lastReadBc = ""
        for read in bam:
            if read.has_tag("CB"):
                currentReadBc = read.get_tag("CB")
                if currentReadBc != lastReadBc:
                    yield lastReadBc, bcList
                    bcList = []
                lastReadBc = currentReadBc
                bcList.append(read.to_dict())
        yield lastReadBc, bcList

    bamIt = __getSameCbReadIt(bamFilePath)
    next(bamIt)
    for x in bamIt:
        yield x


def getOneUmiDirection(readLs, trans2GeneDt):
    mappedGene = None
    mappedDirection = None
    readCounts = 0
    for read in readLs:
        if read['map_quality'] != '255': # unique mapping
            continue
        if ("GX" in read["tags"]) and ("AN" in read["tags"]):
            continue
        elif "GX" in read["tags"]:
            readMappedDirection = 1  # sense
            geneName = read["tags"]["GX"]
            if len(geneName.split(';')) > 1:
                continue
            readMappedGene = read["tags"]["GX"]
            readCounts += 1
        elif "AN" in read["tags"]:
            readMappedDirection = -1  # antisense
            transcriptName = read["tags"]["AN"]
            if len(transcriptName.split(';')) > 1:
                continue
            transcriptName = transcriptName.split(',')[0]
            readMappedGene = trans2GeneDt.get(transcriptName, transcriptName)
            readCounts += 1
        else:
            continue

        if not mappedDirection:
            mappedDirection = readMappedDirection
        else:
            if mappedDirection != readMappedDirection:
                return None, None, 0

        if not mappedGene:
            mappedGene = readMappedGene
        else:
            if mappedGene != readMappedGene:
                return None, None, 0

    return mappedDirection, mappedGene, readCounts


def processOneCell(bcNameWithReadsList, trans2GeneDt, i):
    bcName, bcList = bcNameWithReadsList
    bcUmiDirectionLs = []  # sense, antisense
    for singleRead in bcList:
        singleRead["tags"] = {
            x[0]: x[-1] for x in [y.split(":") for y in singleRead["tags"]]
        }
    bcList = [x for x in bcList if "UB" in x["tags"]]
    bcList.sort(key=lambda x: x["tags"]["UB"])
    for umi, readLs in getOneUmiReadIt(bcList):
        umiDirection, umiMappedGene, umiReadCounts = getOneUmiDirection(readLs, trans2GeneDt)
        if umiDirection:
            bcUmiDirectionLs.append(f"{bcName}_{umi}\t{umiMappedGene}\t{umiDirection}\t{umiReadCounts}")


    resultStr = "\n".join(bcUmiDirectionLs)


    if i % 1e4 == 0:
        logger.info(f"processed finished: {i} cells")

    return resultStr


def addArray(array, df, useBarcodeDt, allGeneDt):
    dfBarcodeIndexLs = [useBarcodeDt[x] for x in df.index]
    dfGeneIndexLs = [allGeneDt[x] for x in df.columns]
    array[np.ix_(dfBarcodeIndexLs, dfGeneIndexLs)] += df.values

def generateAdata(allGeneLs, useBarcodeLs, outUmiDrPath, outAdPath):
    allGeneDt = {y:x for x,y in enumerate(allGeneLs)}
    useBarcodeDt = {y:x for x,y in enumerate(useBarcodeLs)}

    senseAr = np.zeros((len(useBarcodeLs), len(allGeneLs)))
    antisenseAr = np.zeros((len(useBarcodeLs), len(allGeneLs)))

    umiDrDfIt = pd.read_table(outUmiDrPath, chunksize=1e6)

    for partUmiDf in umiDrDfIt:
        partUmiDfBcUmSplitSr = partUmiDf['barcodeUmi'].str.split('_')
        partUmiDf = partUmiDf.assign(Barcode = partUmiDfBcUmSplitSr.str[0], Umi = partUmiDfBcUmSplitSr.str[1])
        partUmiCountDf = partUmiDf.groupby("Direction").apply(
            lambda df: df.groupby(["Barcode", "Gene"])["Umi"].agg("count").unstack()
        ).fillna(0)
        
        partUmiCountDf_sense = partUmiCountDf.xs(1)
        addArray(senseAr, partUmiCountDf_sense, useBarcodeDt, allGeneDt)
        partUmiCountDf_antisense = partUmiCountDf.xs(-1)
        addArray(antisenseAr, partUmiCountDf_antisense, useBarcodeDt, allGeneDt)
    
    logger.info('Start to save matrix as anndata format')
    adata = anndata.AnnData(obs=pd.DataFrame(index=useBarcodeLs), var=pd.DataFrame(index=allGeneLs), layers={'sense': senseAr, 'antisense': antisenseAr})
    adata.write(outAdPath)


@click.command()
@click.option("-i", "cellRangerBamPath")
@click.option("-o", "outPath")
@click.option("-t", "threads", type=int)
@click.option("--bed", "bedPath")
@click.option("--barcode", "barcodePath")
@click.option("--samtools", "samtoolsPath", default="samtools", show_default=True)
def main(cellRangerBamPath, outPath, threads, bedPath, samtoolsPath, barcodePath):
    """
    get umi direction from bam output by cellranger

    \b
    -i : cellRanger bam path
    -o : output path, tsv format
    -t : threads
    --bed: Generated by paftools.js
    --barcode : barcode path, tsv.gz format
    --samtoolsPath : samtools path
    """

    generateH5ad = False
    if barcodePath:
        generateH5ad = True
        useBarcodeLs = addTagToBarcodeFile(barcodePath)
        
    sortedFilePath = sortBam(
        cellRangerBamPath, threads, samtoolsPath, barcodePath
    )

    transWithGeneIdSq = pr.read_bed(bedPath, as_df=True)['Name'].str.split('\|\|')
    trans2GeneDt = {x:y for x,y in transWithGeneIdSq}
    allGeneLs = list(set(trans2GeneDt.values()))
    allGeneLs.sort()

    logger.info("start to parse bam")
    with open(outPath, "w") as fh:
        print("barcodeUmi\tGene\tDirection\treadCounts", file=fh)
        for i, bcNameWithReadsList in enumerate(getSameCbReadIt(sortedFilePath)):
            bcInfoStr = processOneCell(bcNameWithReadsList, trans2GeneDt, i)
            if bcInfoStr:
                print(bcInfoStr, file=fh)

    if generateH5ad:
        logger.info("Start to calculate sense\\antisense matrix")
        outAdPath = outPath + '.h5ad'
        generateAdata(allGeneLs, useBarcodeLs, outPath, outAdPath)
    # with ThreadPoolExecutor(threads) as mtT:
    #     mtpResultsLs = mtT.map(
    #         processOneCell, getSameCbReadIt(sortedFilePath), count(), chunksize=10
    #     )
    # mtpResultsLs = list(mtpResultsLs)

    # with open(outPath, "w") as fh:
    #     print("barcode\tsense\tantisense", file=fh)
    #     for line in mtpResultsLs:
    #         print(line, file=fh)


if __name__ == "__main__":
    main()