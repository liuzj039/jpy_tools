import pysam
import sh
import click
import scanpy as sc
import pandas as pd
from more_itertools import ichunked
from concurrent.futures import ThreadPoolExecutor as mpT
from collections import defaultdict
from loguru import logger


def readBamToContentIt(bamFile, barcodeMap):
    logger.info('read bam')
    chunkedBamFileGenerator = ichunked(bamFile, 83100)
    for singleChunkedBam in chunkedBamFileGenerator:
        chunkedReadContent = defaultdict(lambda: [])
        for read in singleChunkedBam:
            barcode = read.qname.split('_')[0]
            groupName = barcodeMap.get(barcode, False)
            if groupName:
                chunkedReadContent[groupName].append(read)
        yield chunkedReadContent


def writeReadToBam(readContent, groupBamFiles):
    logger.info('write reads into bam')
    with mpT(12) as mT:
        writeRead = lambda bam, reads: [bam.write(read) for read in reads]
        for groupName, groupReads in readContent.items():
            groupBam = groupBamFiles[groupName]
            mT.submit(writeRead, groupBam, groupReads)


@click.command()
@click.option('--bam', 'bamPath')
@click.option('--adata', 'adataPath')
@click.option('-o', 'outDir')
@click.option('-g', 'groupby', default=False)
@click.option('-s', 'splitInfo', multiple=True, default=[])
def main(bamPath, adataPath, outDir, groupby, splitInfo):
    """
    根据adata中对<bam>进行拆分,
    必须要使用 -g 或者 -s \n
    -s : -s Stele:2,9,8 -s Cortex:7 \n
    -g : -g louvain
    """
    try:
        sh.mkdir(outDir)
    except:
        logger.warning(f"{outDir} Existed !!!")
    logger.info('read adata')
    adata = sc.read_h5ad(adataPath)
    if splitInfo:
        splitInfo = {
            x.split(':')[0]: x.split(':')[1].split(',')
            for x in splitInfo
        }
        splitInfo = {z: x for x, y in splitInfo.items() for z in y}
        logger.info(splitInfo)
        adata = adata[adata.obs['louvain'].isin(list(splitInfo.keys()))]
        adata.obs['tempCol'] = adata.obs['louvain'].map(splitInfo)
        groupby = 'tempCol'
    barcodeMap = adata.obs[groupby]
    barcodeMap.index = barcodeMap.index.str.split('-').str[0]
    barcodeMap = barcodeMap.astype(str).to_dict()
    for x, y in barcodeMap.items():
        if '/' in y:
            y_ = '_'.join(y.split('/'))
            logger.warning(f'{y} is changed into {y_}')
            barcodeMap[x] = y_
    groupNameList = set(list(barcodeMap.values()))
    logger.info('adata parsed into dict')
    bamFile = pysam.AlignmentFile(bamPath)
    groupBamFiles = {
        x: pysam.AlignmentFile(f'{outDir}{x}.bam', 'wb', template=bamFile)
        for x in groupNameList
    }
    logger.info('process and split bam')
    for chunkedReadContent in readBamToContentIt(bamFile, barcodeMap):
        writeReadToBam(chunkedReadContent, groupBamFiles)
    [bam.close() for bam in groupBamFiles.values()]
    logger.info('build index start')
    [sh.samtools.index(f'{outDir}{x}.bam') for x in groupNameList]


main()
