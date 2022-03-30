import pysam
import sh
import click
import scanpy as sc
import pandas as pd
from concurrent.futures import ThreadPoolExecutor as mpT
from collections import defaultdict
from loguru import logger
import typing

def chunked(bam, length):
    ls_read = []
    for i,read in enumerate(bam):
        ls_read.append(read)
        if (i + 1) % length == 0:
            yield ls_read
            ls_read = []
    if len(ls_read) > 0:
        yield ls_read

def readBamToContentIt(
    bamFile,
    barcodeWithGroupDt: typing.Dict[str, str],
    barcodeRecord: typing.Union[
        str, typing.Callable[[pysam.libcalignedsegment.AlignedSegment], str]
    ] = "CB",
    clusterRecord: str = None
) -> typing.Dict[str, typing.Sequence[pysam.libcalignedsegment.AlignedSegment]]:
    """
    split Bam file.

    Parameters
    ----------
    bamFile :
        in bam. pysam bam file.
    barcodeWithGroupDt : typing.Dict[str, str]
        key is barcode, value is group information.
    barcodeRecord : typing.Union[ str, typing.Callable[[pysam.libcalignedsegment.AlignedSegment], str] ], optional
        How to extract the barcode information from bam.
        if type is str, will search the str from read's tag.
        if is function, will use the output as barcode information.
        by default "CB"

    Yields
    -------
    Iterator[typing.Dict[str, typing.Sequence[pysam.libcalignedsegment.AlignedSegment]]]
        Key is group name, value is a list which contains reads belong to this group.
    """
    chunkedBamFileGenerator = chunked(bamFile, int(1e7))
    for i, singleChunkedBam in enumerate(chunkedBamFileGenerator):
        logger.info(f"read bam: {i * 1e7} reads")
        chunkedReadContentDt = defaultdict(lambda: [])
        for read in singleChunkedBam:
            if isinstance(barcodeRecord, str):
                if read.has_tag(barcodeRecord):
                    readBarcode = read.get_tag(barcodeRecord)
                else:
                    readBarcode = None
            else:
                readBarcode = readBarcode(read)

            if not readBarcode:
                continue

            groupName = barcodeWithGroupDt.get(readBarcode, False)
            if groupName:
                if clusterRecord:
                    read.set_tag(clusterRecord, groupName)
                chunkedReadContentDt[groupName].append(read)
        yield chunkedReadContentDt


def writeReadToBam(
    readContentDt: typing.Dict[
        str, typing.Sequence[pysam.libcalignedsegment.AlignedSegment]
    ],
    groupBamFileDt: typing.Dict[str, pysam.libcalignmentfile.AlignmentFile],
):
    with mpT(12) as mT:
        writeRead = lambda bam, reads: [bam.write(read) for read in reads]
        for groupName, groupReads in readContentDt.items():
            groupBam = groupBamFileDt[groupName]
            mT.submit(writeRead, groupBam, groupReads)


@click.command()
@click.option("--bam", "inBamPath")
@click.option("--adata", "h5adPath")
@click.option("-o", "outDirPath")
@click.option("-g", "groupby", default=False)
@click.option("-s", "splitInfoLs", multiple=True, default=[])
@click.option("--tag", "addTag")
@click.option("--need-split-bam", "needSplitBam", is_flag=True, show_default=True)
def main(inBamPath, h5adPath, outDirPath, groupby, splitInfoLs,addTag, needSplitBam):
    """
    split Bam file.

    \b
    -g: group label. must store in adata.obs.
    -s: customize group results based on group lable. e.g.: -s Stele:2,9,8 -s Cortex:7.
    --tag: the tag recorded group information, if not specified, group information will be not recorded.
    --need-split-bam: if provide tag, only the bam contained all reads will be generated, if provide this flag, splited bam will be also generated.
    """
    outDirPath = outDirPath.rstrip('/') + '/'
    try:
        sh.mkdir(outDirPath)
    except:
        logger.warning(f"{outDirPath} Existed !!!")
    adata = sc.read_h5ad(h5adPath)

    if splitInfoLs:
        splitInfo = {
            x.split(':')[0]: x.split(':')[1].split(',')
            for x in splitInfoLs
        }
        splitInfoDt = {z: x for x, y in splitInfo.items() for z in y}
        logger.info(splitInfoDt)
        adata = adata[adata.obs[groupby].isin(list(splitInfoDt.keys()))]
        adata.obs['tempCol'] = adata.obs[groupby].map(splitInfoDt)
        groupby = 'tempCol'

    barcodeWithGroupDt = adata.obs[groupby].to_dict()

    groupLableReplaceLs = []
    for x, y in barcodeWithGroupDt.items():
        if '/' in y:
            y_ = y.replace('/', '_')
            if y not in groupLableReplaceLs:
                groupLableReplaceLs.append(y)
                logger.warning(f'{y} is changed into {y_}')
            barcodeWithGroupDt[x] = y_


    inBam = pysam.AlignmentFile(inBamPath)


    if not addTag:
        needSplitBam = True
    else:
        addTagBam = pysam.AlignmentFile(f'{outDirPath}addTag_unsorted.bam', 'wb', template=inBam)

    if needSplitBam:
        groupNameLs = set(list(barcodeWithGroupDt.values()))
        groupBamFileDt = {
            x: pysam.AlignmentFile(f'{outDirPath}{x}.bam', 'wb', template=inBam)
            for x in groupNameLs
        }

    for chunkedReadContentDt in readBamToContentIt(inBam, barcodeWithGroupDt, clusterRecord=addTag):
        if needSplitBam:
            writeReadToBam(chunkedReadContentDt, groupBamFileDt)
        if addTag:
            chunkedReadContentLs = [x for y in chunkedReadContentDt.values() for x in y]
            for read in chunkedReadContentLs:
                addTagBam.write(read) 

    if needSplitBam:
        [bam.close() for bam in groupBamFileDt.values()]
        logger.info('build index start')
        [sh.samtools.index(f'{outDirPath}{x}.bam') for x in groupNameLs]

    if addTag:
        addTagBam.close()
        sh.samtools.sort("-@12", f'{outDirPath}addTag_unsorted.bam', O='bam', o=f'{outDirPath}addTag.bam')
        sh.samtools.index(f'{outDirPath}addTag.bam')
        sh.rm(f'{outDirPath}addTag_unsorted.bam')

main()
