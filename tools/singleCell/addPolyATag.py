'''
Description: 
Author: Liuzj
Date: 2020-09-25 15:45:18
LastEditTime: 2020-09-27 12:50:43
LastEditors: Liuzj
'''
import pysam
import pandas as pd
from jpy_tools.ReadProcess import getRegionRead
import pysam
import click


@click.command()
@click.option('-f', 'featherPath')
@click.option('-p', 'polyACallerResultPath')
@click.option('-i', 'bamFilePath')
@click.option('-o', 'addPolyAFilePath')
@click.option('--tag', 'polyATag', default = 'aL', show_default=True)
def main(featherPath, polyACallerResultPath, bamFilePath, addPolyAFilePath, polyATag):
    """
    输入polyACaller结果 加上polyA tag

    \b
    -f ./step7_parseMismatchResult/nanoporeReadWithBarcode.feather
    -p ./step2_polyACaller/exampleAddPolyALength.h5
    -i inban
    -o outbam
    """
    umiReadMapDt = pd.read_feather(featherPath)
    mapNameToId = umiReadMapDt.loc[:, ['qseqid','name']].set_index('name').to_dict()['qseqid']
    umiReadMapDt = umiReadMapDt.groupby('qseqid')['name'].agg(lambda x: list(x)).to_dict()
    umiLabel = set(umiReadMapDt.keys())

    addPolyAResult = pd.read_hdf(polyACallerResultPath)
    addPolyAResult = addPolyAResult.query('readType not in  ["invalid", "non-polyA/T"]')
    addPolyAResult['umiBarcode'] = addPolyAResult.index.map(mapNameToId)
    addPolyAResult = addPolyAResult.loc[:,['umiBarcode','tailLength']]
    addPolyAResult = addPolyAResult.groupby('umiBarcode')['tailLength'].agg('mean')
    addPolyAResult = addPolyAResult.reindex(umiLabel)
    addPolyAResult.fillna(0, inplace=True)
    addPolyAResult = addPolyAResult.to_dict()

    bamFile = pysam.AlignmentFile(bamFilePath)
    outBamFile = pysam.AlignmentFile(addPolyAFilePath, 'wb', template=bamFile)

    for read in bamFile:
        readUmiBarcode = read.qname[:27]
        polyALength = addPolyAResult[readUmiBarcode]
        read.set_tag(polyATag, polyALength, 'f')
        outBamFile.write(read)

    outBamFile.close()

main()