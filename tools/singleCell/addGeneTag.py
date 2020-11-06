'''
Description: 
Author: Liuzj
Date: 2020-09-27 12:47:39
LastEditTime: 2020-09-27 12:50:25
LastEditors: Liuzj
'''
import pickle
import pysam
import click



@click.command()
@click.option('-p', 'picklePath')
@click.option('-i', 'inBamPath')
@click.option('-o', 'outBamPath')
@click.option('--tag', 'geneIdTag', default='gi', show_default=True)
def main(picklePath, inBamPath, outBamPath, geneIdTag):
    """
    给bam加gene tag

    \b
    -p: ./step11_parseBedtoolsOutput/parseBedtoolsResult.pkl
    -i: inbam
    -o: outbam
    """
    with open(picklePath, 'rb') as fh:
        readWithGeneInformation = pickle.load(fh)

    readWithGeneInformation = {x:y['gene_id'] for x,y in readWithGeneInformation.items()}
    with pysam.AlignmentFile(inBamPath) as inBam:
        with pysam.AlignmentFile(outBamPath, 'wb', template=inBam) as outBam:
            for read in inBam:
                readGeneId = readWithGeneInformation.get(read.qname, 'None')
                read.set_tag(geneIdTag, readGeneId)
                outBam.write(read)

main()