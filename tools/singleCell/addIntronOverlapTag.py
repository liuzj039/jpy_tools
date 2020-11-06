'''
Description: 
Author: Liuzj
Date: 2020-09-28 14:19:17
LastEditTime: 2020-09-28 14:21:14
LastEditors: Liuzj
'''
import pysam
import pandas as pd
import click

@click.command()
@click.option('-i', 'inBamPath')
@click.option('-o', 'outBamPath')
@click.option('--ratio', 'intronOverlapRatioPath')
@click.option('--tag', 'intronOverlapTag', default='IO', show_default=True)
def main(inBamPath, outBamPath, intronOverlapRatioPath, intronOverlapTag):
    """
    添加intron overlap 信息

    /b
    -i : inbam
    -o : outbam
    --ratio : step13_getSplieStats/intronRetationInfoWithRatio.tsv <指定getSpliceStats的 --ratio模式>
    --tag： tag 
    
    """
    intronOverlapRatio = pd.read_table(intronOverlapRatioPath, index_col='Name')
    intronOverlapRatio = intronOverlapRatio['intronOverlapRatioInfo'].dropna().to_dict()
    with pysam.AlignmentFile(inBamPath) as inBam:
        with pysam.AlignmentFile(outBamPath, 'wb', template=inBam) as outBam:
            for read in inBam:
                readIntronOverlapInfo = intronOverlapRatio.get(read.qname, 'None')
                if readIntronOverlapInfo == 'None':
                    pass
                else:
                    read.set_tag(intronOverlapTag, readIntronOverlapInfo)
                outBam.write(read)

main()