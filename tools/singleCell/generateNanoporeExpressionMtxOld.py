'''
Description: 
Author: Liuzj
Date: 2020-09-27 14:30:21
LastEditTime: 2020-10-10 14:23:19
LastEditors: Liuzj
'''
import pyranges as pr
import pandas as pd
import pysam
import portion
from collections import defaultdict
from loguru import logger
from jpy_tools.ReadProcess import transformExpressionMatrixTo10XMtx
import click

def parseReadApaInfo(apaClusterPath, inBamPath, geneTag, expressionInfo):
    """
    解析apa信息 从wp pipeline中获得apacluster info 随后对每个read进行注释
    """
    apaCluster = pr.read_bed(apaClusterPath, True)
    apaCluster['Name'] = apaCluster['Name'] + '_APA'
    apaCluster['geneName'] = apaCluster['Name'].str.split('_').str[0]
    apaCluster = apaCluster.reindex(['geneName','Name','Start', 'End'], axis=1)
    apaClusterDict = defaultdict(lambda :{})
    for line in apaCluster.itertuples():
        apaClusterDict[line.geneName][line.Name] = portion.closedopen(line.Start, line.End)
    readsApaInfo = {}

    with pysam.AlignmentFile(inBamPath) as inBam:
        i = 0
        for read in inBam:
            i += 1
            readGene = read.get_tag(geneTag)
            geneApaInfo = apaClusterDict.get(readGene,'None')
            if geneApaInfo == 'None':
                readApaName = f'{readGene}_N_APA'
            else:
                if read.is_reverse:
                    readEndPos = read.positions[0]
                else:
                    readEndPos = read.positions[-1]

                readApaName = f'{readGene}_N_APA'
                for apaName, apaSpan in geneApaInfo.items():
                    if readEndPos in apaSpan:
                        readApaName = apaName
                        break
            readsApaInfo[read.qname] = readApaName
            if i % 100000 == 0:
                logger.info(f'{i} reads processed')

    readsApaInfo = pd.Series(readsApaInfo)
    useUmi = list(set(readsApaInfo.index) & set(expressionInfo.index))
    expressionInfo = pd.concat([expressionInfo, readsApaInfo])
    expressionInfo = expressionInfo.loc[expressionInfo.index.isin(useUmi)]
    return expressionInfo





@click.command()
@click.option('-a', 'apaClusterPath', default='False')
@click.option('--bam', 'inBamPath', default='False')
@click.option('--tag','geneTag', default='gi')
@click.option('-i', 'inIrInfoPath')
@click.option('-o', 'outMtxDirPath')
@click.option('--ir/--no-ir', 'irMode', default=True)
def main(apaClusterPath, inBamPath, geneTag, inIrInfoPath, outMtxDirPath, irMode):
    """
    生成10X expression mtx 含有splice (APA) Info

    \b
    -a: polyAcluster bed ./single_cell_root.polya_cluster.bed
    --bam: 带有geneId的nanopore Bam 
    --tag: geneId tag名 默认gi
    -i : ./step14_getIrInfo/irInfo.tsv
    -o : 文件夹 10X表达矩阵 
    """
    mode = []
    if (apaClusterPath != 'False') & (inBamPath  != 'False'):
        mode.append('apa')
        logger.warning('apa mode')
    if irMode:
        mode.append('ir')
        logger.warning('ir mode')
    if not mode:
        logger.warning('expression only mode')

    ## 读取ir情况 并且生成含有splice信息的表达矩阵
    readIrInfo = pd.read_csv(inIrInfoPath, sep='\t')
    readIrInfo['fullySpliced'] = readIrInfo['IntronOverlapInfo'].isna().astype(str)
    readIrInfo['fullySpliced'] = readIrInfo.pipe(lambda x:x['geneId'] + '_' + x['fullySpliced'] + '_fullySpliced')
    readIrInfo.set_index('Name', inplace=True)
    if 'ir' in mode:
        expressionInfo = pd.concat([readIrInfo['geneId'], readIrInfo['fullySpliced']])
    else:
        expressionInfo = readIrInfo['geneId']

    if 'apa' in mode:
        expressionInfo = parseReadApaInfo(apaClusterPath, inBamPath, geneTag, expressionInfo)
    
    
    expressionInfo = pd.DataFrame(expressionInfo)
    expressionInfo.reset_index(inplace=True)
    expressionInfo.columns = ['BcUmi', 'expressionInfo']
    expressionInfo['Bc'] = expressionInfo['BcUmi'].str.split('_').str[0]
    expressionInfo = expressionInfo.groupby('Bc')['expressionInfo'].apply(pd.value_counts)
    expressionInfo = expressionInfo.unstack().fillna(0).astype(int)

    transformExpressionMatrixTo10XMtx(expressionInfo, outMtxDirPath)

main()