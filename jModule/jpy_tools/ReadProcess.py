'''
@Date: 2020-06-05 22:05:55
LastEditors: Liuzj
LastEditTime: 2020-10-12 13:58:17
@Description: 主要用于read的处理 bam文件 fast5等
@Author: liuzj
FilePath: /liuzj/softwares/python_scripts/jpy_modules/jpy_tools/ReadProcess.py
'''
import os
import sh
import glob
import pysam
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from ont_fast5_api.fast5_interface import get_fast5_file
from collections import namedtuple
from loguru import logger


class sequence:
    '''
    @description: 反向互补
    '''
    def __init__(self):
        old_chars = "ACGT"
        replace_chars = "TGCA"
        self.tab = str.maketrans(old_chars, replace_chars)

    def original(self, seq):
        return seq

    def complement(self, seq):
        return seq.translate(self.tab)

    def reverse(self, seq):
        return seq[::-1]

    def reverseComplement(self, seq):
        return seq.translate(self.tab)[::-1]


def writeFastq(read, fh):
    '''
    @description: 用于将pyfastx的read输出为fastq
    @param:
        read: pyfastx fastq
        fh: file fh mode w
    @return: None
    '''
    readContent = f'@{read.name}\n{read.seq}\n{read.desc}\n{read.qual}\n'
    fh.write(readContent)

def writeFasta(read, fh):
    """
    @description: 用于将pyfastx的read输出为fasta
    @param:
        read: pyfastx fasta
        fh: file fh mode w
    @return: None
    """
    readContent = f'>{read.name}\n{read.seq}\n'
    fh.write(readContent)


def readFasta(path):
    '''
    @description: 读fasta
    @param {type} fasta路径
    @return: 一个迭代器
    '''
    FastaRead = namedtuple('FastaRead', ['name', 'seq'])

    def _readFasta(path):
        with open(path, 'r') as fh:
            i = 0
            while True:
                lineContent = fh.readline().strip()
                if lineContent == '':
                    break
                if lineContent.startswith('>'):
                    i += 1
                    if i == 1:
                        readName = lineContent[1:].split(' ')[0]
                        readSeq = ''
                    else:
                        read = FastaRead(name=readName, seq=readSeq)
                        yield read
                        readName = lineContent[1:].split(' ')[0]
                        readSeq = ''
                else:
                    readSeq += lineContent
            read = FastaRead(name=readName, seq=readSeq)
            yield read

    return _readFasta(path)


def readFastq(path, length=False):
    '''
    @description: 读fastq
    @param {type} fastq路径, 读取长度从3'算
    @return: 一个迭代器
    '''
    FastqRead = namedtuple('FastqRead', ['name', 'seq', 'desc', 'qual'])

    def _readFastq(path):
        with open(path, 'r') as fh:
            i = 0
            readContent = []
            while True:
                lineContent = fh.readline()
                if lineContent == '':
                    break
                i += 1
                readContent.append(lineContent.strip())
                if i % 4 == 0:
                    if not length:
                        read = FastqRead(name=readContent[0][1:].split(' ')[0],
                                         seq=readContent[1],
                                         desc=readContent[2],
                                         qual=readContent[3])
                    else:
                        read = FastqRead(name=readContent[0][1:].split(' ')[0],
                                         seq=readContent[1][:length],
                                         desc=readContent[2],
                                         qual=readContent[3][:length])
                    yield read
                    readContent = []

    return _readFastq(path)


def getSubFastq(fastqRead, subRegion):
    """
    subRegion: np.array shape n*2
    """
    FastqRead = namedtuple("FastqRead", ["name", "seq", "desc", "qual"])
    name = fastqRead.name
    desc = fastqRead.desc
    seq = ""
    qual = ""
    for singleSubRegion in subRegion:
        seq += fastqRead.seq[singleSubRegion[0]:singleSubRegion[1]]
        qual += fastqRead.qual[singleSubRegion[0]:singleSubRegion[1]]
    read = FastqRead(name=name, desc=desc, seq=seq, qual=qual)
    return read


### single cell


def transformExpressionMatrixTo10XMtx(inputPath, outputDir):
    """
    input:
        path or dataframe

    基因表达矩阵转成10X表达矩阵
    column 为 基因名
    index  为 barcode名 ( 无 -1 )
    """

    try:
        sh.mkdir(outputDir)
    except:
        sh.rm('-rf', outputDir)
        sh.mkdir(outputDir)

    if isinstance(inputPath, str):
        expressionMtx = pd.read_table(
            inputPath,
            index_col=0,
        )
    else:
        expressionMtx = inputPath
        expressionMtx.rename_axis('index', inplace=True)
    expressionMtx = expressionMtx.loc[:, expressionMtx.sum(0) != 0]
    barcodes = pd.Series(expressionMtx.index + '-1')
    barcodes.to_csv(f'{outputDir}barcodes.tsv', header=None, index=None)

    feature = pd.DataFrame(expressionMtx.columns)
    feature[1] = feature.iloc[:, 0]
    feature[2] = 'Gene Expression'
    feature.to_csv(f'{outputDir}features.tsv',
                   sep='\t',
                   header=None,
                   index=None)

    indexMap = {
        i: k
        for i, k in zip(expressionMtx.index,
                        range(1, 1 + len(expressionMtx.index)))
    }

    featureMap = {
        i: k
        for i, k in zip(expressionMtx.columns,
                        range(1, 1 + len(expressionMtx.columns)))
    }

    expressionMtx.index = expressionMtx.index.map(indexMap)
    expressionMtx.columns = expressionMtx.columns.map(featureMap)
    expressionMtx = expressionMtx.astype(int)
    expressionMtx.reset_index(inplace=True)
    expressionMtx = expressionMtx.melt(id_vars='index')

    expressionMtx.columns = ['barcode', 'feature', 'count']
    expressionMtx = expressionMtx.query("count != 0")
    expressionMtx = expressionMtx.reindex(['feature', 'barcode', 'count'],
                                          axis=1)
    expressionMtx.sort_values(['barcode', 'feature'],
                              ascending=[True, False],
                              inplace=True)
    featureCounts, barcodeCounts, rowCounts = max(
        expressionMtx['feature']), max(
            expressionMtx['barcode']), len(expressionMtx)
    with open(f"{outputDir}matrix.mtx", "w") as fh:
        fh.write(
            f'%%MatrixMarket matrix coordinate integer general\n%metadata_json: {{"format_version": 2, "software_version": "3.1.0"}}\n{featureCounts} {barcodeCounts} {rowCounts}'
        )
        for line in expressionMtx.itertuples():
            fh.write(f'\n{line.feature} {line.barcode} {line.count}')

    sh.gzip(glob.glob(f"{outputDir}*"))


def extractReadCountsByUmiFromTenX(molInfoPath, filtered=True):
    '''
    @description: 用于从10Xh5文件中提取read counts
    '''
    def NumToSeq():
        numToBase = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}

        def _numToSeq(num):
            num = int(num)
            numStr = f'{num:020b}'
            return ''.join(
                [numToBase[numStr[2 * x:2 * x + 2]] for x in range(10)])

        return _numToSeq

    numToSeq = NumToSeq()
    molInfo = h5py.File(molInfoPath, 'r')
    allBarcode = molInfo['barcodes'].value.astype(str)
    allFeature = molInfo['features/id'].value.astype(str)

    allUmiCount = pd.DataFrame(np.c_[molInfo['umi'].value, \
        allBarcode[molInfo['barcode_idx'].value], \
            molInfo['count'].value, \
                allFeature[molInfo['feature_idx'].value]])
    if filtered:
        allUmiCount = allUmiCount[allUmiCount[1].isin(
            allBarcode[molInfo['barcode_info/pass_filter'].value[:, 0]])]
    allUmiCount[0] = allUmiCount[0].map(numToSeq)
    allUmiCount['barcodeUmi'] = allUmiCount[1] + allUmiCount[0]
    allUmiCount = allUmiCount.reindex(['barcodeUmi', 3, 2], axis=1, copy=False)
    allUmiCount.columns = ['barcodeUmi', 'featureName', 'readCount']
    return allUmiCount


def getMarkerGeneOverlapExpressionMat(path, adata, cutoff=2, needName=False):
    """
    用于获得marker gene表达情况
    path: marker-gene path:
          group  locus  geneName
          Elongation AT1G02205 CER1
    adata:
        scanpy adata  annotated with enriched gene
    cutoff:
        remove those gene which expressed in more than CUTOFF cluster
    needName:
        need gene name or not
        
    return:
        expression mat 1:
            enriched gene overlap with marker gene
        expression mat 2:
            enriched gene overlap with filtered marker gene
        marker gene raw,
        marker gene express in adata,
        marker gene express in adata and filtered
        (name & filter name)
    version:
         20200919:
            ignore isoform level marker gene;
            get true louvain name

    """
    def readMarkerGeneTsv(path, adata):
        markerGene = pd.read_table(path)
        markerGene = markerGene.groupby('group')['locus'].agg(
            lambda x: set(x)).to_dict()
        markerGeneOverlap = {
            i: (j & set(adata.raw.var_names))
            for i, j in markerGene.items()
        }
        return markerGene, markerGeneOverlap

    def transformMarkerGeneIntoDict(adata):
        allMarkerGene = adata.uns['rank_genes_groups_filtered']['names']
        clusterNum = len(allMarkerGene[0])
        clusterName = adata.uns['rank_genes_groups_filtered']['names'].dtype.names
        allMarkerGene = np.array([i for j in allMarkerGene for i in j])
        allMarkerGene = allMarkerGene.reshape(-1, clusterNum)
        allMarkerGene = allMarkerGene.swapaxes(0, 1)
        allMarkerGene = {
            str(i): k
            for i, k in zip(clusterName, allMarkerGene)
        }
        allMarkerGene = {
            i: set([k.split('.')[0] for k in j if k != 'nan'])
            for i, j in allMarkerGene.items()
        }
        mergeAllMarkerGene = pd.Series(
            [x for y in allMarkerGene.values() for x in y]).value_counts()
        keepMarkerGene = set(
            mergeAllMarkerGene.loc[mergeAllMarkerGene <= cutoff].index)
        allMarkerGeneRemoveDup = {
            i: (k & keepMarkerGene)
            for i, k in allMarkerGene.items()
        }
        return allMarkerGeneRemoveDup

    def getClusterVsMarkerOverlapGene(clusterEnrichedGene, markerGene):
        indexList = list(markerGene.keys())
        indexList.append('All')
        enrichedVsMarkerGene = pd.DataFrame(index=indexList)
        for clusterNum in clusterEnrichedGene.keys():
            clusterOverlapGene = []
            clusterAllOverlapGene = set()
            for markerClass in markerGene.keys():
                clusterOverlapGene.append(clusterEnrichedGene[clusterNum]
                                          & markerGene[markerClass])
                clusterAllOverlapGene = clusterAllOverlapGene | (
                    clusterEnrichedGene[clusterNum] & markerGene[markerClass])
            clusterOverlapGene.append(clusterAllOverlapGene)
            enrichedVsMarkerGene[clusterNum] = clusterOverlapGene
        return enrichedVsMarkerGene

    def transformToOutputFormat(clusterOverlapName, clusterEnrichedGene,
                                markerGene, markerGeneOverlap,
                                markerGeneOverlapFiltered):
        markerGene, markerGeneOverlap, markerGeneOverlapFiltered = markerGene.copy(), markerGeneOverlap.copy(), markerGeneOverlapFiltered.copy()
        markerGene['enrichedGene'], markerGeneOverlap['enrichedGene'], markerGeneOverlapFiltered['enrichedGene'] = set(
            ), set(), set()
        clusterOverlapName.loc['enrichedGene'] = clusterEnrichedGene
        clusterOverlapName.loc[:, 'typeEnrichedGene'] = markerGene.values()
        clusterOverlapName.loc[:,
                               'typeEnrichedExpression'] = markerGeneOverlap.values(
                               )
        clusterOverlapName.loc[:,
                               'typeEnrichedExpressionFiltered'] = markerGeneOverlapFiltered.values(
                               )
        clusterOverlapName = clusterOverlapName.applymap(lambda x: len(x))
        return clusterOverlapName

    markerGene, markerGeneOverlap = readMarkerGeneTsv(path, adata)
    clusterEnrichedGene = transformMarkerGeneIntoDict(adata)
    markerGeneOverlapName = getClusterVsMarkerOverlapGene(
        clusterEnrichedGene, markerGene)
    allClusterMergeOverlapGeneCounts = pd.Series([
        i for j in markerGeneOverlapName.loc['All'] for i in j
    ]).value_counts()
    descardDupAllMergeOverlapGeneName = set(
        allClusterMergeOverlapGeneCounts.loc[
            allClusterMergeOverlapGeneCounts > 1].index)
    markerGeneOverlapFiltered = {
        i: (j - descardDupAllMergeOverlapGeneName)
        for i, j in markerGeneOverlap.items()
    }

    markerGeneOverlapName = markerGeneOverlapName.query("index != 'All'")

    markerGeneOverlapNameFiltered = markerGeneOverlapName.applymap(
        lambda x: x - descardDupAllMergeOverlapGeneName)

    markerGeneExpressionMat = transformToOutputFormat(
        markerGeneOverlapName, clusterEnrichedGene, markerGene,
        markerGeneOverlap, markerGeneOverlapFiltered)

    markerGeneExpressionMatFiltered = transformToOutputFormat(
        markerGeneOverlapNameFiltered, clusterEnrichedGene, markerGene,
        markerGeneOverlap, markerGeneOverlapFiltered)

    if needName:
        return markerGeneExpressionMat, markerGeneExpressionMatFiltered, markerGene, markerGeneOverlap, markerGeneOverlapFiltered, markerGeneOverlapName, markerGeneOverlapNameFiltered
    else:
        return markerGeneExpressionMat, markerGeneExpressionMatFiltered, markerGene, markerGeneOverlap, markerGeneOverlapFiltered


### read process


def getBlock(read, intron):
    '''
    @description: 获取read的bolck
    @param 
        read:{pysam.read} 
        intron:{pysam.intron}
    @return: 
        [(start, end),(start, end)]
    '''
    block = []
    preStart, lastEnd = read.reference_start, read.reference_end
    for intronStart, intronEnd in intron:
        block.append((preStart, intronStart))
        preStart = intronEnd
    block.append((preStart, lastEnd))
    return block


def readHasLongJunction(read, bamFile):
    """
    判断最后一个intron是否过长
    100 没有intron
    00 intron正常
    10 intron 5'异常
    01 intron 3'异常
    11 均异常
    #########
    read: pysam read
    bamFile: pysam bam
    """
    def isExceedExtend(read, introns):
        if len(introns) == 0:
            return 100
        else:
            exons = np.array(getBlock(read, introns))
            introns = np.array(introns)
            exonLength = exons[:, 1] - exons[:, 0]
            intronLength = introns[:, 1] - introns[:, 0]
            result = 0
            if exonLength[-1] / intronLength[-1] <= 0.01:
                result += 1
            if exonLength[0] / intronLength[0] <= 0.01:
                result += 10
            return result

    introns = list(bamFile.find_introns([read]))
    exceedExtend = isExceedExtend(read, introns)

    return exceedExtend


def readHasLongJunctionStrand(read, bamFile):
    """
    判断最后一个intron是否过长 链特异
    
    100: read unmap
    00: not reverse and normal
    01: not reverse and 3' aberrant
    10: reverse and normal
    11: reverse and 5' aberrant
    """

    resultNum = 0
    if read.is_unmapped:
        resultNum += 100
        return resultNum
    else:
        intronInfo = readHasLongJunction(read, bamFile)

        if read.is_reverse:
            resultNum += 10

            if isOne(intronInfo, 1):
                resultNum += 1

        else:
            if isOne(intronInfo, 0):
                resultNum += 1

        return resultNum


## nanopore
def extract_read_data(fast5_filepath, read_id):
    '''
    @description: 
        It can handle fast5 basecalled with flip flop model.

    @param 
        fast5_filepath.
        read_id. 

    @return: 
        raw_data
        event_data
        fastq
        start
        stride
        samples_per_nt
    '''
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        read = f5.get_read(read_id)
        # compute event length vector
        model_type = read.get_analysis_attributes(
            'Basecall_1D_000')['model_type']
        if model_type == 'flipflop':
            # get the data
            raw_data = read.get_raw_data()
            fastq = read.get_analysis_dataset(
                group_name='Basecall_1D_000/BaseCalled_template',
                dataset_name='Fastq')
            fastq = fastq.split('\n')[1]
            start = read.get_summary_data(
                'Segmentation_000')['segmentation']['first_sample_template']
            move = read.get_analysis_dataset(
                group_name='Basecall_1D_000/BaseCalled_template',
                dataset_name='Move')
            stride = read.get_summary_data(
                'Basecall_1D_000')['basecall_1d_template']['block_stride']
            start_col = np.arange(start, start + stride * (len(move) - 1) + 1,
                                  stride)
            event_data = pd.DataFrame({
                'move': move,
                'start': start_col,
                'move_cumsum': np.cumsum(move)
            })
            event_data['model_state'] = event_data['move_cumsum'].map(
                lambda x: fastq[x - 1:x])
            called_events = len(event_data)

            # create event length data for tail normalization
            event_length_vector = np.empty(called_events)
            event_length_vector[:] = np.nan
            count = 0
            for i in range(called_events - 1, -1, -1):
                if event_data['move'][i] == 1:
                    event_length_vector[i] = count + 1
                    count = 0
                else:
                    count += 1
            # multiply moves by length of the event
            event_length_vector = event_length_vector * stride
            event_data['event_length_vector'] = event_length_vector
            #del event_data['move_cumsum']
            # remove NAs
            event_length_vector = event_length_vector[
                ~np.isnan(event_length_vector)]
            # Normalizer for flip-flop based data
            samples_per_nt = np.mean(event_length_vector[
                event_length_vector <= np.quantile(event_length_vector, 0.95)])
        else:
            raise ValueError('model type is not flipflop')

    return raw_data, event_data, fastq, start, stride, samples_per_nt


# bam process


def getRegionRead(
    pysamFile,
    useTag='GI',
    annoBedPath='/public/home/liuzj/data/Araport11/araport11.representative.gene_model.bed'
):
    """
    @introduction:
        从pysam Alignment文件中获得特定的read
        geneName: gene名
        useTag: 基因名tag
        annoBedPath: 代表转录本位置
    @return:
        返回一个函数 该函数的参数为基因名
    """
    def _readGeneAnno(annoPath):
        geneAnnoFile = pd.read_csv(
            annoPath,
            sep='\t',
            names=[
                'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
                'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
                'blockSizes', 'blockStarts'
            ],
            usecols=['chrom', 'chromStart', 'chromEnd', 'name', 'strand'])
        geneAnnoFile['geneId'] = geneAnnoFile['name'].str.split('.').str[0]
        geneAnnoFile.set_index('geneId', inplace=True)
        return geneAnnoFile.to_dict('index')

    def _filterRead(
        read,
        geneName,
        geneIsReverse,
    ):
        if read.has_tag(useTag):
            if read.get_tag(useTag) == geneName:
                return False
            else:
                return True
        else:
            if read.is_reverse ^ geneIsReverse:  # 链方向相同则不过滤
                return True
            else:
                return False

    def _filterGene(geneName):
        geneInfo = annoFile[geneName]
        geneBamFile = pysamFile.fetch(geneInfo['chrom'],
                                      geneInfo['chromStart'],
                                      geneInfo['chromEnd'])
        geneIsReverse = False if geneInfo['strand'] == '+' else True
        filteredRead = []
        for read in geneBamFile:
            if not _filterRead(read, geneName, geneIsReverse):
                filteredRead.append(read)
        # logger.info(f'{i} reads pass filter, based on pos not tag counts is {j/i}')
        return filteredRead

    annoFile = _readGeneAnno(annoBedPath)
    return _filterGene

def counts2tpm(ad, layer, bed_path, logScale=True):
    import pyranges as pr
    def counts2tpm(ad, layer, sr_geneLength):
        gene_len = sr_geneLength.reindex(ad.var.index).to_frame()
        sample_reads = ad.to_df(layer).T.copy()
        rate = sample_reads.values / gene_len.values
        tpm = rate / np.sum(rate, axis=0).reshape(1, -1) * 1e6
        return pd.DataFrame(data=tpm, columns=ad.obs.index, index=ad.var.index).T

    df_bed = pr.read_bed(bed_path, True)
    sr_geneLength = (
        df_bed.assign(
            ExonLength=lambda df: df["BlockSizes"]
            .str.split(",")
            .str[:-1]
            .map(lambda z: sum(int(x) for x in z)),
            GeneName=lambda df: df["Name"].str.split("\|").str[-1],
        )
        .groupby("GeneName")["ExonLength"]
        .agg("max")
    )
    sr_geneLength = sr_geneLength.reindex(ad.var.index)
    df_tpm = counts2tpm(ad, 'raw', sr_geneLength)
    if logScale:
        df_tpm = np.log(df_tpm + 1)
        ad.layers['tpm_log'] = df_tpm
    else:
        ad.layers['tpm'] = df_tpm


