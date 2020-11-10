import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from concurrent.futures import ProcessPoolExecutor as Mtp
import sh


#################
##一些简单的工具###
#################

def mkdir(dirPath):
    try:
        sh.mkdir(dirPath)
    except:
        logger.warning(f'{dirPath} existed!!')

        
def plotCellScatter(adata):
    """
    绘制anndata基本情况
    """
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=0)

    adata.obs["n_count"] = adata.X.sum(axis=1)
    ctGene = (adata.var_names.str.startswith("ATCG")) | (
        adata.var_names.str.startswith("ATMG"))
    adata.obs["percent_ct"] = (np.sum(adata[:, ctGene].X, axis=1)/
                               np.sum(adata.X, axis=1))
    [
        sc.pl.violin(adata, x, jitter=0.4, multi_panel=True)
        for x in ["n_count", "n_genes", "percent_ct"]
    ]

    
def creatAnndataFromDf(df):
    """
    dataframe转换成anndata
    """
    transformedAd = anndata.AnnData(
        X=df.values,
        obs=pd.DataFrame(index = df.index),
        var=pd.DataFrame(index = df.columns)
    )
    return transformedAd


def mergeAdataExpress(adata, groupby='louvain', useRaw=True, logTransformed=True):
    """
    通过adata.obs中的<groupby>对表达量合并
    """
    def __creatAnndataFromDf(df):
        transformedAd = anndata.AnnData(
            X=df.values,
            obs=pd.DataFrame(index = df.index),
            var=pd.DataFrame(index = df.columns)
        )
        return transformedAd


    adataExpressDf = pd.DataFrame(adata.raw.X.A, columns=adata.raw.var.index, index=adata.obs.index) if useRaw else adata.to_df()
    adataExpressDf = np.exp(adataExpressDf) - 1 if logTransformed else adataExpressDf

    groupbyExpressDf = adataExpressDf.join(adata.obs[groupby]).groupby(groupby).agg('mean')
    groupbyExpressDf = np.log(groupbyExpressDf + 1) if logTransformed else groupbyExpressDf

    groupbyExpressAd = __creatAnndataFromDf(groupbyExpressDf)
    
    return groupbyExpressAd


def calculateExpressionRatio(adata, clusterby):
    """
    逐个计算adata中每个基因在每个cluster中的表达比例
    
    adata:
        需要含有raw
    clusterby:
        adata.obs中的某个列名
    """
    transformAdataRawToAd = lambda adata: anndata.AnnData(X = adata.raw.X, obs = adata.obs, var = adata.raw.var)
    rawAd = transformAdataRawToAd(adata)
    expressionOrNotdf = (rawAd.to_df() > 0).astype(int)
    expressionOrNotdf[clusterby] = rawAd.obs[clusterby]
    expressionRatioDf = expressionOrNotdf.groupby(clusterby).agg('sum')/expressionOrNotdf.groupby(clusterby).agg('count')
    return expressionRatioDf


def calculateGeneAverageEx(expressionMtxDf, geneDt, method='mean'):
    """
    根据geneDt对expressionMtxDf计算平均值或中位数
    
    expressionMtxDf:
        形如adata.to_df()
        
    geneDt:
        形如:{
    "type1": [
        "AT5G42235",
        "AT4G00540",
        ],
    "type2": [
        "AT1G55650",
        "AT5G45980",
        ],
    }
    method:
        'mean|median'
            
    """
    averageExLs = []
    for typeName, geneLs in geneDt.items():
        typeAvgExpress = expressionMtxDf.reindex(geneLs, axis=1).mean(1) if method == 'mean' else expressionMtxDf.reindex(geneLs, axis=1).median(1)
        typeAvgExpress.name = typeName
        averageExLs.append(typeAvgExpress)
    averageExDf = pd.concat(averageExLs, axis=1)
    
    return averageExDf


def getClusterEnrichedGene(adata, useAttri = ['names', 'pvals', 'pvals_adj', 'logfoldchanges'], geneAnno = False, geneMarker = False):
    """
    获得每个cluster enriched的基因 
    adata:
        obs中有louvain
        执行过:
            sc.tl.rank_genes_groups(adata)
            sc.tl.filter_rank_genes_groups(adata)
        
    geneAnno：
        dict;
        存在两个key['curator_summary', 'Note'] 从gff文件中转换而来
    
    geneMarker:
        dict;
        key为gene
        value为tissue
    
        
    useAttri = ['names', 'pvals', 'pvals_adj', 'logfoldchanges']
    """
    useLouvain = adata.obs['louvain'].unique()
    useAttriMap = {x:y for x,y in zip(range(len(useAttri)), useAttri)}
    useDict = adata.uns['rank_genes_groups_filtered']
    allLouvainUseAttri = []
    for pointLouvain in useLouvain:
        pointLouvainUseAttri = []
        for pointAttr in useAttri:
            pointLouvainUseAttri.append(pd.Series(useDict[pointAttr][pointLouvain]))
        pointLouvainUseAttri = pd.concat(pointLouvainUseAttri, axis=1).rename(columns=useAttriMap).assign(clusters = pointLouvain)
        allLouvainUseAttri.append(pointLouvainUseAttri)
    allLouvainUseAttri = pd.concat(allLouvainUseAttri).dropna().sort_values('clusters').reindex(['clusters','names', 'pvals', 'pvals_adj', 'logfoldchanges'], axis=1)
    if geneMarker:
        allLouvainUseAttri = allLouvainUseAttri.assign(
            markerInfo = allLouvainUseAttri['names'].map(lambda x:geneMarker.get(x, 'Unknown')))
    if geneAnno:
        allLouvainUseAttri = allLouvainUseAttri.assign(
            curator_summary = allLouvainUseAttri['names'].map(geneAnno['curator_summary']),
            note = allLouvainUseAttri['names'].map(geneAnno['Note'])
        ).reset_index(drop=True)
#     allLouvainUseAttri.dropna(inplace=True)
    return allLouvainUseAttri.query("pvals_adj > 0.05")


def __shuffleLabel(adata, label, i):
    """
    used for getEnrichedScore
    """
    shuffleAd = adata.copy()
    #     shuffleAd.obs[label] = adata.obs[label].sample(frac=1).reset_index(drop=True)
    shuffleAd.obs[label] = adata.obs[label].sample(frac=1, random_state=i).values
    #     print(shuffleAd.obs.iloc[0])
    shuffleClusterDf = (
        mergeAdataExpress(shuffleAd).to_df().reset_index().assign(label=i)
    )

    return shuffleClusterDf


def getEnrichedScore(adata, label, geneLs, threads=12, times=100):
    """
    获得ES值。ES值是通过对adata.obs中的label进行重排times次，然后计算原始label的zscore获得

    adata:
        必须有raw且为log-transformed

    label:
        adata.obs中的列名

    geneLs:
        需要计算的基因

    threads:
        使用核心数

    times:
        重排的次数
    """

    geneLs = geneLs[:]
    geneLs[0:0] = [label]
    adata = adata.copy()


    allShuffleClusterExpressLs = []
    with Mtp(threads) as mtp:
        for time in range(1, times + 1):
            allShuffleClusterExpressLs.append(
                mtp.submit(__shuffleLabel, adata, label, time)
            )

    allShuffleClusterExpressLs = [x.result() for x in allShuffleClusterExpressLs]
    originalClusterDf = (
        mergeAdataExpress(adata).to_df().reset_index().assign(label=0)
    )
    allShuffleClusterExpressLs.append(originalClusterDf)
    allShuffleClusterExpressDf = (
        pd.concat(allShuffleClusterExpressLs).set_index("label").reindex(geneLs, axis=1)
    )
    logger.info(f"start calculate z score")
    allShuffleClusterZscoreDf = (
        allShuffleClusterExpressDf.groupby(label)
        .apply(lambda x: x.set_index(label, append=True).apply(zscore))
        .reset_index(level=0, drop=True)
    )

    clusterZscoreDf = allShuffleClusterZscoreDf.query("label == 0").reset_index(
        level=0, drop=True
    ).fillna(0)
    return clusterZscoreDf

#################
## 细胞注释 
#################

def cellTypeAnnoByCorr(adata, bulkExpressionDf, threads=1, method='pearsonr', reportFinalUseGeneCounts=False, geneCountsCutoff=0, logTransformed=True, returnR=False, keepZero=True, useRaw=True, reportCounts=50):
    """
    通过bulk数据的相关性鉴定细胞类型
    
    adata: log-transformed adata
    
    bulkExpressionDf: 
                                                AT1G01010  AT1G01030  AT1G01040
        bending cotyledon : chalazal endosperm   3.018853   2.430005   8.284994
        bending cotyledon : chalazal seed coat   2.385562   2.364294   8.674318
        bending cotyledon : embryo proper        2.258559   2.249158   7.577717
    
    returnR:
        返回最大值还是所有结果的r
        
    method:
        'pearsonr' | 'spearmanr'
    
    geneCountsCutoff:
        CPM
    """
    def __getSpearmanRForCell(cellExpressionSr):
        nonlocal i, bulkExpressionDf, keepZero, reportCounts, threads, method, geneCountsCutoff, reportFinalUseGeneCounts
        if not keepZero:
            cellExpressionSr = cellExpressionSr.pipe(lambda x:x[x != 0])
        cellExpressionSr = cellExpressionSr.pipe(lambda x:x[x >= geneCountsCutoff])
#         print(cellExpressionSr)
        useGeneLs = cellExpressionSr.index
        bulkExpressionDf = bulkExpressionDf.reindex(useGeneLs, axis=1).dropna(axis=1)
        useGeneLs = bulkExpressionDf.columns
        cellExpressionSr = cellExpressionSr.reindex(useGeneLs)
        
        if reportFinalUseGeneCounts:
            logger.info(len(cellExpressionSr))
        
        if len(cellExpressionSr) <= 1:
            return None
        
        if method == 'spearmanr':
            bulkExpressionCorrDf = bulkExpressionDf.apply(lambda x: spearmanr(cellExpressionSr, x)[0], axis=1)
        elif method == 'pearsonr':
            bulkExpressionCorrDf = bulkExpressionDf.apply(lambda x: pearsonr(cellExpressionSr, x)[0], axis=1)
        else:
            logger.error('Unrecognized method')
            1/0

        i += 1
        if i % reportCounts == 0:
            logger.info(f'{i * threads} / {cellCounts} processed')
            
        if not returnR:
            mostSimilarBulk = bulkExpressionCorrDf.idxmax()
            return mostSimilarBulk
        else:
            return bulkExpressionCorrDf

        
    i = 0
    adata = adata.copy()
    cellCounts = len(adata)
    geneCountsCutoff = np.log(geneCountsCutoff + 1)
    
    adataExpressDf = pd.DataFrame(adata.raw.X.A, columns=adata.raw.var.index, index=adata.obs.index) if useRaw else adata.to_df() 
    adataExpressDf = np.exp(adataExpressDf) - 1 if logTransformed else adataExpressDf
    adataExpressDf = adataExpressDf.div(adataExpressDf.sum(1), axis=0) * 1000000
    adataExpressDf = np.log(adataExpressDf + 1) if logTransformed else adataExpressDf
#     print(adataExpressDf)
    
    if threads == 1:
        cellAnnotatedType = adataExpressDf.apply(__getSpearmanRForCell, axis=1)
    else:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=threads)
        cellAnnotatedType = adataExpressDf.parallel_apply(__getSpearmanRForCell, axis=1)
    return cellAnnotatedType



def cellTypeAnnoByMarker(adata, allMarkerUse, expressionMtx, zscoreby = 'cluster', method='mean'):
    """
    通过marker基因表达量鉴定细胞类型

    adata:
        adata.obs中有louvain
        通过adata.raw.var来判断哪些基因表达
        
    allMarkerUse:
    {
     'Zhang et al.': 
         {
            'Columella root cap': 
                ['AT4G27400','AT3G18250', 'AT5G20045']
         }
    }
    expressionMtx:
         由adata.to_df()获得:
             没有log-transformed
             没有筛选基因
             经过normalize_sum
    
    zscoreby = cluster|cell
    
    method = mean|median
    """

    def _getMarkerExpressionGene(adata, allMarkerUse):
        """
        去除marker中不表达的基因
        adata 存在raw
        allMarkerUse 
            {'Zhang et al.': {'Columella root cap': ['AT4G27400','AT3G18250', 'AT5G20045']}}
        """
        expressionGene = set(adata.raw.var.index)

        integrateMarkerGene = {}
        for x, y in allMarkerUse.items():
            singleMarkerGeneUse = {}
            for j,k in y.items():
                k = list(set(k) & expressionGene)
                singleMarkerGeneUse[j] = k
            integrateMarkerGene[x] = singleMarkerGeneUse
        return integrateMarkerGene
    expressionMtx = expressionMtx.copy(True)
#     expressionMtx = np.log2(expressionMtx + 1)
    
    allMarkers = _getMarkerExpressionGene(adata, allMarkerUse)

    expressionMtx = expressionMtx.join(adata.obs['louvain'], how='inner')
    allLouvain = expressionMtx['louvain'].unique()
    expressionCounts = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: y.sum() / len(y))).fillna(0)
    expressionCounts = np.log2(expressionCounts + 1)
    expressionSizes = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: (y > 0).sum()/len(y))).fillna(0)
    if zscoreby == 'cluster':
        expressionZscore = expressionCounts.apply(zscore)
    elif zscoreby == 'cell':
        expressionMtx = np.log2(expressionMtx.drop('louvain', axis=1) + 1)
        expressionMtx = expressionMtx.apply(zscore)
        expressionZscore = expressionMtx.join(
            adata.obs['louvain'], how='inner'
        ).groupby('louvain').apply(
            lambda x:x.drop('louvain', axis=1
                           ).pipe(
                lambda y: y.sum() / len(y))
        ).fillna(0)
#     expressionCounts = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: y.sum() / len(y))).fillna(0)
#     expressionCounts = expressionMtx.groupby('louvain').apply(lambda x:x.drop('louvain', axis=1).pipe(lambda y: y.sum() / (y > 0).sum())).fillna(0)
    
    groupAllClustersExpressionCounts = []
    groupAllClustersExpressionZscore = []
    groupAllClustersExpressionSizes = []
    groupNames = []
    for stage, tissueGenes in allMarkers.items():
        for tissue, genes in tissueGenes.items():
            if method == 'mean':
                groupGeneCountsDf = expressionCounts.loc[:, genes].mean(1)
                groupGeneZscoreDf = expressionZscore.loc[:, genes].mean(1)
                groupGeneSizesDf = expressionSizes.loc[:, genes].mean(1)
            elif method == 'median':
                groupGeneCountsDf = expressionCounts.loc[:, genes].median(1)
                groupGeneZscoreDf = expressionZscore.loc[:, genes].median(1)
                groupGeneSizesDf = expressionSizes.loc[:, genes].median(1)                
            groupGeneCountsDf.name = f'{stage} {tissue}'
            groupGeneZscoreDf.name = f'{stage} {tissue}'
            groupGeneSizesDf.name = f'{stage} {tissue}'
            groupAllClustersExpressionCounts.append(groupGeneCountsDf)
            groupAllClustersExpressionZscore.append(groupGeneZscoreDf)
            groupAllClustersExpressionSizes.append(groupGeneSizesDf)

    groupAllClustersExpressionCounts = pd.concat(groupAllClustersExpressionCounts, 1)
    groupAllClustersExpressionZscore = pd.concat(groupAllClustersExpressionZscore, 1)
    groupAllClustersExpressionSizes = pd.concat(groupAllClustersExpressionSizes, 1)
    groupAllClustersExpression = pd.concat([groupAllClustersExpressionSizes.stack(), groupAllClustersExpressionZscore.stack(), groupAllClustersExpressionCounts.stack()], axis=1)
    groupAllClustersExpression.reset_index(inplace=True)
    groupAllClustersExpression.columns = ['louvain','tissues','Percentage of expressed nuclei', 'Z-score of Expression', 'Average expression']
#     groupAllClustersExpression = groupAllClustersExpression.reset_index()
#     groupAllClustersExpression.columns = ['louvain','tissues','Percentage of expressed nuclei', 'Average expression']
    return groupAllClustersExpression


def cellTypeAnnoByClusterEnriched(arrayExpressDf_StageTissue, clusterEnrichedGeneDf, useCluster='all', useGeneCounts=10):
    """
    使用cluster enriched基因在bulk数据中的表达情况对cluster进行注释
    暂时只能用在胚乳数据上 待进一步优化
    
    arrayExpressDf_StageTissue: dataframe, 形如
                                             AT1G01010  AT1G01030  AT1G01040
    stage             correctedTissue                                      
    bending cotyledon chalazal endosperm     3.018853   2.430005   8.284994
                      chalazal seed coat     2.385562   2.364294   8.674318
                      embryo proper          2.258559   2.249158   7.577717
                      general seed coat      2.000998   2.168115   7.721052
                      peripheral endosperm   2.503232   2.154924   8.002944
                      
    clusterEnrichedGeneDf: getClusterEnrichedGene输出
    useCluster：'all'|['1', '2']
    useGeneCounts: 每个cluster使用的基因数
    """
    stageOrderLs = [
        'pre-globular',
        'globular',
        'heart',
        'linear-cotyledon',
        'bending cotyledon',
        'mature green',
    ]
    tissueOrderLs = [
        'chalazal endosperm', 'micropylar endosperm', 'peripheral endosperm',
        'chalazal seed coat', 'general seed coat', 'embryo proper', 'suspensor'
    ]
    expressList = list(arrayExpressDf_StageTissue.columns)
    clusterEnrichedGene_FilteredDf = clusterEnrichedGeneDf.sort_values([
        'clusters', 'logfoldchanges'
    ], ascending=[True, False]).groupby('clusters').apply(lambda x: x.loc[x['names'].isin(expressList)].iloc[:useGeneCounts]).reset_index(drop=True)

    clusterEnrichedGeneName_FilteredDf = clusterEnrichedGene_FilteredDf.groupby('clusters')['names'].agg(lambda x:list(x))

    clusterEnrichedGeneFc_FilteredDf = clusterEnrichedGene_FilteredDf.groupby('clusters')['logfoldchanges'].agg(lambda x:np.exp2(x).mean())

    print(clusterEnrichedGeneName_FilteredDf.map(lambda x:len(x)))

    print(clusterEnrichedGeneFc_FilteredDf)

    if useCluster == 'all':
        useClusterLs = list(clusterEnrichedGeneName_FilteredDf.index)
    else:
        useClusterLs = useCluster
        
#     return arrayExpressDf_StageTissue

    # clusterName = useClusterLs[0]
#     import pdb;pdb.set_trace()
    for clusterName in useClusterLs:
        fig, ax = plt.subplots(figsize=[5,3])
        clusterEnrichedGeneExpressPatternInBulkDf = arrayExpressDf_StageTissue.loc[:, clusterEnrichedGeneName_FilteredDf[clusterName]].median(1).unstack().reindex(stageOrderLs).reindex(tissueOrderLs, axis=1)
        sns.heatmap(clusterEnrichedGeneExpressPatternInBulkDf, cmap='Reds', ax=ax)
        ax.set_title(f'Cluster {clusterName}')
        ax.set_xlabel('Tissue')
        ax.set_ylabel('Stage')
    
    EnrichedGeneExpressPatternInBulkDf = clusterEnrichedGeneName_FilteredDf.map(lambda x:arrayExpressDf_StageTissue.loc[:, x].median(1).idxmax())
    return EnrichedGeneExpressPatternInBulkDf


def cellTypeAnnoByEnrichedScore(adata, label, markerGeneDt, threads=12, times=100):
    """
    通过enriched score对cluster进行注释
    
    adata:
        必须有raw且为log-transformed

    label:
        adata.obs中的列名

    markerGeneDt:
        形如:{
    "type1": [
        "AT5G42235",
        "AT4G00540",
        ],
    "type2": [
        "AT1G55650",
        "AT5G45980",
        ],
    }

    threads:
        使用核心数

    times:
        重排的次数
    """
    markerGeneLs = list(set([y for x in markerGeneDt.values() for y in x]))
    clusterEnrichedScoreDf = getEnrichedScore(adata, label, markerGeneLs,threads, times)
    clusterTypeAvgEnrichedScoreDf = calculateGeneAverageEx(clusterEnrichedScoreDf, markerGeneDt)
    return clusterTypeAvgEnrichedScoreDf