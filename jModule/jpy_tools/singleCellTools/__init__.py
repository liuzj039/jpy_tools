'''
Author: your name
Date: 2022-03-02 20:43:02
LastEditTime: 2022-03-02 21:13:14
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /undefined/public1/software/liuzj/scripts/jModule/jpy_tools/singleCellTools/__init__.py
'''
"""
single cell analysis tools wrapper
"""
import tensorflow # if not import tensorflow first, `core dump` will occur
import scanpy as sc
import pandas as pd
import numpy as np
from loguru import logger
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Sequence,
    Literal,
    Any,
    Tuple,
    Iterator,
    Mapping,
    Callable,
)

from ..otherTools import setSeed
from . import (
    basic,
    spatialTools,
    annotation,
    bustools,
    detectDoublet,
    diffxpy,
    scvi,
    normalize,
    multiModel,
    plotting,
    parseCellranger,
    geneEnrichInfo,
    others,
    parseSnuupy,
    recipe,
    removeAmbient
)
from .plotting import PlotAnndata
from .normalize import NormAnndata
from .annotation import LabelTransferAnndata

setSeed()

class DeAnndata(object):
    def __init__(self, ad:sc.AnnData, clusterKey:str, groupby:str, controlName:str, resultKey:str=None):
        self.ad = ad
        self.clusterKey = clusterKey
        self.groupby = groupby
        self.controlName = controlName
        if resultKey is None:
            self.resultKey = f"de_{self.groupby}"
        # if self.resultKey not in self.ad.uns:
        #     self.ad.uns[self.resultKey] = {}

    def augur(self, minCounts=30, nCores=64, layer='normalize_log', **dt_kwargs) -> pd.DataFrame:
        from joblib.externals.loky import get_reusable_executor
        import pertpy as pt
        import gc

        ag_rfc = pt.tl.Augur('random_forest_classifier')
        lsDf_res = []
        dt_geneWeight = {}
        logger.warning(
            f"`cell_type` and `label` will be overwritten by `{self.clusterKey}` and `{self.groupby}`"
        )
        self.ad.obs['cell_type'] = self.ad.obs[self.clusterKey]
        self.ad.obs['label'] = self.ad.obs[self.groupby]
        self.ad.X = self.ad.layers[layer].copy()
        logger.warning(
            f"adata.X will be overwritten by `{layer}`"
        )
        for treatment in self.ad.obs[self.groupby].unique():
            if treatment == self.controlName:
                continue
            logger.info(f"start to predict {treatment}")

            _ad = ag_rfc.load(self.ad, label_col='label', cell_type_col='cell_type', condition_label=self.controlName, treatment_label=treatment)
            ls_usedCelltype = (_ad.obs.value_counts(['label', 'cell_type']).unstack().min(0) > minCounts).loc[lambda _: _].index.to_list()
            _ad = _ad[_ad.obs['cell_type'].isin(ls_usedCelltype)].copy()
            _ad_res, df_res = ag_rfc.predict(_ad, n_threads=nCores, **dt_kwargs)

            df_res, df_geneWeight = df_res['summary_metrics'], df_res["feature_importances"]
            df_res = df_res.melt(var_name=self.clusterKey, ignore_index=False).rename_axis(index='metric').reset_index().assign(treatment=treatment, control=self.controlName)
            lsDf_res.append(df_res)
            dt_geneWeight[treatment] = df_geneWeight
            del _ad, _ad_res, df_res
            get_reusable_executor().shutdown(wait=True)
            gc.collect()

        df_res = pd.concat(lsDf_res, axis=0, ignore_index=True)
        self.ad.uns[f"{self.resultKey}_augur"] = df_res
        lsDf = []
        for treatment, df_geneWeight in dt_geneWeight.items():
            df_geneWeight = df_geneWeight.assign(treatment=treatment)
            lsDf.append(df_geneWeight)
        df_geneWeight = pd.concat(lsDf, axis=0, ignore_index=True)

        self.ad.uns[f"{self.resultKey}_augurGeneWeight"] = df_geneWeight
        return df_res, df_geneWeight
    
    def metaNeighbor(
            self, ls_hvg=None, fastVersion:bool=True, symmetricOutput=False, layer='normalize_log',
            **kwargs
        ):

        import pymn
        ad = self.ad
        logger.info("Change the .X of ad_ref and ad_query")
        logger.info(f"{layer} is used for ad")
        ad.X = ad.layers[layer]

        if ls_hvg is None:
            ar_hvg = pymn.variableGenes(ad, study_col=self.groupby, return_vect=True)
            ls_hvg = ar_hvg[ar_hvg].index.to_list()

        ad = ad[:, ad.var.index.isin(ls_hvg)].copy()
        if fastVersion:
            ad.obs[self.groupby] = ad.obs[self.groupby].astype(str)
            ad.obs[self.clusterKey] = ad.obs[self.clusterKey].astype(str)
        df_ptrained = None

        df_res = pymn.MetaNeighborUS(ad, study_col=self.groupby, ct_col=self.clusterKey, fast_version=fastVersion, symmetric_output=symmetricOutput, trained_model=df_ptrained, **kwargs)
        df_res = ad.uns['MetaNeighborUS']
        self.ad.uns[f'{self.resultKey}_MetaNeighborUS'] = df_res
        return df_res
    
    def identifyDegUsePseudoBulk(
            self, groupby:str=None, replicateKey:str=None, npseudoRep:int=None, clusterKey:str = None, method:Literal['DESeq2', 'edgeR']='edgeR', groups:Union[None, Tuple[str, ...], Tuple[str, Tuple[str, ...]]]=None, 
            randomSeed:int=39, shrink:Optional[Literal["apeglm", "ashr", "normal"]]=None, njobs:int=1, layer:str = 'raw'
        ) -> pd.DataFrame:
        """
        Identifies differentially expressed genes using pseudo-bulk\\rep analysis.

        Parameters:
            groupby (str, optional): The column name in the observation metadata to group the cells by. Defaults to None.
            replicateKey (str, optional): The column name in the observation metadata that indicates the replicate information. Cannot be used together with npseudoRep. Defaults to None.
            npseudoRep (int, optional): The number of pseudo-replicates to generate. Cannot be used together with replicateKey. Defaults to None.
            clusterKey (str, optional): The column name in the observation metadata that indicates the cluster information. Defaults to None.
            method (Literal['DESeq2', 'edgeR'], optional): The method to use for differential expression analysis. Defaults to 'edgeR'.
            groups (Union[None, Tuple[str, ...], Tuple[str, Tuple[str, ...]]], optional): The groups to compare. Defaults to None.
            randomSeed (int, optional): The random seed for reproducibility. Defaults to 39.
            shrink (Optional[Literal["apeglm", "ashr", "normal"]], optional): The method to use for shrinkage estimation. Defaults to None.
            njobs (int, optional): The number of parallel jobs to run. Defaults to 1.
            layer (str, optional): The layer to use for analysis. Defaults to 'raw'.

        Returns:
            pd.DataFrame: A DataFrame containing the differentially expressed genes.

        Raises:
            ValueError: If both replicateKey and npseudoRep are None or not None.

        """
        
        from .geneEnrichInfo import findDegUsePseudobulk, findDegUsePseudoRep
        from .basic import splitAdata
        if replicateKey is None and npseudoRep is None:
            raise ValueError("replicateKey and npseudoRep cannot be both None")
        if replicateKey is not None and npseudoRep is not None:
            raise ValueError("replicateKey and npseudoRep cannot be both not None")
        
        if groupby is None:
            groupby = self.groupby
        if clusterKey is None:
            clusterKey = self.clusterKey
        
        ad = sc.AnnData(self.ad.X, obs=self.ad.obs[[groupby, clusterKey]], var=self.ad.var)
        ad.layers[layer] = self.ad.layers[layer]

        lsDf_res = []
        for cluster, _ad in splitAdata(ad, clusterKey, needName=True):
            if groups is None:
                ls_allSample = _ad.obs[groupby].unique()
                _groups = [self.controlName, [x for x in ls_allSample if x != self.controlName]]
            if replicateKey is not None:
                df_res = findDegUsePseudobulk(_ad, compareKey=groupby, replicateKey=replicateKey, method=method, groups=_groups, randomSeed=randomSeed, shrink=shrink, njobs=njobs, layer=layer)
                df_res = df_res.assign(cluster=cluster)
                lsDf_res.append(df_res)
            elif npseudoRep is not None:
                df_res = findDegUsePseudoRep(_ad, compareKey=groupby, npseudoRep=npseudoRep, method=method, groups=_groups, randomSeed=randomSeed, shrink=shrink, njobs=njobs, layer=layer)
                df_res = df_res.assign(cluster=cluster)
                lsDf_res.append(df_res)
        df_res = pd.concat(lsDf_res, axis=0, ignore_index=True)
        return df_res
        
class QcAnndata(object):
    def __init__(self, ad, rawLayer):
        self.ad = ad
        self.rawLayer = rawLayer

    def removeDoubletByScDblFinder(self,
        groupby: Optional[str] = None,
        doubletRatio: Optional[float] = None,
        skipCheck: bool = False,
        dropDoublet: bool = True,
        BPPARAM=None
    ):
        from .detectDoublet import byScDblFinder
        byScDblFinder(self.ad, layer=self.rawLayer, copy=False, batch_key=groupby, doubletRatio=doubletRatio, skipCheck=skipCheck, dropDoublet=dropDoublet, BPPARAM=BPPARAM)
    
    def removeAmbientBySoupx(self, ad_raw:sc.AnnData, layerRaw:str='raw', res=1, correctedLayerName:str='soupX_corrected', forceAccept=True, rEnv=None):
        '''`removeAmbientBySoupx` removes the ambient signal from the data by using the soupX algorithm
        
        Parameters
        ----------

        ad_raw : sc.AnnData
            the raw data, which is used to calculate the ambient

        layerRaw : str, optional
            the layer in ad_raw that contains the raw counts
        correctedLayerName : str, optional
            the name of the layer that will be created in the ad object.
        rEnv
            R environment to use. If None, will create a new one.
        
        '''
        from .removeAmbient import removeAmbientBySoupx
        removeAmbientBySoupx(self.ad, ad_raw, layerAd=self.rawLayer, layerRaw=layerRaw, res=res, correctedLayerName=correctedLayerName, forceAccept=forceAccept, rEnv=rEnv)

class clusterAnndata(object):
    def __init__(self, ad, rawLayer):
        self.ad = ad
        self.rawLayer = rawLayer
    
    def getShilouetteScore(
            self, ls_res: List[float], obsm: Union[str, np.ndarray], clusterKey:str='leiden', subsample=None, metric='euclidean', show=True, check=True, pcs:int = 50, cores:int = 1, 
        ) -> Dict[str, float]:
        '''The function performs clustering using the Leiden algorithm on an AnnData object and calculates the silhouette score for each clustering result.

        Parameters
        ----------
        ad : sc.AnnData
            The parameter `ad` is an AnnData object, which is a data structure commonly used in single-cell RNA sequencing (scRNA-seq) analysis. It contains the gene expression data and associated metadata for each cell.
        ls_res : List[float]
            A list of resolution values to use for the Leiden clustering algorithm.
        obsm : Union[str, np.ndarray]
            The parameter `obsm` is the name of the key in the `ad` object's `.obsm` attribute that contains the data matrix used for clustering. It can be either a string representing the key name or a numpy array containing the data matrix itself.
        subsample
            The `subsample` parameter is an optional parameter that specifies the fraction of cells to subsample from the input `ad` AnnData object. If provided, only a fraction of cells will be used for calculating the silhouette score. If not provided, all cells in the `ad` Ann
        metric, optional
            The `metric` parameter specifies the distance metric to be used for calculating pairwise distances between observations. The default value is 'euclidean', which calculates the Euclidean distance between two points. Other possible values include 'manhattan' for Manhattan distance, 'cosine' for cosine similarity, and many more

        Returns
        -------
            a dictionary where the keys are the resolution values from the input list `ls_res` and the values are the corresponding silhouette scores calculated using the Leiden clustering algorithm.

        '''
        from .others import clusteringAndCalculateShilouetteScore
        return clusteringAndCalculateShilouetteScore(self.ad, ls_res, obsm, clusterKey=clusterKey, subsample=subsample, metric=metric, show=show, check=check, pcs=pcs, cores=cores)

    def getClusterSpecGeneByCellex(
            self,
            clusterName: str = "leiden",
            batchKey: Optional[str] = None,
            check=True,
            kayAddedPrefix: Optional[str] = None,
            layer=None,
            dt_kwargsForCellex: dict = {},
        ):
        from .geneEnrichInfo import calculateEnrichScoreByCellex
        layer = self.rawLayer if layer is None else layer
        calculateEnrichScoreByCellex(self.ad, layer=layer, clusterName=clusterName, batchKey=batchKey, copy=False, check=check, kayAddedPrefix=kayAddedPrefix, dt_kwargsForCellex=dt_kwargsForCellex)

class EnhancedAnndata(object):
    """
    A class representing an enhanced version of the sc.AnnData object.

    Attributes:
    - ad (sc.AnnData): Annotated data object.
    - rawLayer (str): Name of the raw layer in the AnnData object.
    - pl (PlotAnndata): An instance of the PlotAnndata class for plotting functionalities.
    - norm (NormAnndata): An instance of the NormAnndata class for normalization functionalities.
    - anno (LabelTransferAnndata): An instance of the LabelTransferAnndata class for label transfer functionalities.
    - de (DeAnndata): An instance of the DeAnndata class for differential expression analysis functionalities.
    """

    def __init__(self, ad: sc.AnnData, rawLayer:str = 'raw'):
        """
        Initialize the EnhancedAnndata class.

        Parameters:
        - ad (sc.AnnData): Annotated data object.
        - rawLayer (str): Name of the raw layer in the AnnData object.
        """
        self.ad = ad
        self.rawLayer = rawLayer
        self.pl = PlotAnndata(self.ad, rawLayer=self.rawLayer)
        self.norm = NormAnndata(self.ad, rawLayer=self.rawLayer)
        self.qc = QcAnndata(self.ad, rawLayer=self.rawLayer)
        self.cl = clusterAnndata(self.ad, rawLayer=self.rawLayer)
    
    @property
    def rawLayer(self):
        return self._rawLayer

    @rawLayer.setter
    def rawLayer(self, value):
        logger.warning(f"rawLayer will be overwritten by {value} and all the related objects will be re-initialized")
        self._rawLayer = value
        self.pl = PlotAnndata(self.ad, rawLayer=self.rawLayer)
        self.norm = NormAnndata(self.ad, rawLayer=self.rawLayer)
        self.qc = QcAnndata(self.ad, rawLayer=self.rawLayer)
        self.cl = clusterAnndata(self.ad, rawLayer=self.rawLayer)

    def __repr__(self):
        ls_object = [f"enhancedAnndata: {self.ad.__repr__()}"]
        ls_object.append(f"rawLayer: {self.rawLayer}")
        if hasattr(self, 'anno'):
            ls_object.append(f"anno: initialized, refLabel: {self.anno.refLabel}, refLayer: {self.anno.refLayer}, resultKey: {self.anno.resultKey}")
        return '\n'.join(ls_object)
    
    def __getitem__(self, key) -> 'EnhancedAnndata':
        """
        Get the subset of the current dataset.

        Parameters:
            key (str): The key to subset the dataset.

        Returns:
            EnhancedAnndata: The subset of the current dataset.
        """
        return EnhancedAnndata(self.ad[key], rawLayer=self.rawLayer)
    
    def to_df(self, layer=None) -> pd.DataFrame:
        return self.ad.to_df(layer=layer)

    def addRef(self, ad_ref: sc.AnnData, refLabel:str, refLayer:str = 'raw', resultKey:str=None):
        """
        Add a reference dataset to the current dataset.

        Parameters:
            ad_ref (sc.AnnData): The reference dataset to be added.
            refLabel (str): The label of the reference dataset.
            refLayer (str, optional): The layer of the reference dataset to be used. Defaults to 'raw'.
            resultKey (str, optional): The key to store the result. Defaults to None.
        """
        self.anno = LabelTransferAnndata(ad_ref, self.ad, refLabel=refLabel, refLayer=refLayer, queryLayer=self.rawLayer, resultKey=resultKey)
    
    def initDe(self, clusterKey: str, groupby: str, controlName: str, resultKey: str = None):
        """
        Initializes the differential expression analysis object.

        Parameters:
        - clusterKey (str): The key for the cluster column in the AnnData object.
        - groupby (str): The key for the grouping column in the AnnData object.
        - controlName (str): The name of the control group.
        - resultKey (str, optional): The key for storing the differential expression results. Defaults to None.
        """
        self.de = DeAnndata(self.ad, groupby=groupby, clusterKey=clusterKey, controlName=controlName, resultKey=resultKey)

    # set uns obs var obsm varm layers X
    @property
    def uns(self):
        return self.ad.uns

    @uns.setter
    def uns(self, value):
        self.ad.uns = value
    
    @property
    def obs(self):
        return self.ad.obs
    
    @obs.setter
    def obs(self, value):
        self.ad.obs = value
    
    @property
    def var(self):
        return self.ad.var
    
    @var.setter
    def var(self, value):
        self.ad.var = value

    @property
    def obsm(self):
        return self.ad.obsm
    
    @obsm.setter
    def obsm(self, value):
        self.ad.obsm = value

    @property
    def varm(self):
        return self.ad.varm
    
    @varm.setter
    def varm(self, value):
        self.ad.varm = value
    
    @property
    def layers(self):
        return self.ad.layers
    
    @layers.setter
    def layers(self, value):
        self.ad.layers = value
    
    @property
    def X(self):
        return self.ad.X
    
    @X.setter
    def X(self, value):
        self.ad.X = value