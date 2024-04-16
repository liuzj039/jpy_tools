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
# import tensorflow # if not import tensorflow first, `core dump` will occur
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
# from . import (
    # basic,
    # spatialTools,
    # annotation,
    # bustools,
    # detectDoublet,
    # diffxpy,
    # scvi,
    # normalize,
    # multiModel,
    # plotting,
    # parseCellranger,
    # geneEnrichInfo,
    # others,
    # parseSnuupy,
    # recipe,
    # removeAmbient
# )
from .plotting import PlotAnndata
from .normalize import NormAnndata
from .annotation import LabelTransferAnndata
from ..rTools import rcontext

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
        _ls_keepCol = [groupby, clusterKey] if replicateKey is None else [groupby, clusterKey, replicateKey]
        ad = sc.AnnData(self.ad.X, obs=self.ad.obs[_ls_keepCol], var=self.ad.var)
        ad.layers[layer] = self.ad.layers[layer]

        lsDf_res = []
        for cluster, _ad in splitAdata(ad, clusterKey, needName=True):
            if groups is None:
                ls_allSample = _ad.obs[groupby].unique()
                _groups = [self.controlName, [x for x in ls_allSample if x != self.controlName]]
            if replicateKey is not None:
                df_res = findDegUsePseudobulk(_ad, compareKey=groupby, replicateKey=replicateKey, method=method, groups=_groups, shrink=shrink, njobs=njobs, layer=layer)
                df_res = df_res.assign(cluster=cluster)
                lsDf_res.append(df_res)
            elif npseudoRep is not None:
                df_res = findDegUsePseudoRep(_ad, compareKey=groupby, npseudoRep=npseudoRep, method=method, groups=_groups, shrink=shrink, njobs=njobs, layer=layer)
                df_res = df_res.assign(cluster=cluster)
                lsDf_res.append(df_res)
        df_res = pd.concat(lsDf_res, axis=0, ignore_index=True)
        self.ad.uns[f'{self.resultKey}_deg'] = df_res
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

class ClusterAnndata(object):
    def __init__(self, ad, rawLayer):
        self.ad = ad
        self.rawLayer = rawLayer
    
    @rcontext
    def getSeuratSnn(self, obsm, n_neighbors, n_pcs=50, keyAdded='seurat', metric='euclidean', rEnv=None, **dt_kwargsToFindNeighbors):
        """
        The provided Python function, getSeuratSnn, integrates Python and R environments to perform neighborhood analysis on scRNA-seq data using the Seurat package, a popular tool in the R ecosystem for single-cell genomics analysis. This function requires an anndata object and utilizes both Python and R libraries to calculate shared nearest neighbors (SNN) based on a specified number of principal components (PCs) and neighbors. 

        ### Function Signature

        ```python
        def getSeuratSnn(self, obsm, n_neighbors, n_pcs=50, keyAdded='seurat', rEnv=None, **dt_kwargsToFindNeighbors):
        ```

        ### Parameters

        - `obsm` (str): The key within the `.obsm` attribute of the `anndata` object where the dimensionality-reduced data is stored. For example, 'X_pca' for PCA-reduced data.
        - `n_neighbors` (int): The number of neighbors to use when constructing the SNN graph.
        - `n_pcs` (int, optional): The number of principal components to consider in the analysis. Defaults to 50.
        - `keyAdded` (str, optional): The base key used to store the SNN graph and nearest neighbors in the `.obsp` attribute of the `anndata` object. Defaults to 'seurat'.
        - `rEnv` (rpy2.robjects.environments.Environment, optional): An R environment object used for passing variables between Python and R. If not provided, it may need to be initialized beforehand.
        - `**dt_kwargsToFindNeighbors`: Arbitrary keyword arguments to be passed to the `FindNeighbors` function in Seurat.

        ### Description

        This function performs the following steps:

        1. Imports necessary R packages and Python modules, particularly for handling sparse matrices and converting between `anndata` objects and Seurat objects.
        2. Extracts the specified dimensionality-reduced data (`obsm`) up to the number of specified principal components (`n_pcs`) from the `anndata` object associated with the instance (`self.ad`).
        3. Creates a temporary `anndata` object with this reduced data, copying the 'obs' and 'var' from the original `anndata` object. This temporary object is then used to create a Seurat object.
        4. Sets up and performs the `FindNeighbors` function in Seurat through R's environment, based on the specified `n_neighbors` and any additional arguments (`**dt_kwargsToFindNeighbors`). This function computes the SNN based on the reduced dataset.
        5. Converts the Seurat object back into an `anndata` object and updates the original `anndata` object's `.obsp` attribute with the SNN graph and nearest neighbors information under keys derived from the `keyAdded` parameter.

        ### Notes

        - This function requires an understanding of both Python and R programming, as well as familiarity with the `anndata` format used in Python for single-cell analysis and the Seurat package in R.
        - It leverages `rpy2` to interface between Python and R, allowing for the use of R functions within Python.
        - The user must ensure that the required R packages (`Seurat`, `DescTools`) are installed in their R environment.
        - This function is designed to be part of a larger class (indicated by the use of `self`), which should contain the `anndata` object (`self.ad`) to be analyzed.

        ### Example Usage

        This example assumes that `getSeuratSnn` is a method within a class that contains an `anndata` object as `self.ad`.

        ```python
        # Assuming `ead` is an instance of the class containing `getSeuratSnn`
        ead.getSeuratSnn(obsm='X_pca', n_neighbors=30)
        ```

        This call will update the `anndata` object within `analyzer` with the SNN graph and nearest neighbors information, based on the first 50 PCs and 30 neighbors, adding these under keys with the base 'seurat' in the `.obsp` attribute.
        """
        import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        import scipy.sparse as ss
        from ..rTools import py2r, ad2so, so2ad
        importr("Seurat")
        importr("DescTools")

        ad = self.ad
        ar_embedding = ad.obsm[obsm][:, :n_pcs]

        ad_temp = sc.AnnData(ss.csr_matrix(ad.shape), obs=ad.obs, var=ad.var)
        ad_temp.obsm[obsm] = ar_embedding

        ad_temp.layers['count'] = ad_temp.X.copy()
        ad_temp.layers['norm'] = ad_temp.X.copy()

        so = ad2so(ad_temp, dataLayer='norm', layer='count')

        _obsm = obsm[2:] if obsm.startswith('X_') else obsm
        dt_kwargs = dict(
            object=so, 
            reduction=_obsm, 
            dims=py2r(np.arange(1, n_pcs + 1)),\
        )
        dt_kwargs['k.param'] = n_neighbors
        dt_kwargs['annoy.metric'] = metric
        dt_kwargs.update(dt_kwargsToFindNeighbors)
        rEnv['kwargs'] = ro.r.list(**dt_kwargs)
        ro.r("so <- DoCall(FindNeighbors, kwargs)")

        ad_nn = so2ad(rEnv['so'])
        ad.obsp[f'{keyAdded}_snn'] = ad_nn.obsp['RNA_snn']
        ad.obsp[keyAdded] = ad_nn.obsp['RNA_nn']


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

    def __init__(self, ad: sc.AnnData, rawLayer:Optional[str] = None):
        """
        Initialize the EnhancedAnndata class.

        Parameters:
        - ad (sc.AnnData): Annotated data object.
        - rawLayer (str): Name of the raw layer in the AnnData object.
        """
        self.ad = ad
        if rawLayer is None:
            if 'EnhancedAnndata_rawLayer' in ad.uns:
                rawLayer = ad.uns['EnhancedAnndata_rawLayer']
            else:
                rawLayer = 'raw'
        # test whether '_' in ad.obs.index 
        if ad.obs.index.str.contains('_').any():
            logger.warning("The index of ad.obs contains '_' which may cause some problems, please remove it first")
        # test whether '_' in ad.var.index 
        if ad.var.index.str.contains('_').any():
            logger.warning("The index of ad.var contains '_' which may cause some problems, please remove it first")
                
        self.rawLayer = rawLayer
        self.pl = PlotAnndata(self.ad, rawLayer=self.rawLayer)
        self.norm = NormAnndata(self.ad, rawLayer=self.rawLayer)
        self.qc = QcAnndata(self.ad, rawLayer=self.rawLayer)
        self.cl = ClusterAnndata(self.ad, rawLayer=self.rawLayer)
    
    def initLayer(self, layer=None, total=1e4, needScale=False, logbase=2):
        """
        overwrite layer: `raw`, `normalize_log`, `normalize_log_scale`, 'X'
        """
        layer = self.rawLayer if layer is None else layer
        ad = self.ad
        ad.layers['raw'] = ad.layers[layer].copy()
        ad.layers['normalize_log'] = ad.layers['raw'].copy()
        sc.pp.normalize_total(ad, total, layer='normalize_log')
        sc.pp.log1p(ad, layer='normalize_log', base=logbase)
        if needScale:
            ad.layers['normalize_log_scale'] = ad.layers['normalize_log'].copy()
            sc.pp.scale(ad, layer='normalize_log_scale')
        ad.X = ad.layers['normalize_log'].copy()
    
    def copy(self) -> 'EnhancedAnndata':
        return EnhancedAnndata(self.ad.copy(), rawLayer=self.rawLayer)

    def subsample(self, n:int, randomSeed:int=39, copy:bool=False) -> 'EnhancedAnndata':
        """
        Subsample the current dataset.

        Parameters:
            n (int): The number of cells to subsample.
            randomSeed (int, optional): The random seed for reproducibility. Defaults to 39.
            copy (bool, optional): Whether to return a copy of the subsampled dataset. Defaults to False.

        Returns:
            EnhancedAnndata: The subsampled dataset.
        """
        df = self.ad.obs
        df = df.sample(n=n, random_state=randomSeed)
        ead = self[df.index, :].copy() if copy else self[df.index, :]
        return ead

    @property
    def rawLayer(self):
        return self.uns['EnhancedAnndata_rawLayer']

    @rawLayer.setter
    def rawLayer(self, value):
        logger.warning(f"rawLayer will be overwritten by {value} and all the related objects will be re-initialized")
        self.uns['EnhancedAnndata_rawLayer'] = value
        self.pl = PlotAnndata(self.ad, rawLayer=self.rawLayer)
        self.norm = NormAnndata(self.ad, rawLayer=self.rawLayer)
        self.qc = QcAnndata(self.ad, rawLayer=self.rawLayer)
        self.cl = ClusterAnndata(self.ad, rawLayer=self.rawLayer)

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
    def obsp(self):
        return self.ad.obsp
    
    @obsp.setter
    def obsp(self, value):
        self.ad.obsp = value

    @property
    def varm(self):
        return self.ad.varm
    
    @varm.setter
    def varm(self, value):
        self.ad.varm = value

    @property
    def varp(self):
        return self.ad.varp
    
    @varp.setter
    def varm(self, value):
        self.ad.varp = value

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