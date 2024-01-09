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
from loguru import logger
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
    def __init__(self, ad:sc.AnnData, groupby:str, treatmentCol:str, controlName:str, resultKey:str=None):
        self.ad = ad
        self.groupby = groupby
        self.treatmentCol = treatmentCol
        self.controlName = controlName
        if resultKey is None:
            self.resultKey = f"de_{self.treatmentCol}"
        # if self.resultKey not in self.ad.uns:
        #     self.ad.uns[self.resultKey] = {}

    def augur(self, minCounts=30, nCores=64, layer='normalize_log', **dt_kwargs) -> pd.DataFrame:
        from joblib.externals.loky import get_reusable_executor
        import pertpy as pt
        import gc

        ag_rfc = pt.tl.Augur('random_forest_classifier')
        lsDf_res = []
        logger.warning(
            f"`cell_type` and `label` will be overwritten by `{self.groupby}` and `{self.treatmentCol}`"
        )
        self.ad.obs['cell_type'] = self.ad.obs[self.groupby]
        self.ad.obs['label'] = self.ad.obs[self.treatmentCol]
        self.ad.X = self.ad.layers[layer].copy()
        logger.warning(
            f"adata.X will be overwritten by `{layer}`"
        )
        for treatment in self.ad.obs[self.treatmentCol].unique():
            if treatment == self.controlName:
                continue
            logger.info(f"start to predict {treatment}")

            _ad = ag_rfc.load(self.ad, label_col='label', cell_type_col='cell_type', condition_label=self.controlName, treatment_label=treatment)
            ls_usedCelltype = (_ad.obs.value_counts(['label', 'cell_type']).unstack().min(0) > minCounts).loc[lambda _: _].index.to_list()
            _ad = _ad[_ad.obs['cell_type'].isin(ls_usedCelltype)].copy()
            _ad_res, df_res = ag_rfc.predict(_ad, n_threads=nCores, **dt_kwargs)

            df_res = df_res['summary_metrics']
            df_res = df_res.melt(var_name=self.groupby, ignore_index=False).rename_axis(index='metric').reset_index().assign(treatment=treatment, control=self.controlName)
            lsDf_res.append(df_res)
            del _ad, _ad_res, df_res
            get_reusable_executor().shutdown(wait=True)
            gc.collect()

        df_res = pd.concat(lsDf_res, axis=0, ignore_index=True)
        self.ad.uns[f"{self.resultKey}_augur"] = df_res
        return df_res

class EnhancedAnndata(object):
    def __init__(self, ad: sc.AnnData, rawLayer:str = 'raw'):
        self.ad = ad
        self.rawLayer = rawLayer
        self.pl = PlotAnndata(self.ad, rawLayer=self.rawLayer)
        self.norm = NormAnndata(self.ad, rawLayer=self.rawLayer)

    def __repr__(self):
        ls_object = [f"enhancedAnndata: {self.ad.__repr__()}"]
        ls_object.append(f"rawLayer: {self.rawLayer}")
        if hasattr(self, 'anno'):
            ls_object.append(f"anno: initialized, refLabel: {self.anno.refLabel}, refLayer: {self.anno.refLayer}, resultKey: {self.anno.resultKey}")
        return '\n'.join(ls_object)
    
    def addRef(self, ad_ref: sc.AnnData, refLabel:str, refLayer:str = 'raw', resultKey:str=None):
        self.anno = LabelTransferAnndata(ad_ref, self.ad, refLabel=refLabel, refLayer=refLayer, queryLayer=self.rawLayer, resultKey=resultKey)
    
    def initDe(self, groupby:str, treatmentCol:str, controlName:str, resultKey:str=None):
        self.de = DeAnndata(self.ad, groupby=groupby, treatmentCol=treatmentCol, controlName=controlName, resultKey=resultKey)