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