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

setSeed()
