'''
Author: your name
Date: 2022-02-16 16:12:50
LastEditTime: 2022-02-16 16:37:10
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /undefined/public/home/liuzj/scripts/jModule/jpy_tools/singleCellTools/__init__.py
'''
"""
single cell analysis tools wrapper
"""
from ..otherTools import setSeed
from . import (
    basic,
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
    recipe
)

setSeed()
