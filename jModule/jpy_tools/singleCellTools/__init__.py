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
    multiModle,
    plotting,
    parseCellranger,
    geneEnrichInfo,
    others,
    parseSnuupy,
    recipe
)

setSeed()
