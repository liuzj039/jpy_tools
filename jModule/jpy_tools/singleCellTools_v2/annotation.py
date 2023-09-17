import scvi
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from scipy.stats import spearmanr, pearsonr, zscore
from loguru import logger
from io import StringIO
from concurrent.futures import ProcessPoolExecutor as Mtp
from concurrent.futures import ThreadPoolExecutor as MtT
import sh
import h5py
from tqdm import tqdm
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
import collections
from . import utils
from ..rTools import rcontext
from ..otherTools import F


class Annotation(object):
    def __init__(self, ad_ref, ad_query, refLayer, queryLayer, refLabel, resultKey):
        self.ad_ref = ad_ref
        self.ad_query = ad_query
        self.refLayer = refLayer
        self.queryLayer = queryLayer
        self.resultKey = resultKey
        self.refLabel = refLabel

    @rcontext
    def cellid(
        self,
        markerCount=200,
        n_top_genes=5000,
        ls_use_gene: Optional[List[str]] = None,
        cutoff: float = 2.0,
        nmcs: int = 30,
        queryBatchKey: Optional[str] = None,
        refLayer=None,
        queryLayer=None,
        rEnv=None
    ):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as ro
        from ..singleCellTools import geneEnrichInfo
        from ..rTools import ad2so, py2r, r2py
        rBase = importr("base")
        cellId = importr("CelliD")

        queryLayer = self.queryLayer if not queryLayer else queryLayer
        refLayer = self.refLayer if not refLayer else refLayer

        ad_query, ad_ref = utils.getOverlap(self.ad_query, self.ad_ref, copy=True)
        ad_query.X = ad_query.layers[queryLayer]
        ad_ref.X = ad_ref.layers[refLayer]
        utils.testAllCountIsInt(ad_query)
        utils.testAllCountIsInt(ad_ref)

        ad_integrated = sc.concat(
            {"ref": ad_ref, "query": ad_query}, label="batch_cellid", index_unique="-"
        )

        if not ls_use_gene:
            sc.pp.highly_variable_genes(
                ad_integrated,
                n_top_genes=n_top_genes,
                flavor="seurat_v3",
                batch_key="batch_cellid",
                subset=True,
            )
            ls_useGene = ad_integrated.var.index.to_list()
        else:
            ls_useGene = ls_use_gene

        sc.pp.normalize_total(ad_ref, 1e4)
        sc.pp.normalize_total(ad_query, 1e4)

        ad_ref = ad_ref[:, ls_useGene].copy()
        ad_query = ad_query[:, ls_useGene].copy()

        VectorR_Refmarker = geneEnrichInfo.getEnrichedGeneByCellId(
            ad_ref,
            "X",
            self.refLabel,
            markerCount,
            copy=True,
            returnR=True,
            nmcs=nmcs,
        )

        if not queryBatchKey:
            _ad = utils.getPartialLayersAdata(ad_query, ["X"])
            sc.pp.scale(_ad, max_value=10)
            adR_query = py2r(_ad)
            adR_query = cellId.RunMCA(adR_query, slot="X", nmcs=nmcs)
            df_labelTransfered = r2py(
                rBase.data_frame(
                    cellId.RunCellHGT(
                        adR_query, VectorR_Refmarker, dims=py2r(np.arange(1, 1 + nmcs))
                    ),
                    check_names=False,
                )
            ).T
        else:
            lsDf_labelTransfered = []
            for _ad in utils.splitAdata(ad_query, queryBatchKey):
                _ad = utils.getPartialLayersAdata(_ad, ["X"])
                sc.pp.scale(_ad, max_value=10)
                adR_query = py2r(_ad)
                adR_query = cellId.RunMCA(adR_query, slot="X", nmcs=nmcs)
                df_labelTransfered = r2py(
                    rBase.data_frame(
                        cellId.RunCellHGT(
                            adR_query, VectorR_Refmarker, dims=py2r(np.arange(1, 1 + nmcs))
                        ),
                        check_names=False,
                    )
                ).T
                lsDf_labelTransfered.append(df_labelTransfered)
            df_labelTransfered = pd.concat(lsDf_labelTransfered).reindex(
                self.ad_query.obs.index
            )

        self.ad_query.obsm[f"cellid_{self.refLabel}_labelTranferScore"] = df_labelTransfered
        self.ad_query.obs[f"cellid_{self.refLabel}_labelTranfer"] = self.ad_query.obsm[
            f"cellid_{self.refLabel}_labelTranferScore"
        ].pipe(lambda df: np.select([df.max(1) > cutoff], [df.idxmax(1)], "unknown"))
    
    
