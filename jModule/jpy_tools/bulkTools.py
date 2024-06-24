import pyranges as pr
import pandas as pd
import numpy as np
from rpy2.robjects.functions import Function
import scanpy as sc
from itertools import product
from joblib import Parallel, delayed
from typing import Collection, Tuple, Optional, Union, Callable, Literal
from loguru import logger
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.ipython.html
rpy2.ipython.html.init_printing()
from .rTools import py2r, r2py, r_inline_plot, rHelp, trl, rSet, rGet, ad2so, so2ad
from .otherTools import F
from . import loadPkl, toPkl
rBase = importr('base')
rUtils = importr('utils')
dplyr = importr('dplyr')
R = ro.r
R("options(browser='firefox', shiny.port=6533)")

def counts2deseq2(ad, layer, fc_logScale:Optional[Callable] = None):
    importr('DESeq2')
    renv = ro.Environment()
    with ro.local_context(renv) as rl:
        rl['df_mtx'] = py2r(ad.to_df(layer).T)
        rl['df_design'] = py2r(ad.obs[[]].assign(empty='1'))

        R("""
        dds <- DESeqDataSetFromMatrix(countData = df_mtx,
                                    colData = df_design,
                                    design = ~ 1)
        dds <- estimateSizeFactors(dds)
        normalized_counts <- counts(dds, normalized=TRUE)
        normalized_counts <- as.data.frame(normalized_counts)
        """)
        df_normalizedCounts = r2py(rl['normalized_counts']).T
    if fc_logScale:
        df_tpm = fc_logScale(df_normalizedCounts + 1)
        ad.layers['deseq2_log'] = df_tpm
    else:
        ad.layers['deseq2'] = df_normalizedCounts

def counts2tpm(ad, layer, bed_path, fc_logScale:Optional[Callable] = None):
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
    df_tpm = counts2tpm(ad, layer, sr_geneLength)
    if fc_logScale:
        df_tpm = fc_logScale(df_tpm + 1)
        ad.layers['tpm_log'] = df_tpm
    else:
        ad.layers['tpm'] = df_tpm


def deByEdger(ad, layer, obsKey, contrast, filterByEdger=True):
    # import rpy2.robjects as ro
    # from rpy2.robjects.packages import importr
    # R = ro.r
    from statsmodels.stats.multitest import fdrcorrection

    edgeR = importr("edgeR")
    dfR_exp = py2r(ad.to_df(layer).T)
    vtR_group = py2r(ad.obs[obsKey])
    
    degLs = edgeR.DGEList(counts=dfR_exp,group=vtR_group)
    if filterByEdger:
        keep = edgeR.filterByExpr(degLs)
        degLs = R("""\(x, y) {
            x[y,,keep.lib.sizes=F]
            }""")(degLs, keep)
    degLs = edgeR.calcNormFactors(degLs)
    design = R("""\(x) {
        model.matrix(~0 + x)
    }""")(vtR_group) 
    degLs = edgeR.estimateDisp(degLs,design)
    fit = edgeR.glmFit(degLs,design,)
    # print(contrast)
    # print(design)
    contrast = R.makeContrasts(contrast, levels=design)
    lrt = edgeR.glmLRT(fit, contrast=contrast)
    lrt = R("as.data.frame")(lrt)
    df_res = r2py(R("tibble::rownames_to_column")(lrt)).rename(columns={'rowname':'gene'})
    df_res["fdr"] = fdrcorrection(df_res["PValue"])[1]

    # funcR_callEdgeRContrast = R(
    #     f"""
    # \(dfR_exp,{obsKey}, contrast) {{
    #     y <- DGEList(counts=dfR_exp,group={obsKey})
    #     keep <- filterByExpr(y)
    #     y <- y[keep,,keep.lib.sizes=FALSE]
    #     y <- calcNormFactors(y)
    #     design <- model.matrix(~0+{obsKey})
    #     contrast <- do.call(makeContrasts, list(contrast, levels=design))
    #     y <- estimateDisp(y,design)
    #     fit <- glmFit(y,design,)
    #     lrt <- glmLRT(fit, contrast=contrast)
    #     lrt <- as.data.frame(lrt)
    #     lrt <- tibble::rownames_to_column(data.frame(lrt))
    #     return(lrt)
    # }}
    # """
    # )

    # renv = ro.Environment()
    # with ro.local_context(renv) as rl:
    #     df_lrt = funcR_callEdgeRContrast(dfR_exp, vtR_group, contrast) >> F(
    #         r2py
    #     )
    #     df_lrt["fdr"] = fdrcorrection(df_lrt["PValue"])[1]
    # return df_lrt.copy()
    return df_res

def deByDeseq2(ad, layer, groupDesign, ls_obs, contrast, shrink:Optional[Literal["apeglm", "ashr", "normal"]]=None):
    "coef only worked for apeglm shrink"
    # import rpy2.robjects as ro
    # from rpy2.robjects.packages import importr
    # R = ro.r

    importr("DESeq2")
    renv = ro.Environment()
    assert (shrink=='ashr') or (shrink is None), 'Unsupported shrink method'

    shrink = R("NULL") if shrink is None else shrink
    with ro.local_context(renv) as rl:
        rl['df_mtx'] = py2r(ad.to_df(layer).T)
        rl['df_design'] = py2r(ad.obs[ls_obs])
        rl['group_design'] = groupDesign
        rl['contrast'] = contrast
        rl['shrink'] = shrink
        R("""
        dds <- DESeqDataSetFromMatrix(countData = df_mtx,
                                    colData = df_design,
                                    design = group_design)
        dds <- DESeq(dds)
        res <- results(dds, contrast=contrast)
        if (shrink %>% is.null) {
            invisible()
        } else {
            res <- lfcShrink(dds, type=shrink, res=res)
        }
        
        res <- tibble::rownames_to_column(data.frame(res))
        """)
        res = r2py(rl['res'])
    return res.copy()

def deByDeseq2_old(ad, layer, group_design: str, threads=1):
    '''> For each pair of groups in the `group_design` column, run DESeq2 and store the results in `ad.uns[f"deseq2_{group_design}"]`

    The function is a bit long, but it's not too complicated. 

    The first part of the function is a nested function called `_twoGroup`. This function takes in the `ad`, `layer`, `grp1`, `grp2`, and `group_design` and returns the results of DESeq2. 

    The second part of the function is a loop that loops through all the possible pairs of groups in the `group_design` column. For each pair of groups, it calls the `_twoGroup` function and stores the results in a dictionary. 

    The dictionary is then stored in `ad.uns[f"deseq2_{group_design}"]`

    Parameters
    ----------
    ad
        AnnData object
    layer
        the layer of the AnnData object that you want to use for differential expression analysis.
    group_design : str
        the column name of the column in ad.obs that contains the group information
    threads, optional
        number of threads to use for parallelization

    '''
    logger.warning("Deprecated")
    def _twoGroup(ad, layer, grp1, grp2, group_design):
        importr('DESeq2')
        ad_sub = ad[ad.obs[group_design].isin([grp1, grp2])]
        renv = ro.Environment()
        with ro.local_context(renv) as rl:
            rl['df_mtx'] = py2r(ad_sub.to_df(layer).T)
            rl['df_design'] = py2r(ad_sub.obs[[group_design]])
            rl['group_design'] = group_design
            rl['grp1'] = grp1
            rl['grp2'] = grp2

            R(f"""
            dds <- DESeqDataSetFromMatrix(countData = df_mtx,
                                        colData = df_design,
                                        design = ~ {group_design})
            dds <- DESeq(dds)
            res <- results(dds, contrast=c(group_design, grp1, grp2))
            res <- data.frame(res)
            """)
            res = r2py(rl['res'])
        return (grp1, grp2), res

    ls_groups = []
    ls_group = list(ad.obs['diffxpy_temp'].cat.categories)
    for i, j in product(range(len(ls_group)), range(len(ls_group))):
        if i >= j:
            continue
        grp1 = ls_group[i]
        grp2 = ls_group[j]
        ls_groups.append((grp1, grp2))

    ls_res = Parallel(n_jobs=threads)(delayed(_twoGroup)(ad, layer, grp1, grp2, group_design) for grp1, grp2 in ls_groups)
    dt_res = {x[0]:x[1] for x in ls_res}
    ad.uns[f"deseq2_{group_design}"] = dt_res

    #     ad_sub = ad[ad.obs[group_design].isin([grp1, grp2])]
        
    #     renv = ro.Environment()
    #     with ro.local_context(renv) as rl:
    #         rl['df_mtx'] = py2r(ad_sub.to_df(layer).T)
    #         rl['df_design'] = py2r(ad_sub.obs[[group_design]])
    #         rl['group_design'] = group_design
    #         rl['grp1'] = grp1
    #         rl['grp2'] = grp2

    #         R(f"""
    #         dds <- DESeqDataSetFromMatrix(countData = df_mtx,
    #                                     colData = df_design,
    #                                     design = ~ {group_design})
    #         dds <- DESeq(dds)
    #         res <- results(dds, contrast=c(group_design, grp1, grp2))
    #         res <- data.frame(res)
    #         """)
    #         dt_deseq2[(ls_group[i], ls_group[j])] = r2py(rl['res'])
    # ad.uns[f"deseq2_{group_design}"] = dt_deseq2
    # dt_deseq2 = {}
    # ls_group = ad.obs[group_design].unique().tolist()
    # for i, j in product(range(len(ls_group)), range(len(ls_group))):
    #     if i >= j:
    #         continue
    #     with ro.local_context(renv) as rl:
    #         rl['grp1'] = ls_group[i]
    #         rl['grp2'] = ls_group[j]
    #         R(f"""
    #         res <- DESeq(dds)

    #         res <- results(dds, contrast=c(group_design, grp1, grp2))
    #         res <- data.frame(res)
    #         """)
    #         dt_deseq2[(ls_group[i], ls_group[j])] = r2py(rl['res'])
    # ad.uns[f"deseq2_{group_design}"] = dt_deseq2
