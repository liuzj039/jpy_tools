# from inflect import engine
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import marsilea as ma
import marsilea.plotter as mp
from itertools import product
from functools import reduce
import patchworklib as pw
import seaborn.objects as so
from cycler import cycler
import pandas as pd
import numpy as np
import scipy
from joblib import Parallel, delayed
from loguru import logger


class GeneOverlapBase(object):
    """
    A class for computing overlap statistics between groups of genes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing gene, group, and category information.
    geneKey : str
        The column name for the gene identifier.
    groupKey : str
        The column name for the group identifier.
    categoryKey : str
        The column name for the category identifier.
    categoryTarget : bool, default=True
        Whether the category of interest is the target category.

    Methods:
    --------
    sMod(N, m, n1, n2)
        Computes the probability of observing m or more shared genes between two groups.
    getSmodParameter(df_comp1, df_comp2, geneKey, categoryKey, categoryTarget)
        Computes the parameters needed for sMod.
    getOverlapInfo(ls_groups=None, diagS=1, diagQ=1)
        Computes overlap statistics between all pairs of groups.
    dotplot(sizeKey='m', colorKey='smod', clusterKey='smod', cmap='Reds', size_legend_kws={"title": "Intersect Counts"}, color_legend_kws={"title": "S-MOD"}, addCounts=True, **kwargs)
        Generates a dot plot of overlap statistics.
    tileplot(colorKey='smod', clusterKey='smod', cmap='Reds', label='S-MOD', addCounts=True, **kwargs)
        Generates a tile plot of overlap statistics.
    """
    def __init__(self, df, geneKey, groupKey, categoryKey, categoryTarget=True):
        self.df = df.copy()
        self.geneKey = geneKey
        self.categoryKey = categoryKey
        self.groupKey = groupKey
        self.ls_groups = None
        self.df_overlapInfo = None
        self.categoryTarget = categoryTarget

    @staticmethod
    def sMod(N, m, n1, n2):
        """
        Computes the probability of observing m or more shared genes between two groups.

        Parameters:
        -----------
        N : int
            The total number of shared genes between two groups.
        m : int
            The number of shared genes in the target category.
        n1 : int
            The number of genes in the target category in the first group.
        n2 : int
            The number of genes in the target category in the second group.

        Returns:
        --------
        p : float
            The probability of observing m or more shared genes between two groups.
        """
        import math
        p = 0
        denominator = math.comb(N, n1) * math.comb(N, n2)
        for i in range(m, min(n1, n2) + 1):
            numerator = (
                math.comb(N, i) * math.comb(N - i, n1 - i) * math.comb(N - n1, n2 - i)
            )
            p += numerator / denominator
        if p < 1e-323:
            p = 1e-323
        return p
    
    def getSmodParameter(self, df_comp1, df_comp2, geneKey, categoryKey, categoryTarget):
        """
        Computes the parameters needed for sMod.

        Parameters:
        -----------
        df_comp1 : pandas.DataFrame
            The input dataframe for the first group.
        df_comp2 : pandas.DataFrame
            The input dataframe for the second group.
        geneKey : str
            The column name for the gene identifier.
        categoryKey : str
            The column name for the category identifier.
        categoryTarget : bool
            Whether the category of interest is the target category.

        Returns:
        --------
        N : int
            The total number of shared genes between two groups.
        m : int
            The number of shared genes in the target category.
        n1 : int
            The number of genes in the target category in the first group.
        n2 : int
            The number of genes in the target category in the second group.
        N-n1: int
            The number of genes in the non-target category in the first group.
        n1-n2 : int
            The number of genes in the target category in the first group but not in the second group.
        """
        ls_sharedGenes = list(set(df_comp1[geneKey]) & set(df_comp2[geneKey]))
        N = len(ls_sharedGenes)
        set_n1 = set(df_comp1.query(f"{categoryKey} == @categoryTarget & {geneKey} in @ls_sharedGenes", engine='python')[geneKey])
        set_n2 = set(df_comp2.query(f"{categoryKey} == @categoryTarget & {geneKey} in @ls_sharedGenes", engine='python')[geneKey])
        m = len(set_n1 & set_n2)
        n1 = len(set_n1)
        n2 = len(set_n2)
        N_n1 = N - n1
        n1_n2 = len(set_n1 - set_n2)
        return N, m, n1, n2, N_n1, n1_n2
    
    @staticmethod
    def getOverlapResSingle(comp1, comp2, N, m, n1, n2, N_n1, n1_n2, diagS, diagQ, additionalTest):
        from scipy.stats import chi2_contingency, fisher_exact
        if comp1 == comp2:
            smodP = diagS
            qmod = diagQ
            chi2P = diagS
            fisherP = diagS
        else:
            try:
                smodP = GeneOverlapBase.sMod(N, m, n1, n2)
                if additionalTest:
                    chi2P = chi2_contingency([[N_n1, n1], [n1_n2, m]]).pvalue
                    chi2P = max(chi2P, diagS)
                    fisherP = fisher_exact([[N_n1, n1], [n1_n2, m]]).pvalue
                    fisherP = max(fisherP, diagS)
                else:
                    chi2P = diagS
                    fisherP = diagS
            except:
                logger.error(comp1, comp2, N, m, n1, n2)
                chi2P = 1
                fisherP = 1

            qmod = m / n1 if n1 > 0 else 0
        jaccard = m / (n1 + n2 - m) if (n1 + n2 - m) > 0 else 0
        return comp1, comp2, N, m, n1, n2, N_n1, n1_n2, smodP, jaccard, qmod, chi2P, fisherP


    def getOverlapInfo(self, ls_groups=None, diagS=1e-323, diagQ=1, additionalTest=False, threads=64):
        """
        Computes overlap statistics between all pairs of groups.

        Parameters:
        -----------
        ls_groups : list, default=None
            The list of group identifiers to compute overlap statistics for.
        diagS : float, default=1
            The diagonal value for sMod when comparing a group to itself.
        diagQ : float, default=1
            The diagonal value for qMod when comparing a group to itself.
        """
        from itertools import product
        from tqdm import tqdm
        from scipy.stats import chi2_contingency, fisher_exact

        df = self.df
        geneKey = self.geneKey
        groupKey = self.groupKey
        categoryKey = self.categoryKey
        categoryTarget = self.categoryTarget

        if ls_groups is None:
            ls_groups = df[groupKey].astype('category').cat.categories
        
        ls_res = []
        ls_parameters = []
        for comp1, comp2 in tqdm(product(ls_groups, ls_groups), total=len(ls_groups)**2):
            df_comp1 = df.query(f"{groupKey} == @comp1")
            df_comp2 = df.query(f"{groupKey} == @comp2")
            try:
                N, m, n1, n2, N_n1, n1_n2 = self.getSmodParameter(df_comp1, df_comp2, geneKey, categoryKey, categoryTarget)
                ls_parameters.append([comp1, comp2, N, m, n1, n2, N_n1, n1_n2])
            except:
                assert False, f"Error in {comp1} and {comp2}"
        
        ls_res = Parallel(n_jobs=threads)(delayed(self.getOverlapResSingle)(comp1, comp2, N, m, n1, n2, N_n1, n1_n2, diagS, diagQ, additionalTest) for comp1, comp2, N, m, n1, n2, N_n1, n1_n2 in tqdm(ls_parameters))

            #     if comp1 == comp2:
            #         smodP = diagS
            #         qmod = diagQ
            #         chi2P = diagS
            #         fisherP = diagS
            #     else:
            #         try:
            #             smodP = self.sMod(N, m, n1, n2)
            #             if additionalTest:
            #                 chi2P = chi2_contingency([[N_n1, n1], [n1_n2, m]]).pvalue
            #                 chi2P = max(chi2P, diagS)
            #                 fisherP = fisher_exact([[N_n1, n1], [n1_n2, m]]).pvalue
            #                 fisherP = max(fisherP, diagS)
            #             else:
            #                 chi2P = diagS
            #                 fisherP = diagS
            #         except:
            #             print(comp1, comp2, N, m, n1, n2)
            #             assert False
            #         qmod = m / n1 if n1 > 0 else 0
            #     jaccard = m / (n1 + n2 - m) if (n1 + n2 - m) > 0 else 0

            #     ls_res.append([comp1, comp2, N, m, n1, n2, N_n1, n1_n2, smodP, jaccard, qmod, chi2P, fisherP])
            # except:
            #     assert False, f"Error in {comp1} and {comp2}"
        df_res = pd.DataFrame(ls_res, columns=['comp1', 'comp2', 'N', 'm', 'n1', 'n2', 'N_n1', 'n1_n2', 'smod', 'jaccard', 'qmod', 'chi2P', 'fisherP'])           
        df_res['smod'] = -np.log10(df_res['smod'])
        df_res['chi2'] = -np.log10(df_res['chi2P'])
        df_res['fisher'] = -np.log10(df_res['fisherP'])
        df_res['comp1'] = df_res['comp1'].astype('category').cat.set_categories(ls_groups)
        df_res['comp2'] = df_res['comp2'].astype('category').cat.set_categories(ls_groups)
        self.ls_groups = ls_groups
        self.df_overlapInfo = df_res
    
    def dotplot(self, sizeKey='m', colorKey='smod', clusterKey='smod', cmap='Reds', size_legend_kws={"title": "Intersect Counts"}, color_legend_kws={"title": "S-MOD"}, dt_labelKwargs={},addCounts=True, barColor='Red', **kwargs):
        """
        Generates a dot plot of overlap statistics.

        Parameters:
        -----------
        sizeKey : str, default='m'
            The column name for the size of the dots.
        colorKey : str, default='smod'
            The column name for the color of the dots.
        clusterKey : str, default='smod'
            The column name for the cluster data.
        cmap : str, default='Reds'
            The colormap to use.
        size_legend_kws : dict, default={"title": "Intersect Counts"}
            Keyword arguments for the size legend.
        color_legend_kws : dict, default={"title": "S-MOD"}
            Keyword arguments for the color legend.
        addCounts : bool, default=True
            Whether to add a bar plot of the counts.
        """
        # size_legend_kws={'title': 'Intersect Counts', 'show_at':np.interp([1, 50, 100, 200, 400, df_intersectCounts.max().max()], [df_intersectCounts.min().min(), df_intersectCounts.max().max()], [0,1]), 
        #                         'spacing':'uniform'}
        df_overlapInfo = self.df_overlapInfo
        df_size = df_overlapInfo.pivot_table(index='comp1', columns='comp2', values=sizeKey).fillna(0)
        df_color = df_overlapInfo.pivot_table(index='comp1', columns='comp2', values=colorKey).fillna(0) 
        df_cluster = df_overlapInfo.pivot_table(index='comp1', columns='comp2', values=clusterKey).fillna(0)
        h = ma.SizedHeatmap(
            df_size.values, color=df_color.values, cmap=cmap, color_legend_kws=color_legend_kws, cluster_data=df_cluster.values,
            size_legend_kws=size_legend_kws, **kwargs
        )
        h.add_left(
            mp.Labels(df_size.index, **dt_labelKwargs), pad=0.1
        )
        h.add_top(
            mp.Labels(df_size.index, **dt_labelKwargs), pad=0.1
        )

        if addCounts:
            ls_order = df_color.index
            df_counts = df_overlapInfo.query("comp1 == comp2").set_index('comp1').loc[ls_order, 'm'].to_frame().T
            h.add_top(
                mp.Bar(df_counts, color=barColor, label='Counts'), size=1, pad=0.1
            )

        h.add_legends()
        return h
    
    def tileplot(self, colorKey='smod', clusterKey='smod', cmap='Reds', label='S-MOD', addCounts=True, barColor='Red', dt_labelKwargs={}, **kwargs):
        """
        Generates a tile plot of overlap statistics.

        Parameters:
        -----------
        colorKey : str, default='smod'
            The column name for the color of the tiles.
        clusterKey : str, default='smod'
            The column name for the cluster data.
        cmap : str, default='Reds'
            The colormap to use.
        label : str, default='S-MOD'
            The label for the colorbar.
        addCounts : bool, default=True
            Whether to add a bar plot of the counts.
        """
        df_overlapInfo = self.df_overlapInfo
        df_color = df_overlapInfo.pivot_table(index='comp1', columns='comp2', values=colorKey).fillna(0) 
        df_cluster = df_overlapInfo.pivot_table(index='comp1', columns='comp2', values=clusterKey).fillna(0)
        h = ma.Heatmap(
            df_color.values, cmap=cmap, cluster_data=df_cluster.values, label=label,
            **kwargs
        )
        h.add_left(
            mp.Labels(df_color.index, **dt_labelKwargs), pad=0.1
        )
        h.add_top(
            mp.Labels(df_color.index, **dt_labelKwargs), pad=0.1
        )
        if addCounts:
            ls_order = df_color.index
            df_counts = df_overlapInfo.query("comp1 == comp2").set_index('comp1').loc[ls_order, 'm'].to_frame().T
            h.add_top(
                mp.Bar(df_counts, color=barColor, label='Counts'), size=1, pad=0.1
            )

        h.add_legends()
        return h
