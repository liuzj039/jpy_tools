import pandas as pd
import scanpy as sc
import pysam
import tqdm
from loguru import logger
from typing import Literal, List, Optional, Union
import pyranges as pr
import matplotlib.pyplot as plt

import seaborn as sns
from marsilea.base import RenderPlan
import marsilea as ma
import marsilea.plotter as mp
import numpy as np

geneRepBedPath = (
    "/data/Zhaijx/liuzj/data/Araport11/araport11.representative.gene_model.bed"
)

# 0: Intron, 1: Exon, -1: Uncovered, 2: PolyA
palette = {0: "#FFFFFF", 1: "#3982b5", -1: "#FFFFFF", 2: "#ee616f"}


# class ReadHeatmap(ma.CatHeatmap):
#     def __init__(self, data, reverse, palette=palette, *args, **kwargs):
#         if reverse:
#             super().__init__(data=np.flip(data, 1), palette=palette, *args, **kwargs)
#         else:
#             super().__init__(data=data, palette=palette, *args, **kwargs)


class ReadColor(mp.Colors):
    def __init__(self, data, reverse, palette=palette, *args, **kwargs):
        if reverse:
            data = np.flip(data, 1)

        super().__init__(data=data, palette=palette, *args, **kwargs)

def processRow(row, number, position='first'):
    # Initialize a row of zeros with the same length as the input row
    new_row = np.zeros_like(row)
    try:
        if position == 'first':
            # Find the first occurrence of the number
            index = row.tolist().index(number)
        else:
            # Find the last occurrence (reverse the row for searching)
            index = len(row) - row.tolist()[::-1].index(number) - 1
        # Set the found position to 1
        new_row[index] = 1
    except ValueError:
        assert False, f"Number {number} not found in row {row}"
    return new_row

def setPosition2one(df, number, position='first'):
    """
    For each row in the DataFrame, finds the first or last occurrence of a specified number.
    The found position is set to 1, while all other elements are set to 0.
    If the number is not found and fill_others_as_zero is True, the entire row is set to 0.

    Parameters:
    - df: pandas DataFrame to process.
    - number: The number to search for in each row.
    - position: 'first' for the first occurrence, 'last' for the last occurrence.
    - fill_others_as_zero: If True, rows where the number is not found will be filled with 0.

    Returns:
    - A new pandas DataFrame with modified rows based on the search criteria.
    """
    
    # Apply the process_row function to each row in the DataFrame
    new_df = df.apply(processRow, number=number, position=position, axis=1, result_type='expand')
    return new_df

class ReadMeta(RenderPlan):
    def __init__(self, data, reverse, sr_readStrandInGenome, type='coverage', color='black', linewidths=1, foundNum=1):
        assert type in ['coverage', '3', '5'], f"Invalid type: {type}"
        title = {
            'coverage': 'Coverage',
            '3': "3' End",
            '5': "5 ' End"
        }[type]
        # if reverse:
        #     data = np.flip(data, 1)
        df = pd.DataFrame(data)
        if type == 'coverage':
            df = (df == foundNum)
            sr = df.sum(axis=0)
        else:
            sr = pd.Series(np.zeros(df.shape[1]))
            for strand, (i, row) in zip(sr_readStrandInGenome, df.iterrows()):
                if (strand == '+') ^ (type == '3'):
                    index = np.where(row == foundNum)[0][0]
                else:
                    index = np.where(row == foundNum)[0][-1]
                sr[index] += 1
        if reverse:
            data = np.flip(data, 1)
            sr = sr[::-1]


        # elif reverse ^ (type == '3'):
        #     df = setPosition2one(df, 1, position='first')
        #     sr = df.sum(axis=0)
        # else:
        #     df = setPosition2one(df, 1, position='last')
        #     sr = df.sum(axis=0)
        self.set_data(data)
        self.processedCov = sr
        self.color = color
        self.linewidths = linewidths
        self.title = title
    
    def render_ax(self, spec):
        ax = spec.ax
        data = spec.data
        sr = self.processedCov
        ax.plot([x+0.5 for x in range(len(sr.index))], sr.values, color=self.color, linewidth=self.linewidths)
        ax.set_xlim(0, len(sr.index))
        ax.set_title(self.title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_major_locator(plt.NullLocator())

def findNumberIntervalsOpen(sequence, target=0):
    sequence = list(sequence)
    intervals = []
    start = None
    for i, num in enumerate(sequence):
        if num == target and start is None:
            start = i
        elif num != target and start is not None:
            intervals.append((start, i))
            start = None
    # Handle case where sequence ends with 0
    if start is not None:
        intervals.append((start, len(sequence)))
    return intervals


class GeneLine(RenderPlan):
    render_main = True

    def __init__(self, data, reverse, color="black", linewidths=1):
        if reverse:
            self.set_data(np.flip(data, 1))
        else:
            self.set_data(data)

        self.color = color
        self.linewidths = linewidths

    def render_ax(self, spec):
        ax = spec.ax
        data = spec.data
        # df = pd.DataFrame(np.where(data == 1, )).T.groupby(0)[1].agg(['min', 'max'])
        sr = pd.DataFrame(data).apply(findNumberIntervalsOpen, axis=1).explode()
        ax.hlines(
            sr.index + 0.5,
            sr.str[0],
            sr.str[1],
            colors=self.color,
            linewidths=self.linewidths,
        )
        # ax.hlines(df.index+0.5, df['min'], df['max'], colors=self.color, linewidths=1)

class LibrarySplitLine(RenderPlan):
    def __init__(self, data,  color="black", linewidths=2):
        self.set_data(data)
        self.color = color
        self.linewidths = linewidths

    def render_ax(self, spec):
        ax = spec.ax
        data = spec.data
        ax.axhline(y=0, linewidth=self.linewidths, color=self.color)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_major_locator(plt.NullLocator())     

class GeneSplitLine(RenderPlan):
    render_main = True

    def __init__(self, data, color="white", linewidth=0.1):
        self.set_data(data)
        self.color = color
        self.linewidth = linewidth

    def render_ax(self, spec):
        ax = spec.ax
        data = spec.data
        for i in range(data.shape[0]):
            ax.axhline(y=i, linewidth=self.linewidth, color=self.color)

   

class GeneStruct(RenderPlan):
    def __init__(
        self,
        data,
        geneName,
        geneStrand,
        reverse,
        color="black",
        exonwidth=9,
        intronwidth=1,
        utrwidth=5,
        geneNameSize=16,
        xticksMethod="gene",
        xticksSpacing=1000,
        firstPosInGenome=None,
    ):
        if reverse:
            self.set_data(np.flip(data, 1))
            self.geneStrand = "+" if geneStrand == "-" else "-"
        else:
            self.set_data(data)
            self.geneStrand = geneStrand
        self.color = color
        self.exonwidth = exonwidth
        self.intronwidth = intronwidth
        self.utrwidth = utrwidth
        self.geneName = geneName
        self.geneNameSize = geneNameSize
        self.xticksMethod = xticksMethod
        self.xticksSpacing = xticksSpacing
        self.firstPosInGenome = firstPosInGenome

    def render_ax(self, spec):
        from matplotlib import ticker

        ax = spec.ax
        data = spec.data
        data = pd.DataFrame(data)
        assert data.shape[0] == 1, "Unsupported data shape"
        sr = data.iloc[0]
        ls_exon = findNumberIntervalsOpen(sr, 1)
        ax.hlines(
            [0] * len(ls_exon),
            [x[0] for x in ls_exon],
            [x[1] for x in ls_exon],
            colors=self.color,
            linewidths=self.exonwidth,
        )
        ls_intron = findNumberIntervalsOpen(sr, 0)
        ax.hlines(
            [0] * len(ls_intron),
            [x[0] for x in ls_intron],
            [x[1] for x in ls_intron],
            colors=self.color,
            linewidths=self.intronwidth,
        )
        ls_utr = findNumberIntervalsOpen(sr, 2)
        ax.hlines(
            [0] * len(ls_utr),
            [x[0] for x in ls_utr],
            [x[1] for x in ls_utr],
            colors=self.color,
            linewidths=self.utrwidth,
        )

        if len(ls_utr) == 0:
            ls_forMark = ls_exon
        else:
            ls_forMark = ls_utr

        if self.geneStrand == "+":
            pos = ls_forMark[0][0]
            pos = ax.transData.transform((pos, 0))[0]
            pos += 5000
            pos = ax.transData.inverted().transform((pos, 0))[0]
            ax.vlines(pos, 0, 0.8, colors=self.color, linewidth=1)

            rightPos = ax.transData.transform((pos, 0))[0]
            rightPos += 500000
            rightPos = ax.transData.inverted().transform((rightPos, 0))[0]

            ax.hlines(0.8, ls_forMark[0][0], rightPos, colors=self.color, linewidth=1)

            # add direction
            rightPos = ax.transData.transform((rightPos, 0))[0]
            rightPos += 100000
            rightPos = ax.transData.inverted().transform((rightPos, 0))[0]

            ax.annotate(
                "",
                (rightPos, 0.8),
                xytext=(ls_forMark[0][0], 0.8),
                ha="right",
                arrowprops=dict(
                    arrowstyle="-|>", connectionstyle="angle", color=self.color
                ),
            )

        else:
            pos = ls_forMark[-1][-1]
            pos = ax.transData.transform((pos, 0))[0]
            pos -= 5000
            pos = ax.transData.inverted().transform((pos, 0))[0]
            ax.vlines(pos, 0, 0.8, colors=self.color, linewidth=1)

            leftPos = ax.transData.transform((pos, 0))[0]
            leftPos -= 500000
            leftPos = ax.transData.inverted().transform((leftPos, 0))[0]

            ax.hlines(0.8, leftPos, ls_forMark[-1][-1], colors=self.color, linewidth=1)

            # add direction
            leftPos = ax.transData.transform((leftPos, 0))[0]
            leftPos -= 100000
            leftPos = ax.transData.inverted().transform((leftPos, 0))[0]

            ax.annotate(
                "",
                (leftPos, 0.8),
                xytext=(ls_forMark[-1][-1], 0.8),
                ha="right",
                arrowprops=dict(
                    arrowstyle="-|>", connectionstyle="angle", color=self.color
                ),
            )

        ax.text(
            (ls_forMark[-1][-1] + ls_forMark[0][0]) / 2,
            0.8,
            self.geneName,
            ha="center",
            va="bottom",
            fontsize=self.geneNameSize,
            fontdict=dict(style="italic"),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_xlim(0, len(sr))
        ax.set_ylim(-1, 1)

        if self.xticksMethod is None:
            ax.spines["bottom"].set_visible(False)
            ax.xaxis.set_major_locator(plt.NullLocator())
        elif self.xticksMethod == "gene":
            if self.geneStrand == "+":
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(self.xticksSpacing, offset=ls_forMark[0][0])
                )
                ls_text = ax.get_xticklabels()
                _ls_text = []
                for text in ls_text:
                    label = text.get_text().replace("−", "-")
                    text.set_text(int(label) - ls_forMark[0][0])
                    _ls_text.append(text)
                ax.set_xticklabels(_ls_text)
            else:
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(
                        self.xticksSpacing, ls_forMark[-1][-1] % self.xticksSpacing
                    )
                )
                ls_text = ax.get_xticklabels()
                _ls_text = []
                for text in ls_text:
                    label = text.get_text().replace("−", "-")
                    text.set_text(ls_forMark[-1][-1] - int(label))
                    _ls_text.append(text)
                ax.set_xticklabels(_ls_text)

        elif self.xticksMethod == "genome":
            assert False, "Not implemented"


def bamToBinary(
    bamPath,
    df_bed,
    gene=None,
    chromosome=None,
    start=None,
    end=None,
    strand=None,
    # reverseStrand=None,
    cutRead=False,
    geneTag="ZG",
    polyaTag="ZP",
    typeTag="ZT",
    padding=500,
    sameOriOnly: Optional[bool] = None,
    filterGene: Optional[bool] = None,
    addPolya=True,
    minPolya=5,
) -> sc.AnnData:
    """
    ### Documentation for `bamToBinary` Function

    The `bamToBinary` function converts reads from a BAM file associated with a specific gene into a binary representation based on their alignment to exon regions, optionally including PolyA tail information. This representation can be used for further analysis in single-cell RNA sequencing data analysis pipelines. The function outputs an AnnData object.

    #### Parameters:

    - `gene` (str, optional): The name of the gene to filter reads. If provided, only reads associated with this gene are considered.
    - `bamPath` (str): Path to the BAM file containing read alignments.
    - `df_bed` (DataFrame): A DataFrame containing gene annotations, typically derived from a BED file. Must include columns for chromosome, start, end, and name, with an optional 'Gene' column.
    - `chromosome` (str, optional): The chromosome to fetch reads from. If not specified, it is inferred from the gene's annotation.
    - `start` (int, optional): The start position for fetching reads. If not specified, it is inferred from the gene's annotation.
    - `end` (int, optional): The end position for fetching reads. If not specified, it is inferred from the gene's annotation.
    - `strand` (str, optional): The strand of the gene. If not specified, it is inferred from the gene's annotation.
    - `reverseStrand` (bool, optional): Whether to consider the reverse strand. If not specified, it is inferred from the gene's strand.
    - `cutRead` (bool, optional): If True, trims reads to the specified gene region. Otherwise, uses the full read alignment.
    - `geneTag` (str, optional): The BAM tag used to associate reads with genes.
    - `polyaTag` (str, optional): The BAM tag used to indicate the length of the PolyA tail.
    - `typeTag` (str, optional): The BAM tag used to indicate the type of read.
    - `padding` (int, optional): The number of base pairs to extend beyond the start and end of the gene region when fetching reads.
    - `sameOriOnly` (bool, optional): If True, only considers reads that match the orientation of the gene.
    - `filterGene` (bool, optional): If True, filters reads to include only those associated with the specified gene.
    - `addPolya` (bool, optional): If True, includes PolyA tail information in the binary representation.
    - `minPolya` (int, optional): The minimum length of a PolyA tail to be considered significant.

    #### Returns:
    - `sc.AnnData`: An AnnData object containing the binary representation of the read alignments and their annotations. The main data matrix (`X`) has rows representing reads and columns representing positions, with values indicating the presence of a read (1), absence (-1), or presence of a PolyA tail (2).

    #### Usage Notes:
    - The function requires the `pysam` library for reading BAM files and the `scanpy` library for creating and working with the AnnData object.
    - Users should ensure that the input DataFrame (`df_bed`) is correctly formatted, with appropriate columns for gene annotations.
    - The binary representation created by this function can be used for downstream visualization.
    """
    if "Gene" in df_bed.columns:
        pass
    else:
        df_bed["Gene"] = df_bed["Name"].str.split(".").str[0]

    if not gene is None:
        sr_gene = df_bed.query("Gene == @gene", engine="python").iloc[0]
        filterGene = True if filterGene is None else filterGene
        sameOriOnly = True if sameOriOnly is None else sameOriOnly
        chromosome = sr_gene["Chromosome"]
        start = sr_gene["Start"]
        end = sr_gene["End"]
        strand = sr_gene["Strand"]
        # reverseStrand = strand == "-" if reverseStrand is None else reverseStrand
    else:
        filterGene = False
        sameOriOnly = False if sameOriOnly is None else sameOriOnly
        if not padding is None:
            logger.warning("Padding is not None.")

    psm = pysam.AlignmentFile(bamPath, "rb")
    ls_read = list(psm.fetch(chromosome, start - padding, end + padding))
    
    if filterGene:
        _ls_read = []
        for x in ls_read:
            if not x.has_tag(geneTag):
                continue
            if x.get_tag(geneTag) != gene:
                continue
            _ls_read.append(x)
        ls_read = _ls_read
        del _ls_read
    psm.close()
    if sameOriOnly:
        if strand == "+":
            ls_read = [read for read in ls_read if not read.is_reverse]
        else:
            ls_read = [read for read in ls_read if read.is_reverse]

    # transfer read to a list. if the position is coverd by the read, set 1, otherwise 0, consider intron
    # 0 : intron, 1: Exon, -1: Uncovered, 2: PolyA
    ls_posBinary = []
    ls_comment = []
    minLeft = start - padding
    maxRight = end + padding

    for read in tqdm.tqdm(ls_read):
        ls_blocks = read.get_blocks()
        ls_binary = [0] * (ls_blocks[-1][-1] - ls_blocks[0][0])
        currentMinPos = ls_blocks[0][0]
        currentMaxPos = ls_blocks[-1][-1]

        for block in ls_blocks:
            ls_binary[block[0] - ls_blocks[0][0] : block[1] - ls_blocks[0][0]] = [1] * (
                block[1] - block[0]
            )

        sr_binary = pd.Series(
            ls_binary, index=range(ls_blocks[0][0], ls_blocks[-1][-1])
        )

        if addPolya:
            if read.has_tag(polyaTag):
                readPolya = read.get_tag(polyaTag)
                if readPolya >= minPolya:
                    if read.is_reverse:
                        currentMinPos = ls_blocks[0][0] - readPolya
                        sr_polya = pd.Series(
                            [2] * readPolya, index=range(currentMinPos, ls_blocks[0][0])
                        )
                        sr_binary = pd.concat([sr_polya, sr_binary])
                    else:
                        currentMaxPos = ls_blocks[-1][-1] + readPolya
                        sr_polya = pd.Series(
                            [2] * readPolya,
                            index=range(ls_blocks[-1][-1], currentMaxPos),
                        )
                        sr_binary = pd.concat([sr_binary, sr_polya])
        minLeft = currentMinPos if currentMinPos < minLeft else minLeft
        maxRight = currentMaxPos if currentMaxPos > maxRight else maxRight


        name = read.query_name
        blockStart = ls_blocks[0][0]
        blockEnd = ls_blocks[-1][-1]
        if read.has_tag(typeTag):
            readType = read.get_tag(typeTag)
        else:
            readType = "Unknown"
        readStrandInGenome = '-' if read.is_reverse else '+'
        readGene = None
        if read.has_tag(geneTag):
            readGene = read.get_tag(geneTag)
        sr_comment = pd.Series(
            {"name": name, "start": blockStart, "end": blockEnd, "type": readType, 'strandInGenome': readStrandInGenome, 'gene':readGene}
        )

        ls_posBinary.append(sr_binary)
        ls_comment.append(sr_comment)
    if cutRead:
        ad = sc.AnnData(
            X=pd.DataFrame(ls_posBinary, columns=range(start - padding, end + padding))
            .fillna(-1)
            .astype(int)
        )
    else:
        ad = sc.AnnData(
            X=pd.DataFrame(ls_posBinary, columns=range(minLeft, maxRight))
            .fillna(-1)
            .astype(int)
        )
    # ad.var.index = ad.var.index.astype(int)
    ad.obs = pd.DataFrame(ls_comment)
    ad.obs = ad.obs.set_index("name")

    return ad


def geneBedToBinary(ad, df_bed, gene=None, chromosome=None, start=None, end=None):
    """
    This Python function, `getGeneStrucBinary`, is designed for processing genetic data, specifically to map the structure of genes within a given range or a specific gene in a binary format. It takes in an AnnData object (`ad`), a pandas DataFrame containing gene annotation data (`df_bed`), and optional parameters specifying a gene name, chromosome, start, and end positions. The function updates the `ad` object with gene structure information where regions are marked as -1 (uncovered), 1 (exon), 0 (intron), or 2 (UTR - untranslated regions).

    ### Parameters:
    - `ad`: AnnData object to be updated with gene structure information. AnnData is a convention for storing large, annotated matrices. It is commonly used in bioinformatics and computational biology.
    - `df_bed`: pandas DataFrame containing gene annotation data. It is expected to have columns like `Chromosome`, `Start`, `End`, `Gene`, `Strand`, `BlockStarts`, and `BlockSizes` which are typical of BED file format.
    - `gene` (optional): The name of the specific gene to be processed. If provided, the function focuses on this gene only.
    - `chromosome` (optional): The chromosome number or name where the gene(s) is located. Required if `start` and `end` are provided.
    - `start` (optional): The start position of the chromosome region to be considered. Must be provided alongside `chromosome` and `end`.
    - `end` (optional): The end position of the chromosome region to be considered. Must be provided alongside `chromosome` and `start`.

    ### Workflow:
    1. **Gene Selection**: If a `gene` name is provided, the function filters `df_bed` for this specific gene. If `gene` is not provided, it assumes a genomic region is specified by `chromosome`, `start`, and `end` and selects genes within this region.
    2. **Structure Mapping**: For each selected gene, the function iterates through its structural elements as defined in the `df_bed` DataFrame, updating the AnnData object with binary indicators for different gene regions (exon, intron, UTR, or uncovered).
    3. **Region Marking**:
        - Initializes all positions as -1 (uncovered).
        - Marks intron regions as 0.
        - Marks exon regions as 1, based on `BlockStarts` and `BlockSizes`.
        - Marks UTR regions as 2, using `ThickStart` and `ThickEnd` positions.

    ### Usage Notes:
    - Ensure that the AnnData object (`ad`) and the DataFrame (`df_bed`) are correctly formatted and contain the required columns.
    - The function modifies the AnnData object in place, adding binary gene structure annotations to it.
    - This function can be used for gene structure analysis in genomic studies, especially when focusing on the distribution and characteristics of exons, introns, and untranslated regions within specific genes or genomic regions.
    """
    if not gene is None:
        configName = gene
    else:
        configName = f"{chromosome}:{start}-{end}"

    ad.var.index = ad.var.index.astype(int)

    if gene is None:
        df_geneStruc = df_bed.query(
            "Chromosome == @chromosome & End >= @start & Start < @end", engine="python"
        )
    else:
        sr_gene = df_bed.query("Gene == @gene", engine="python").iloc[0]
        df_geneStruc = pd.DataFrame(sr_gene).T

        # print(df_geneStruc)

    for _, sr_gene in df_geneStruc.iterrows():
        geneStruName = sr_gene["Gene"]
        geneStruStrand = sr_gene["Strand"]

        # -1 uncovered; 1: exon, 0:intron, 2: utr
        ad.var[geneStruName] = -1

        ad.var[geneStruName].loc[sr_gene["Start"] : sr_gene["End"]] = 0

        for blockStart, size in zip(
            sr_gene["BlockStarts"].split(","), sr_gene["BlockSizes"].split(",")
        ):
            if blockStart is "":
                continue
            blockStart = int(blockStart) + sr_gene["Start"]
            size = int(size)
            ad.var[geneStruName].loc[blockStart : blockStart + size] = 1

        ad.var[geneStruName].loc[sr_gene["Start"] : sr_gene["ThickStart"]] = 2
        ad.var[geneStruName].loc[sr_gene["ThickEnd"] : sr_gene["End"]] = 2

        ad.uns[geneStruName] = sr_gene.to_dict()
    ad.uns[f"{configName}_genes"] = df_geneStruc['Gene'].tolist()

    ad.var.index = ad.var.index.astype(str)


class Igv(object):

    def __init__(
        self,
        df_bed,
        reverseStrand=None,
        cutRead=False,
        geneTag="ZG",
        polyaTag="ZP",
        typeTag="ZT",
        padding=500,
        sameOriOnly: Optional[bool] = None,
        filterGene: Optional[bool] = None,
        addPolya=True,
        minPolya=5,
        palette=palette,
    ):
        self.df_bed = df_bed
        self.reverseStrand = reverseStrand
        self.cutRead = cutRead
        self.geneTag = geneTag
        self.polyaTag = polyaTag
        self.typeTag = typeTag
        self.padding = padding
        self.sameOriOnly = sameOriOnly
        self.filterGene = filterGene
        self.addPolya = addPolya
        self.minPolya = minPolya
        self.palette = palette
        self.dtBamPath = {}
        self.dtAd = {}
    
    def addLibrary(self, name, bamPath):
        self.dtBamPath[name] = bamPath
    
    def synAd(self, configName, ls_name=None):
        if ls_name is None:
            ls_name = list(self.dtBamPath.keys())
        
        _dt_ad = {}
        for name in ls_name:
            _dt_ad[name] = self.dtAd[f"{name}_{configName}"]
        self.dtAd[configName] = sc.concat(_dt_ad, join="outer", index_unique='-', fill_value=-1, label='library', uns_merge='same')
        for geneAnno in _dt_ad[name].var.columns:
            self.dtAd[configName].var[geneAnno] = _dt_ad[name].var[geneAnno].fillna(-1).astype(int)

    def addReadBinary(self, ls_name=None, gene=None, chromosome=None, start=None, end=None, subsampleTo=None):
        if not gene is None:
            configName = gene
        else:
            configName = f"{chromosome}:{start}-{end}"
        
        if ls_name is None:
            ls_name = self.dtBamPath.keys()
        
        for name in ls_name:
            ad = bamToBinary(
                self.dtBamPath[name],
                self.df_bed,
                gene=gene,
                chromosome=chromosome,
                start=start,
                end=end,
                cutRead=self.cutRead,
                geneTag=self.geneTag,
                polyaTag=self.polyaTag,
                typeTag=self.typeTag,
                padding=self.padding,
                sameOriOnly=self.sameOriOnly,
                filterGene=self.filterGene,
                addPolya=self.addPolya,
                minPolya=self.minPolya,
            )
            if not subsampleTo is None:
                ad = ad[np.random.choice(ad.obs.index, subsampleTo, replace=False)].copy()
            geneBedToBinary(ad, self.df_bed, gene=gene, chromosome=chromosome, start=start, end=end)
            self.dtAd[f"{name}_{configName}"] = ad
        
        self.synAd(configName, ls_name)
        # _dt_ad = {}
        # for name in ls_name:
        #     _dt_ad[name] = self.dtAd[f"{name}_{configName}"]
        # self.dtAd[configName] = sc.concat(_dt_ad, join="outer", index_unique='-', fill_value=-1, label='library', uns_merge='same')
        # for geneAnno in _dt_ad[name].var.columns:
        #     self.dtAd[configName].uns[geneAnno] = _dt_ad[name].uns[geneAnno].fillna(-1).astype(int)

    def filterSplitSort(self, groupby, *, gene=None, chromosome=None, start=None, end=None, ls_name=None, sortby=None, ascending=True, ls_group=None, dt_color=None, ls_addPolyaGroup=None):
        if not gene is None:
            configName = gene
        else:
            configName = f"{chromosome}:{start}-{end}" 
        
        if ls_name is None:
            ls_name = list(self.dtBamPath.keys())
        
        if isinstance(ls_name, str):
            ls_name = [ls_name]
        
        
        ls_groupOrg = ls_group
        _ls_group = None
        for library in ls_name:
            ad = self.dtAd[f"{library}_{configName}"]
            ad._sanitize()

            if ls_groupOrg is None:
                ls_group = ad.obs[groupby].cat.categories.tolist()
            elif isinstance(ls_groupOrg, str):
                ls_group = [ls_groupOrg]
            else:
                ls_group = ls_groupOrg

            if _ls_group is None:
                _ls_group = ls_group
            else:
                assert _ls_group == ls_group, f"Group not match: {library}"

            ad.obs[groupby] = ad.obs[groupby].cat.set_categories(ls_group)
            if dt_color is None:
                dt_color = {group: color for group, color in zip(ls_group, sns.color_palette('Paired', n_colors=len(ls_group)).as_hex())}
            
            ad = ad[ad.obs[groupby].isin(ls_group)]
            dt_numChange = {
                x:i for i, x in enumerate(ls_group, 10)
            }
            dt_readId2Group = ad.obs[groupby].to_dict()
                
            _ls = []
            for readId, sr in tqdm.tqdm(ad.to_df().iterrows(), total=ad.shape[0]):
                sr = sr.map(lambda _: dt_numChange[dt_readId2Group[readId]] if _ == 1 else _)
                if not ls_addPolyaGroup is None:
                    if dt_readId2Group[readId] in ls_addPolyaGroup:
                        pass
                    else:
                        sr = sr.map(lambda _: -1 if _ == 2 else _)
                # print(sr.unique())
                _ls.append(sr)
            df_changeNum = pd.DataFrame(_ls)
            ad.layers[groupby] = df_changeNum.reindex(ad.obs.index)
            if sortby is None:
                pass
            else:
                df_sorted = ad.obs.sort_values(by=sortby, ascending=ascending)
                ad = ad[df_sorted.index]
            ad.uns['dt_group2code'] = dt_numChange
            ad.uns['dt_group2color'] = dt_color
            self.dtAd[f"{library}_{configName}"] = ad

        self.synAd(configName, ls_name)
        # _dt_ad = {}
        # for name in ls_name:
        #     _dt_ad[name] = self.dtAd[f"{name}_{configName}"]
        #     _dt_ad[name].uns['dt_group2code'] = dt_numChange
        #     _dt_ad[name].uns['dt_code2color'] = dt_color
        #     self.dtAd[f"{name}_{configName}"] = _dt_ad[name]

        # self.dtAd[configName] = sc.concat(_dt_ad, join="outer", index_unique='-', fill_value=-1, label='library', uns_merge='same')

    def addReadPlotter(self, configName, library:Union[None, str], h: ma.base.ClusterBoard, reverse, intronWidth=1, geneSplitLineWidth=1, groupby=None, libraryColor=None,):
        if library is None:
            ad = self.dtAd[configName]
        else:
            ad = self.dtAd[f"{library}_{configName}"]

        if groupby is None:
            palette = self.palette
        else:
            assert not configName is None, "configName must be provided when groupby is not None"
            dt_splitColors = ad.uns['dt_group2color']
            dt_group2code = ad.uns['dt_group2code']
            dt_code2color = {y: dt_splitColors[x] for x, y in dt_group2code.items()}
            palette = self.palette
            palette.update(dt_code2color)        

        if groupby is None:
            h.add_layer(ReadColor(ad.X, reverse, palette=palette), legend=False)
        else:
            h.add_layer(ReadColor(ad.layers[groupby], reverse, palette=palette), legend=False)
            h.hsplit(labels=ad.obs[groupby], order=ad.obs[groupby].cat.categories.tolist())
            h.add_left(mp.Colors(ad.obs[groupby], palette=dt_splitColors), size=0)

        h.add_layer(GeneLine(ad.X, reverse, linewidths=intronWidth))
        if geneSplitLineWidth > 0:
            h.add_layer(GeneSplitLine(ad.X, linewidth=geneSplitLineWidth))
        if not libraryColor is None:
            if groupby is None:
                ratio = [1]
            else:
                ratio = [len(ad.obs[groupby].cat.categories)]
            h.add_left(mp.FixedChunk(
                [library], fill_colors=libraryColor, ratio=ratio
            ))
        return h

    def addGeneStructurePlotter(self, h: ma.base.ClusterBoard, reverse, configName, pad=0.8):
        
        ad = self.dtAd[configName]
        ls_genes = ad.uns[f"{configName}_genes"]
        if reverse:
            ls_genes = ls_genes[::-1]
        for gene in ls_genes:
            h.add_top(GeneStruct(ad.var[[gene]].T.values, gene, ad.uns[gene]['Strand'], reverse, color='black'), pad=pad, size=0.4)
        
        return h
        

    def makePlotTemplate(self, ls_name=None, *, gene=None, chromosome=None, start=None, end=None,
                   width=10, height=5, dt_libraryColor=None, addGeneStruAll=False, groupby=None, 
                   geneSplitLineWidth=0.2, intronWidth=1, addMetaInfo=True, dt_metaKwargs={'type':'3'}) -> ma.base.ClusterBoard:
        if not gene is None:
            configName = gene
        else:
            configName = f"{chromosome}:{start}-{end}"

        if ls_name is None:
            ls_name = list(self.dtBamPath.keys())
        
        if isinstance(ls_name, str):
            ls_name = [ls_name]
        
        if dt_libraryColor is None:
            dt_libraryColor = {name: '#d6ece1' for name in ls_name}
        
        
        assert configName in self.dtAd, f"{configName} not in self.dtAd"

        if groupby is None:
            pass
        else:
            assert groupby in self.dtAd[configName].layers, f"please run `filterSplitSort` first"

        if len(ls_name) == 1:
            library = ls_name[0]
            ad = self.dtAd[f"{library}_{configName}"]
            ad.obs['library'] = library
        else:
            ad = self.dtAd[configName]
        
        ad_all = ad[ad.obs['library'].isin(ls_name)]

        dt_plotter = {}
        addGeneStru = True

        if not gene is None:
            geneStrand = ad.uns[gene]['Strand']
            if self.reverseStrand is None:
                reverseStrand = geneStrand == "-"
        else:
            reverseStrand = False if self.reverseStrand is None else self.reverseStrand 

        for library in ls_name:
            ad = ad_all[ad_all.obs['library'] == library]
            h = ma.base.ClusterBoard(ad.X, width=width, height=height)
            h = self.addReadPlotter(configName, library, h, reverseStrand, intronWidth=intronWidth, geneSplitLineWidth=geneSplitLineWidth, groupby=groupby, libraryColor=dt_libraryColor[library])
            # h.add_layer(ReadColor(ad.X, reverseStrand), legend=False)
            # h.add_layer(GeneLine(ad.X, reverseStrand))
            # h.add_layer(GeneSplitLine(ad.X, linewidth=0))
            if addMetaInfo:
                h.add_top(ReadMeta(ad.X, reverseStrand, sr_readStrandInGenome=ad.obs['strandInGenome'], **dt_metaKwargs), pad=0.1)
            if addGeneStru:
                h = self.addGeneStructurePlotter(h, reverseStrand, configName, pad=0.65)
                addGeneStru = False
            if addGeneStruAll:
                addGeneStru = True
            # h.hsplit(labels = ad.obs['type'])
            # h.add_left(mp.Colors(ad.obs['type']), size=0.2, name = 'Read Category')
            # h.add_legends(order=['Read Category'])
            h.add_bottom(LibrarySplitLine(ad.X), size=0.1)

            dt_plotter[library] = h
        return dt_plotter