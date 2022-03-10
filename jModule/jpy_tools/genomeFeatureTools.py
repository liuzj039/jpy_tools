import pandas as pd
import numpy as np
import pyranges as pr
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from typing import Collection, Tuple, Optional, Union
from tqdm import tqdm
import portion as P


def getRemoveLs(df: pd.DataFrame):
    "for `GtfProcess.getNonoverlapGene`"
    removeGene_ls = []
    for line in df.itertuples():
        start = line.Start
        end = line.End
        if line.gene_id in removeGene_ls:
            continue
        else:
            overlap_df = df.query("Start < @end & End > @start")
            if len(overlap_df) > 1:
                removeGene_ls.extend(list(overlap_df["gene_id"]))
    return removeGene_ls


class GtfProcess(object):
    @staticmethod
    def getNonoverlapGene(
        gtfPath: str, outPath: Optional[str] = None, threads: int = 20
    ) -> None:
        """
        get non-overlap gene. Only protein_coding genes were used

        Parameters
        ----------
        gtfPath : str
            has gene_biotype feature
        outPath : Optional[str], optional
            by default None, which means the outPath is `gtfPath` ".noOverlapWithOthers.tsv"
        threads : int, optional
            by default 20
        """
        if not outPath:
            outPath = gtfPath + ".noOverlapWithOthers.tsv"
        gtf = pr.read_gtf(gtfPath, as_df=True)
        gtf = gtf.query("Feature == 'gene'").filter(
            ["Chromosome", "Start", "End", "gene_biotype", "gene_id"]
        )
        gtf = gtf.query("gene_biotype == 'protein_coding'")
        logger.info(f"input feature counts: {len(gtf['gene_id'].unique())}")
        with ProcessPoolExecutor(threads) as mtP:
            removeGene_ls = []
            for chrName, df in gtf.groupby("Chromosome"):
                removeGene_ls.append(mtP.submit(getRemoveLs, df))

        removeGene_ls = [x.result() for x in removeGene_ls]
        removeGene_ls = [y for x in removeGene_ls for y in x]
        gtf = gtf.query("gene_id not in @removeGene_ls")
        logger.info(f"output feature counts: {len(gtf['gene_id'].unique())}")
        gtf.to_csv(outPath, sep="\t", index=False)

    @staticmethod
    def addPrerna(gtfPath: str, outPath: Optional[str] = None) -> None:
        gtf = pr.read_gtf(gtfPath, as_df=True)
        gtf = gtf[
            [
                "Chromosome",
                "Source",
                "Feature",
                "Start",
                "End",
                "Score",
                "Strand",
                "Frame",
                "gene_id",
                "transcript_id",
            ]
        ].query("Feature in ['transcript', 'exon']")
        ls_needPre = (
            gtf.query("Feature == 'exon'")
            .groupby("transcript_id")["Feature"]
            .agg("count")
            .pipe(lambda sr: sr[sr > 1].index)
            .to_list()
        )
        gtf_tr = gtf.query(f"Feature == 'transcript' & transcript_id in {ls_needPre}")
        gtf_tr["transcript_id"] = gtf_tr["transcript_id"] + "_pre"
        gtf = pd.concat([gtf, gtf_tr, gtf_tr.assign(Feature="exon")]).sort_values(
            ["Chromosome", "Start"]
        )
        gtf = pr.PyRanges(gtf)
        if not outPath:
            outPath = gtfPath + ".withPre.gtf"
        gtf.to_gtf(outPath)

    @staticmethod
    def addTranscriptAndGeneInfo(
        path_gtf: str,
        path_out: Optional[str] = None,
        forceGene: Optional[bool] = None,
        forceTrans: Optional[bool] = None,
        needName: bool = True,
        return_df: bool = False,
    ) -> None:
        """
        get transcript and gene information from exon information

        Parameters
        ----------
        path_gtf : str
        path_out : Optional[str], optional
            by default None, path_out is `path_gtf` + '.addTrsAndGene.gtf'
        forceGene : Optional[bool], optional
            by default None, which means if original gtf has gene information, keep it. if original gtf has no gene information, add it based on exon information
        forceTrans : Optional[bool], optional
            by default None, which means if original gtf has transcript information, keep it. if original gtf has no transcript information, add it based on exon information
        """

        def _mergeExon(df, feature="transcript"):
            sr = df.iloc[0]
            sr["Feature"] = feature
            sr["Start"] = df["Start"].min()
            sr["End"] = df["End"].max()
            if feature == "gene":
                sr["transcript_id"] = float("nan")
            return sr

        if path_out is None:
            path_out = path_gtf + ".addTrsAndGene.gtf"

        df_gtf = pr.read_gtf(path_gtf, as_df=True)

        _ls = ["gene",  "transcript", "exon"]
        df_gtf = df_gtf.filter(
            [
                "Chromosome",
                "Source",
                "Feature",
                "Start",
                "End",
                "Score",
                "Strand",
                "Frame",
                "gene_id",
                "transcript_id",
            ]
        ).query("Feature in @_ls")
        if forceGene is None:
            if "gene_id" in df_gtf["Feature"].unique():
                forceGene = False
            else:
                forceGene = True
        if forceTrans is None:
            if "transcript_id" in df_gtf["Feature"].unique():
                forceTrans = False
            else:
                forceTrans = True
        if forceGene:
            df_geneGtf = (
                df_gtf.query("Feature == 'exon'")
                .groupby("gene_id", as_index=False)
                .apply(_mergeExon, feature="gene")
            )
            df_gtf = pd.concat([df_gtf, df_geneGtf])
        if forceTrans:
            df_trGtf = (
                df_gtf.query("Feature == 'exon'")
                .groupby("transcript_id", as_index=False)
                .apply(_mergeExon, feature="transcript")
            )
            df_gtf = pd.concat([df_gtf, df_trGtf])
            
        df_gtf['Feature'] = df_gtf['Feature'].astype('category').cat.reorder_categories(_ls)
        df_gtf = df_gtf.sort_values(["Chromosome", "Start", "Feature"])
        df_gtf["transcript_id"] = df_gtf["transcript_id"].fillna("GENE_EMPTY")
        ls_trsOrder = df_gtf["transcript_id"].unique().tolist()
        ls_geneOrder = df_gtf["gene_id"].unique().tolist()
        df_gtf = (
            df_gtf.pipe(
                lambda df: df.assign(
                    gene_id=df["gene_id"]
                    .astype("category")
                    .cat.reorder_categories(ls_geneOrder),
                    transcript_id=df["transcript_id"]
                    .astype("category")
                    .cat.reorder_categories(ls_trsOrder),
                    Start=np.where(df["Strand"] == "+", df["Start"], -df["Start"]),
                )
            )
            .sort_values(["gene_id", "transcript_id", "Feature", "Start"])
            .assign(
                Start=lambda df: np.where(
                    df["Strand"] == "+", df["Start"], -df["Start"]
                ),
                transcript_id=lambda df: df["transcript_id"]
                .astype(str)
                .map(lambda x: x if x != "GENE_EMPTY" else float("nan")),
                gene_id=lambda df: df["gene_id"].astype(str),
            )
            .reset_index(drop=True)
        )

        # df_gtf['Feature'] = df_gtf['Feature'].astype('category').cat.reorder_categories(['gene', 'transcript','exon'])
        # df_gtf = df_gtf.sort_values(['Chromosome', 'Start', 'Feature'])
        # df_gtf = df_gtf.reset_index(drop=True).sort_values(
        #     ["Chromosome", "Start"]
        # )
        if needName:
            df_gtf["gene_name"] = df_gtf["gene_id"]
            df_gtf["transcript_name"] = df_gtf["transcript_id"]
        if return_df:
            return df_gtf
        else:
            GtfProcess.writeGtf(df_gtf, path_out)

    @staticmethod
    def writeGtf(df_gtf, path_out):
        ls_gtfBase = [
            "Chromosome",
            "Source",
            "Feature",
            "Start",
            "End",
            "Score",
            "Strand",
            "Frame",
        ]
        ls_gtfExtended = [x for x in df_gtf.columns if x not in ls_gtfBase]
        with open(path_out, "w") as fh:
            for tp_line in df_gtf.itertuples():
                str_line = "\t".join([str(getattr(tp_line, x)) for x in ls_gtfBase])
                str_lineExtended = (
                    "; ".join(
                        [
                            f'{x} "{getattr(tp_line, x)}"'
                            for x in ls_gtfExtended
                            if not pd.isna(getattr(tp_line, x))
                        ]
                    )
                    + "; "
                )
                str_line = str_line + "\t" + str_lineExtended
                print(str_line, file=fh)


def getLongestIsoform(path_bed, path_tempBed=None):
    df_bed = pr.read_bed(path_bed, as_df=True)
    df_bed["IsoformLength"] = df_bed["BlockSizes"].map(
        lambda z: sum([int(x) for x in z.split(",")[:-1]])
    ) - df_bed.eval("`ThickStart` - Start + End - `ThickEnd`")
    df_bed["Gene"] = df_bed["Name"].str.split("\|").str[-1]
    df_bed = (
        df_bed.sort_values("IsoformLength", ascending=False)
        .drop_duplicates("Gene")
        .sort_values(["Chromosome", "Start"])
        .drop(columns=["IsoformLength", "Gene"])
    )
    if not path_tempBed:
        return df_bed
    else:
        df_bed.to_csv(path_tempBed, sep="\t", header=None, index=None)
        return path_tempBed


def _getExons(line):
    ls_tuple = []
    for start, length in zip(
        line.BlockStarts.split(",")[:-1], line.BlockSizes.split(",")[:-1]
    ):
        start = int(start)
        length = int(length)
        ls_tuple.append(P.closedopen(start, start + length))
    iv_exon = P.Interval(*ls_tuple)
    ls_exon = list(iv_exon)
    if not bed12:
        if line.Strand == "-":
            ls_exon = ls_exon[::-1]

        ls_exonFeature = []
        for exonNum, iv_singleExon in zip(range(1, 1 + len(ls_exon)), ls_exon):
            ls_intronFeature.append(
                [
                    f"{line.Name}_intron{intronNum}",
                    line.Chromosome,
                    line.Start + iv_singleIntron.lower,
                    line.Start + iv_singleIntron.upper,
                    line.Strand,
                ]
            )
        return ls_intronFeature


def getExonsFromBed(path_bed, longest_isoform_only=True) -> pd.DataFrame:
    if longest_isoform_only:
        df_bed = getLongestIsoform(path_bed)
    else:
        df_bed = pr.read_bed(path_bed, as_df=True)
    ls_exons = [
        _getExons(x)
        for x in tqdm(
            df_bed.itertuples(),
            total=len(df_bed),
        )
    ]
    df_exons = pd.DataFrame(
        [y for x in ls_exons for y in x],
        columns=["Name", "Chromosome", "Start", "End", "Strand"],
    )
    df_exons["Chromosome"] = df_exons["Chromosome"].astype(str)
    df_exons = df_exons.sort_values(["Chromosome", "Start"])

    return df_exons


def _getIntrons(line, bed12):
    if int(line.BlockCount) <= 1:
        return None

    ls_tuple = []
    for start, length in zip(
        line.BlockStarts.split(",")[:-1], line.BlockSizes.split(",")[:-1]
    ):
        start = int(start)
        length = int(length)
        ls_tuple.append(P.closedopen(start, start + length))
    iv_exon = P.Interval(*ls_tuple)
    iv_gene = P.closedopen(0, int(line.End) - int(line.Start))
    iv_intron = iv_gene - iv_exon
    ls_intron = list(iv_intron)
    if not bed12:
        if line.Strand == "-":
            ls_intron = ls_intron[::-1]

        ls_intronFeature = []
        for intronNum, iv_singleIntron in zip(range(1, 1 + len(ls_intron)), ls_intron):
            ls_intronFeature.append(
                [
                    f"{line.Name}_intron{intronNum}",
                    line.Chromosome,
                    line.Start + iv_singleIntron.lower,
                    line.Start + iv_singleIntron.upper,
                    line.Strand,
                ]
            )
        return ls_intronFeature
    else:
        Start = line.Start + ls_intron[0].lower
        BlockStarts = (
            ",".join([str(x.lower - ls_intron[0].lower) for x in ls_intron]) + ","
        )
        BlockSizes = ",".join([str(x.upper - x.lower) for x in ls_intron]) + ","
        BlockCount = len(ls_intron)
        End = line.Start + ls_intron[-1].upper
        sr_intron = pd.Series(
            dict(
                Chromosome=line.Chromosome,
                Start=Start,
                End=End,
                Name=line.Name,
                Score=line.Score,
                Strand=line.Strand,
                ThickStart=Start,
                ThickEnd=End,
                ItemRGB=line.ItemRGB,
                BlockCounts=BlockCount,
                BlockSizes=BlockSizes,
                BlockStarts=BlockStarts,
            )
        )
        return sr_intron


def getIntronsFromBed(path_bed, longest_isoform_only=True, bed12=False) -> pd.DataFrame:
    if longest_isoform_only:
        df_bed = getLongestIsoform(path_bed)
    else:
        df_bed = pr.read_bed(path_bed, as_df=True)
    ls_introns = [
        _getIntrons(x, bed12)
        for x in tqdm(
            df_bed.query("BlockCount > 1").itertuples(),
            total=len(df_bed.query("BlockCount > 1")),
        )
    ]
    if not bed12:
        df_introns = pd.DataFrame(
            [y for x in ls_introns for y in x],
            columns=["Name", "Chromosome", "Start", "End", "Strand"],
        )
        df_introns["Chromosome"] = df_introns["Chromosome"].astype(str)
        df_introns = df_introns.sort_values(["Chromosome", "Start"])
    else:
        df_introns = pd.DataFrame(
            ls_introns,
        )
        df_introns["Chromosome"] = df_introns["Chromosome"].astype(str)
        df_introns = df_introns.sort_values(["Chromosome", "Start"])

    return df_introns


NAMES = [
    "Chromosome",
    "Start",
    "End",
    "Name",
    "Score",
    "Strand",
    "ThickStart",
    "ThickEnd",
    "ItemRGB",
    "BlockCount",
    "BlockSizes",
    "BlockStarts",
    "geneChromosome",
    "geneStart",
    "geneEnd",
    "geneName",
    "geneScore",
    "geneStrand",
    "geneThickStart",
    "geneThickEnd",
    "geneItemRGB",
    "geneBlockCount",
    "geneBlockSizes",
    "geneBlockStarts",
    "cov",
]
USECOLS = [
    "Chromosome",
    "Start",
    "End",
    "Name",
    "Strand",
    "BlockSizes",
    "BlockStarts",
    "geneStart",
    "geneEnd",
    "geneName",
    "geneBlockCount",
    "geneBlockSizes",
    "geneBlockStarts",
    "cov",
]
