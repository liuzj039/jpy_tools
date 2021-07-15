import pandas as pd
import pyranges as pr
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from typing import Collection, Tuple, Optional, Union
from tqdm import tqdm


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
