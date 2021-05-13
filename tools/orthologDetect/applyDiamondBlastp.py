import sh
import os
import click
import pandas as pd
from loguru import logger


def buildDiamondIndex(proteinFaPath):
    indexPath = proteinFaPath + ".diamond"
    if os.path.exists(indexPath + ".dmnd"):
        pass
    else:
        sh.diamond.makedb(
            "--in",
            proteinFaPath,
            d=indexPath,
        )


def blastpByDiamond(targetProteinFaPath, queryProteinFaPath, threads, outputPath):
    buildDiamondIndex(targetProteinFaPath)
    targetProteinIndexPath = targetProteinFaPath + ".diamond.dmnd"
    subprocess = sh.diamond.blastp(
        "--ultra-sensitive",
        "--query-cover",
        "80",
        "--subject-cover",
        "80",
        p=threads,
        d=targetProteinIndexPath,
        q=queryProteinFaPath,
        o=outputPath,
        e=1e-5,
        _bg=True,
    )
    return subprocess


def getRBH(inPath, ALabel, BLabel, ALambda, BLambda, outputPath):
    ALambda = eval(f"lambda {ALambda}")
    BLambda = eval(f"lambda {BLambda}")

    colNameLs = [
        "query",
        "target",
        "seqIdentity",
        "Length",
        "mismatches",
        "gapCounts",
        "queryStart",
        "queryEnd",
        "targetStart",
        "targetEnd",
        "eValue",
        "bitScore",
    ]
    useColLs = ["query", "target", "bitScore"]

    aDf = pd.read_table(
        f"{inPath}target{ALabel}_query{BLabel}_diamondBlastp.tsv",
        names=colNameLs,
        usecols=useColLs,
    ).pipe(
        lambda df: df.assign(
            target=df["target"].map(ALambda), query=df["query"].map(BLambda), 
        )
    )
    bDf = pd.read_table(
        f"{inPath}target{BLabel}_query{ALabel}_diamondBlastp.tsv",
        names=colNameLs,
        usecols=useColLs,
    ).pipe(
        lambda df: df.assign(
            target=df["target"].map(BLambda), query=df["query"].map(ALambda)
        )
    )
    aDf = (
        aDf.sort_values(["query", "bitScore"], ascending=False)
        .drop_duplicates("query")
        .assign(mergeBy=lambda df: df["query"] + "_split_" + df["target"])
    )
    bDf = (
        bDf.sort_values(["query", "bitScore"], ascending=False)
        .drop_duplicates("query")
        .assign(mergeBy=lambda df: df["target"] + "_split_" + df["query"])
    )

    useHitSq = aDf["mergeBy"][aDf["mergeBy"].isin(bDf["mergeBy"])].str.split("_split_")
    rbDf = pd.DataFrame()
    rbDf = rbDf.assign(specieA=useHitSq.str[0], specieB=useHitSq.str[1])
    rbDf.columns = [ALabel, BLabel]
    rbDf.to_csv(outputPath, sep="\t", index=None)


@click.command()
@click.option("-a", "AProteinFaPath")
@click.option("-b", "BProteinFaPath")
@click.option("--al", "ALabel")
@click.option("--bl", "BLabel")
@click.option("--alambda", "ALambda", default='x:x.split(".")[-2]', show_default=True)
@click.option("--blambda", "BLambda", default='x:x.split(".")[-2]', show_default=True)
@click.option("-t", "threads", type=int)
@click.option("-o", "outputPath")
def main(AProteinFaPath, BProteinFaPath, ALabel, BLabel, ALambda, BLambda, threads, outputPath):
    """
    \b
    use diamond to perform reciprocal blastp.
    \b
    AProteinFaPath / BProteinFaPath: fa path
    ALabel / BLabel: corresponding label
    ALambda / BLambda: used for transform transcript name to gene name
    threads: Total threads.
    outputPath: output directory
    """
    outputPath = outputPath.rstrip("/") + "/"
    ALabel = ALabel.capitalize()
    BLabel = BLabel.capitalize()

    targetFaPathLs = [AProteinFaPath, BProteinFaPath]
    queryFaPathLs = [BProteinFaPath, AProteinFaPath]
    threadLs = [threads // 2, threads // 2]
    outputPathLs = [
        f"{outputPath}target{ALabel}_query{BLabel}_diamondBlastp.tsv",
        f"{outputPath}target{BLabel}_query{ALabel}_diamondBlastp.tsv",
    ]

    logger.info("start perform blastp by diamond")
    subprocessLs = list(
        map(blastpByDiamond, targetFaPathLs, queryFaPathLs, threadLs, outputPathLs)
    )
    [x.wait() for x in subprocessLs]

    logger.info("start find reciprocal best hit from blastp result")
    rbhResultPath = f"{outputPath}{ALabel}_with_{BLabel}_RBH.tsv"
    getRBH(outputPath, ALabel, BLabel, ALambda, BLambda, rbhResultPath)


main()