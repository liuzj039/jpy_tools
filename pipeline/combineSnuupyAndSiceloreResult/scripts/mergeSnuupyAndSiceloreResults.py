import pandas as pd
import click
import pysam
from collections import namedtuple
from itertools import chain, repeat
from loguru import logger
import readProcessTools

@click.command()
@click.option('--si-bam', 'siceloreBamPath')
@click.option('--si-fasta', 'siceloreReadFastaPath')
@click.option('--sn-fea', 'snuupyAssignResultPath')
@click.option('--sn-fasta', 'snuupyReadFastaPath')
@click.option('--out-fasta', 'targetOutFastaPath')
@click.option('--out-fea', 'targetOutAssignPath')
def main(
    siceloreBamPath,
    snuupyAssignResultPath,
    siceloreReadFastaPath,
    snuupyReadFastaPath,
    targetOutFastaPath,
    targetOutAssignPath,
):
    """
    \b
    用于合并snuupy和sicelore获得的read。
    read被两个软件分配给了不同UMI的read会被移除。
    \b
    siceloreBamPath:
        umi_found bam
    snuupyAssignResultPath:
        assign result; feather format
    siceloreReadFastaPath:
        sicelore consensus fasta
    snuupyReadFastaPath:
        snuupy consensus fasta
    targetOutFastaPath:
        merged fasta
    targetOutAssignPath:
        merged barcode assign result; feater format
    """
    ### read input ###
    snuupyAssignDf = pd.read_feather(snuupyAssignResultPath).reindex(
        ["name", "qseqid"], axis=1
    )
    siceloreBam = pysam.AlignmentFile(siceloreBamPath)
    siceloreAssignDt = {}
    for read in siceloreBam:
        siceloreAssignDt[read.qname.split("_")[0]] = (
            read.get_tag("BC") + "_" + read.get_tag("U8")
        )
    siceloreAssignDf = pd.DataFrame.from_dict(siceloreAssignDt, "index").reset_index()
    siceloreAssignDf.columns = ["name", "qseqid"]

    ### remove conflict reads ###
    excludeReadsSt = set(
        snuupyAssignDf.merge(
            siceloreAssignDf,
            left_on="name",
            right_on="name",
            suffixes=("_snuupy", "_sicelore"),
        ).query("qseqid_snuupy != qseqid_sicelore")["name"]
    )
    assignDf = (
        pd.concat([snuupyAssignDf, siceloreAssignDf])
        .query("name not in @excludeReadsSt")
        .drop_duplicates("name")
        .reset_index(drop=True)
    )
    snuupyUseUmiSt = set(snuupyAssignDf.query("qseqid in @assignDf.qseqid")["qseqid"])
    siceloreUseUmiSt = set(
        siceloreAssignDf.query(
            "qseqid in @assignDf.qseqid and qseqid not in @snuupyUseUmiSt"
        )["qseqid"]
    )
    siceloreReadFasta = readProcessTools.readFasta(siceloreReadFastaPath)
    snuupyReadFasta = readProcessTools.readFasta(snuupyReadFastaPath)

    ### write ###
    i = 0
    splitChrDt = {"snuupy": "_", "sicelore": "-"}
    useUmiDt = {"snuupy": snuupyUseUmiSt, "sicelore": siceloreUseUmiSt}
    with open(targetOutFastaPath, "w") as fh:
        for sourceName, read in chain(
            zip(repeat("snuupy"), snuupyReadFasta),
            zip(repeat("sicelore"), siceloreReadFasta),
        ):
            i += 1
            if i % 10000 == 0:
                logger.info(f"{i} reads processed")

            readNameLs = read.name.split(splitChrDt[sourceName])

            readBarcodeUmi = readNameLs[0] + "_" + readNameLs[1]
            if readBarcodeUmi in useUmiDt[sourceName]:
                read.name = "_".join(readNameLs)
                readProcessTools.writeFasta(read, fh)
    assignDf.to_feather(targetOutAssignPath)

if __name__ == "__main__":
    main()