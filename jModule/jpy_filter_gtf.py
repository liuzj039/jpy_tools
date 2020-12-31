#!/usr/bin/env python
import csv
import click
import pandas as pd

@click.command()
@click.argument('gtfpath')
@click.argument('outpath')
def main(gtfpath, outpath):
    """
    只保留'3UTR', '5UTR', 'CDS', 'exon', 'start_codon', 'stop_codon', 'transcript'类型的行
    """
    pd.read_table(
        gtfpath,
        comment="#",
        names=[
            "Chromosome",
            "Source",
            "Feature",
            "Start",
            "End",
            "Score",
            "Strand",
            "Frame",
            "Attr",
        ],
    ).query(
        "Feature in ['3UTR', '5UTR', 'CDS', 'exon', 'start_codon', 'stop_codon', 'transcript']"
    ).to_csv(outpath,
             sep='\t',
             header=None,
             index=None,
             quoting=csv.QUOTE_NONE)

main()