'''
@Date: 2020-07-29 17:04:28
@LastEditors: liuzj
@LastEditTime: 2020-07-29 17:08:31
@Description: file content
@Author: liuzj
@FilePath: /liuzj/scripts/pipeline/analyzeNGSData/scripts/gatherIrRpkm.py
'''

import pandas as pd
import numpy as np
import click

@click.command()
@click.option('-i', 'IR_PATH')
@click.option('-R', 'RPKM_PATH')
@click.option('-o', 'OUT_PATH')
def main(IR_PATH, RPKM_PATH, OUT_PATH):
    singleSampleDt = pd.read_table(
        IR_PATH, usecols=["intron_id", "a", "b", "ab", "c", "o", "t", "iratio"],
    )

    singleSampleDt["geneName"] = singleSampleDt["intron_id"].str.split(".").str[0]

    singleSampleDt["irDenominator"] = (
        lambda x: x["a"] + x["b"] + 2 * (x["ab"] + x["c"] + x["o"])
    )(singleSampleDt)
    singleSampleDt["irNumerator"] = (lambda x: x["a"] + x["b"] + 2 * x["ab"])(
        singleSampleDt
    )

    singleSampleIrInfo = pd.DataFrame(
        singleSampleDt.groupby("geneName").apply(
            lambda x: x["irNumerator"].sum() / x["irDenominator"].sum()
        )
    )

    singleSampleIrInfo.columns = ["irRatio"]

    singleSamplRpkmDt = pd.read_table(RPKM_PATH, index_col=0, names=["rpkm"])

    singleSampleIrInfo = singleSampleIrInfo.join(singleSamplRpkmDt)

    singleSampleIrInfo.to_csv(OUT_PATH, sep="\t")

main()