#!/usr/bin/env python
'''
Description: 
Author: Liuzj
Date: 2020-11-28 16:23:31
LastEditTime: 2020-11-28 16:29:19
LastEditors: Liuzj
'''
import sh
import click

def transformFeature(inPath, outPath):
    with open(outPath, 'w') as fh:
        sh.awk('{print $1"\t"$1"\tGene Expression"}', inPath, _out=fh)


def transformMtx(inPath, outPath):
    with open(outPath, 'w') as fh:
        sh.head("-n", "+3", inPath, _out = fh)
        sh.awk(sh.tail("-n", "+4", inPath), '-F', ' ', '{print $2" "$1" "$3}', _out = fh)


@click.command()
@click.option('-i', 'inpathWithPrefix', help='inpathWithPrefix, e.g.: unspliced')
@click.option('-o', 'outDir', help='outDir, end with "/"')
def main(inpathWithPrefix, outDir):
    """
    用于将kb输出的mtx转成cr输出的mtx
    """
    sh.mkdir(outDir)
    outBarcodePath = f'{outDir}barcodes.tsv'
    outFeaturePath = f'{outDir}features.tsv'
    outMatrixPath = f'{outDir}matrix.mtx'
    
    inBarcodePath = f'{inpathWithPrefix}.barcodes.txt'
    inFeaturePath = f'{inpathWithPrefix}.genes.txt'
    inMatrixPath = f'{inpathWithPrefix}.mtx'
    
    transformFeature(inFeaturePath, outFeaturePath)
    transformMtx(inMatrixPath, outMatrixPath)
    sh.cp(inBarcodePath, outBarcodePath)

    sh.gzip(outBarcodePath, outFeaturePath, outMatrixPath)

main()

