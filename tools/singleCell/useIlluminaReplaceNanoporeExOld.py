'''
Description: 
Author: Liuzj
Date: 2020-09-27 15:33:36
LastEditTime: 2020-09-27 15:33:36
LastEditors: Liuzj
'''
import scanpy as sc
import pandas as pd
from jpy_tools.ReadProcess import transformExpressionMatrixTo10XMtx
import click

@click.command()
@click.option('-i', 'illuminaEx')
@click.option('-n', 'nanoporeEx')
@click.option('-o', 'nanoporeCorrectMtx')
def main(illuminaEx, nanoporeEx, nanoporeCorrectMtx):
    """
    用illumina表达量替代nanopore
    """
    illuAdata = sc.read_10x_h5(illuminaEx, genome=None, gex_only=True)
    nanoAdata = sc.read_10x_mtx(nanoporeEx)
    illuEx = illuAdata.to_df()
    nanoEx = nanoAdata.to_df()
    nanoEx = nanoEx.loc[:,nanoEx.columns.str.find('_') != [-1]]
    nanoEx = nanoEx.join(illuEx, how='inner')
    nanoEx.index = nanoEx.index.str.split('-').str[0]
    transformExpressionMatrixTo10XMtx(nanoEx, nanoporeCorrectMtx)

main()