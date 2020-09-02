#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Author       : windz
@Date         : 2020-06-04 11:53:34
@LastEditTime : 2020-06-16 10:10:18
@Description  : 
'''


from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import joblib
import click


import logging
logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s %(filename)s: %(message)s',  
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    )


NAMES=[
    'Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 
    'ThickStart', 'ThickEnd', 'ItemRGB', 'BlockCount', 'BlockSizes', 'BlockStarts', 
    'geneChromosome', 'geneStart', 'geneEnd', 'geneName', 'geneScore', 'geneStrand', 
    'geneThickStart', 'geneThickEnd', 'geneItemRGB', 'geneBlockCount', 'geneBlockSizes',   'geneBlockStarts', 'cov'
    ]


USECOLS = [
    'Chromosome', 'Start', 'End', 'Name', 'Strand',
    'geneStart', 'geneEnd', 'geneName', 'geneStrand', 'geneBlockSizes', 'geneBlockStarts', 'cov'
    ]


@click.command()
@click.option('-i', '--infile', required=True)
@click.option('-o', '--outfile', required=True)
def main(infile, outfile):
    logging.info('Start read csv')
    df = pd.read_csv(
        infile, 
        sep='\t', 
        names=NAMES,
        usecols=USECOLS,
        header=None
        )
    logging.info('Read csv Done!')

    logging.info('Start find Splice Sites')
    df['geneBlockSizes'] = df['geneBlockSizes'].map(lambda x: np.fromstring(x, sep=','))
    df['geneBlockStarts'] = df['geneBlockStarts'].map(lambda x: np.fromstring(x, sep=','))
    df['five_ss'] = (df['geneStart']+df['geneBlockSizes']+df['geneBlockStarts']).map(lambda x: x[:-1])
    df['three_ss'] = (df['geneStart']+df['geneBlockStarts']).map(lambda x: x[1:])
    logging.info('Find Splice Sites Done!')

    logging.info('Main function')
    results = defaultdict(
        lambda : {
            'cov': 0, 
            'is_splicing_intermediates': False, 
            'gene_len': 0,
            'gene_id': None
            }
    )
    for item in df.itertuples():
        # 判断是否为剪切中间体
        if item.Strand == '+':
            is_splicing_intermediates = (abs(item.End-item.five_ss)<=10).any() 
        else:
            is_splicing_intermediates = (abs(item.Start-item.three_ss)<=10).any()
        results[item.Name]['is_splicing_intermediates'] = results[item.Name]['is_splicing_intermediates'] or is_splicing_intermediates
        # 取exon覆盖度最大的作为这条read的可能注释，不一定有用，先存着
        if results[item.Name]['cov'] < item.cov:
            results[item.Name]['cov'] = item.cov
            results[item.Name]['gene_id'] = item.geneName.split('.')[0]
            results[item.Name]['gene_len'] = item.geneEnd - item.geneStart
    logging.info('Main function Done!')
    
    with open(outfile, 'wb') as o:
        pickle.dump(dict(results), o)
    

if __name__ == "__main__":
    main()