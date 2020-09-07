'''
@LastEditors: liuzj
@LastEditTime: 2020-05-11 11:21:30
@Description: 输出更改为pandas.hdf
@FilePath: /liuzj/projects/split_barcode/01_20200507/01_pipeline/00_pipeline/buildIndex.py
'''
'''
@LastEditors: liuzj
@LastEditTime: 2020-05-08 16:37:57
@Description: 更改为命令行读取，添加了--junc-bed参数
@FilePath: /liuzj/projects/split_barcode/01_20200507/01_pipeline/00_pipeline/buildIndex.py
'''
'''
@Author       : windz
@Date         : 2020-04-01 23:36:45
@LastEditTime: 2020-05-11 10:00:11
@Description  : build index for basecalled fastq files
                构建read_id索引，以json格式保存
                索引只保留唯一比对结果的reads
                索引包含query_start, query_end, read_id所在fast5文件路径
@Usage        : build_index.py -i <in.fq> -c <config.yaml>
'''

import subprocess
from io import StringIO
import pickle
import click
import pandas as pd
import os


def read_mm2_output(mm2out):
    '''
    read the tsv file format produced by minimap2
    '''
    MM2_COLUMNS = ['qname', 'qstart', 'qend']
    df = pd.read_csv(mm2out,
                     sep='\t',
                     header=None,
                     names=MM2_COLUMNS,
                     usecols=[0, 2, 3])
    # remove multi-alignment

    df = df.drop_duplicates(subset=['qname'], keep='first')
    print(len(df))
    return df


#     return df.set_index(["qname"]).T.to_dict()


@click.command()
@click.option('-i',
              'infile',
              help='Input fastq file.',
              required=True,
              type=click.Path(exists=True))
@click.option('--genome',
              'genome',
              help='genome file.',
              required=True,
              type=click.Path(exists=True))
@click.option('-t', 'threads', help='threads', default=10, type=int)
@click.option('--f5dir',
              'f5dir',
              help='basecalling dir',
              required=True,
              type=click.Path(exists=True))
@click.option('--f5summary',
              'f5summary',
              help='basecalling sequencing summary',
              required=True,
              type=click.Path(exists=True))
@click.option('--bed',
              'bed',
              help='annotation bed',
              required=True,
              type=click.Path(exists=True))
@click.option('-o', 'outfile', help='output file', required=True)
def build_index(infile, genome, threads, f5dir, f5summary, bed, outfile):
    '''Build index mapping for basecalled reads'''
    # get config
    ref = genome
    mm2thread = str(threads)
    fast5_dir = f5dir
    sequencing_summary = f5summary
    bed = bed
    f5dir = os.popen(f'realpath {f5dir}').read()
    f5dir = f5dir.rstrip() + '/'
    proc = subprocess.run([
        'minimap2', '-x', 'splice', '-t', mm2thread, '--junc-bed', bed, '-uf',
        '-k14', '--secondary=no', '-G', '10000', ref, infile
    ],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    idx_data = read_mm2_output(StringIO(proc.stdout.decode()))
    summary_dt = pd.read_table(f5summary, low_memory=False)
    summary_dt = summary_dt.loc[:, ['filename', 'read_id']]
    summary_dt.set_index('read_id', inplace=True)
    idx_data.set_index('qname', inplace=True)
    idx_data = pd.merge(idx_data,
                        summary_dt,
                        how='left',
                        left_index=True,
                        right_index=True)
    idx_data.filename = f5dir + idx_data.filename
    idx_data.rename({'filename': 'fast5_filepath'}, axis=1, inplace=True)
    idx_data.to_hdf(outfile, key='default')


if __name__ == "__main__":
    build_index()

#     idx_data = idx_data.T.to_dict()
#     with open(outfile, 'wb') as o:
#         pickle.dump(idx_data, o)

# if __name__ == "__main__":
#     build_index()