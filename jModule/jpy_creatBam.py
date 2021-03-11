#!/usr/bin/env python
'''
@Author: liuzj
@Date: 2020-05-14 12:47:05
@LastEditors: liuzj
@LastEditTime: 2020-05-19 20:13:51
@Description: sam转bam
@FilePath: /liuzj/softwares/python_scripts/jpy_creatBam.py
'''
import os
import click
@click.command()
@click.option('-i','infile',required=True,help='sam file')
@click.option('-o','outfile',required=True,help='bam file')
@click.option('-r','remove',is_flag=True,help='whether romove the input sam')
def main(infile,outfile,remove):
    tempFile = outfile+'.temp'
    os.system(f'samtools view -Sb {infile} -o {tempFile}')
    os.system(f'samtools sort -@ 12 {tempFile} -o {outfile}')
    os.system(f'samtools index {outfile}')
    os.system(f'rm {tempFile}')
    if remove:
        os.system(f'rm {infile}')
if __name__ == '__main__':
    main()