import os
import sh
import glob
from more_itertools import divide
import click

@click.command()
@click.option('-i', 'dir_input', required=True)
@click.option('-o', 'dir_output', required=True)
@click.option('-n', 'nparts', type=int, required=True)
def main(dir_input:str, dir_output:str, nparts:int):
    """
    \b
    split fast5 into `nparts` parts

    \b
    Parameters
    ----------
    -i, dir_input
    -o, dir_output
    -n, nparts
    """    
    dir_input = dir_input + "/"
    dir_output = dir_output + "/"
    ls_inputF5 = glob.glob(f"{dir_input}/*fast5")
    for i, ls_chunkf5 in enumerate(divide(nparts, ls_inputF5)):
        dir_chunkOutput = dir_output + f'{i}/'
        sh.mkdir(dir_chunkOutput, p=True)
        [sh.ln(x, dir_chunkOutput,s=True) for x in ls_chunkf5]

main()