import sh
import glob
import click
from tqdm import tqdm

def main(ls_input, dir_output, threads, fast5):
    ls_pass = []
    ls_fail = []
    ls_h5 = []
    ls_summary = []
    sh.mkdir(dir_output, p=True)
    for dir_onePart in ls_input:
        dir_onePart = dir_onePart + '/'
        ls_pass.extend(glob.glob(f"{dir_onePart}/pass/*.fastq"))
        ls_fail.extend(glob.glob(f"{dir_onePart}/fail/*.fastq"))
        ls_h5.extend(glob.glob(f"{dir_onePart}/workspace/*.fast5"))
        ls_summary.extend(glob.glob(f"{dir_onePart}/sequencing_summary.txt"))

    sh.cat(*ls_pass, _out=f"{dir_output}/allPass.fastq")
    sh.pigz(f"{dir_output}/allPass.fastq", p=threads)
    sh.cat(*ls_fail, _out=f"{dir_output}/allFail.fastq")
    sh.pigz(f"{dir_output}/allFail.fastq", p=threads)
    if fast5:
        dir_outputH5 = f"{dir_output}/workspace/"
        sh.mkdir(dir_outputH5, p=True)
        assert len(ls_h5) == len(set(ls_h5)), "fast5\' name is not unique "
        for path_h5 in tqdm(ls_h5):
            sh.cp(path_h5, dir_outputH5)
        sh.rm(f"{dir_output}/temp_summary_no_header.txt")
        with open(f"{dir_output}/temp_summary_no_header.txt", 'a') as fh:
            for path_summary in ls_summary:
                sh.tail(path_summary, n='+2', _out=fh)
        sh.sort('-k1', '-k2' ,f"{dir_output}/temp_summary_no_header.txt")
        sh.head(ls_summary[0], n=1, _out=f"{dir_output}/temp_summary_header.txt")
        sh.cat(f"{dir_output}/temp_summary_header.txt", f"{dir_output}/temp_summary_no_header.txt", _out=f"{dir_output}/sequencing_summary.txt")
            
@click.command()
@click.argument('ls_input', nargs=-1)
@click.option('-o', 'dir_output', required=True)
@click.option('-t', 'threads', type=int, required=True)
@click.option('--f5', 'fast5', is_flag=True, required=True)
def mainCmd(ls_input, dir_output, threads, fast5):
    main(ls_input, dir_output, threads, fast5)

mainCmd()