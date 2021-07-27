import sh
import glob
import click

def main(ls_input, dir_output, threads):
    ls_pass = []
    ls_fail = []
    for dir_onePart in ls_input:
        dir_onePart = dir_onePart + '/'
        ls_pass.extend(glob.glob(f"{dir_onePart}/pass/*.fastq"))
        ls_fail.extend(glob.glob(f"{dir_onePart}/fail/*.fastq"))

    sh.cat(*ls_pass, _out=f"{dir_output}/allPass.fastq")
    sh.pigz(f"{dir_output}/allPass.fastq", p=threads)
    sh.cat(*ls_fail, _out=f"{dir_output}/allFail.fastq")
    sh.pigz(f"{dir_output}/allFail.fastq", p=threads)

@click.command()
@click.argument('ls_input', nargs=-1)
@click.option('-o', 'dir_output', required=True)
@click.option('-t', 'threads', type=int, required=True)
def mainCmd(ls_input, dir_output, threads):
    main(ls_input, dir_output, threads)

mainCmd()