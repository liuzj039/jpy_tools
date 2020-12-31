#!/usr/bin/env python
import os
import sh
import click
import time


@click.command()
@click.option("-c", "CONFIG_FILE", help="configFile")
@click.option("-t", "THREADS", default=128, help="threads; default = 128")
@click.option("-h",
              "HOST_THREADS",
              default=64,
              help="max host threads; default = 64")
@click.option("-S", "SUFFIX", default=0, help="suffix; default = 0")
@click.option("-n", "NOT_RUN", help="just print", is_flag=True)
@click.option("-u", "UNLOCK", help="unlock dir", is_flag=True)
@click.option("-F", "FORCE_RERUN", help="force rerun", is_flag=True)
@click.option("-s", 'SNAKEFILE', default='./snakefile', help='snakefile path')
def main(CONFIG_FILE,
         THREADS,
         HOST_THREADS,
         NOT_RUN,
         UNLOCK,
         FORCE_RERUN,
         SNAKEFILE,
         SUFFIX="0"):

    configFile = CONFIG_FILE
    configFile = str(sh.realpath(configFile)).strip()
    if NOT_RUN:
        if FORCE_RERUN:
            with os.popen(
                    f"snakemake -npFj {THREADS} --local-cores {HOST_THREADS} -s {SNAKEFILE} --configfile {configFile}"
            ) as fh:
                print(fh.read())
        else:
            with os.popen(
                    f"snakemake -npj {THREADS} --local-cores {HOST_THREADS} -s {SNAKEFILE} --configfile {configFile}"
            ) as fh:
                print(fh.read())

    elif UNLOCK:
        with os.popen(
                f"snakemake -j {THREADS} -s {SNAKEFILE} --unlock --configfile {configFile}"
        ) as fh:
            print(fh.read())

    else:
        localTime = time.strftime("%Y%m%d_%H%M", time.localtime())
        logFile = (configFile.split(".yaml")[0] + "_" + str(SUFFIX) + "_" +
                   localTime + ".log")

        messageTitleTrue = f"在{localTime}提交的snakemake已经完成了, bug free bug free"
        messageTitleFalse = f"在{localTime}提交的snakemake失败了,dedededededebugbugbug"
        messageRunTrue = (
            f"jpy_sendMessage.py --title '{messageTitleTrue}' --content '看不到我看不到我看不到我'"
        )
        messageRunFalse = (
            f"jpy_sendMessage.py --title '{messageTitleFalse}' --content '看不到我看不到我看不到我'"
        )

        if FORCE_RERUN:
            os.system(
                f"nohup snakemake -pFj {THREADS} --local-cores {HOST_THREADS} -s {SNAKEFILE} --configfile {configFile} -c \
                    'jsub.py --sm -N {{rulename}} -t {{threads}} -g {{params.gpu}}'\
                        > {logFile} 2>&1 && {messageRunTrue} || {messageRunFalse} &"
            )
        else:
            os.system(
                f"nohup snakemake -pj {THREADS} --local-cores {HOST_THREADS} -s {SNAKEFILE} --configfile {configFile} -c \
                    'jsub.py --sm -N {{rulename}} -t {{threads}} -g {{params.gpu}}'\
                        > {logFile} 2>&1 && {messageRunTrue} || {messageRunFalse} &"
            )


main()
