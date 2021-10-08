#!/usr/bin/env python
"""
@Author: liuzj
@Date: 2020-05-30 16:30:18
LastEditors: Liuzj
LastEditTime: 2020-09-14 10:08:02
@Description: qdel任务同时在pbslog新建一个空文件
FilePath: /liuzj/softwares/python_scripts/jpy_qdel.py
"""
import os
import sh
import click
import re
from cool import F


def delOneJob(pbs_id):
    try:
        sh.Command("/usr/local/torque/bin/qdel")(pbs_id)
        print(f"qdel {pbs_id}")
    except:
        sh.Command("/opt/ibm/lsfsuite/lsf/10.1/linux2.6-glibc2.3-x86_64/bin/bkill")(
            pbs_id
        )
        print(f"bkill {pbs_id}")


@click.command()
@click.option("--sm", "SNAKEMAKE_MODE", is_flag=True, help="snakemake mode")
@click.argument("PBS_ID", nargs=-1)
def main(pbs_id, SNAKEMAKE_MODE):
    """
    if not snakemake mode, PBS_ID is pbsId; else snakemake logpath
    """
    logPath = "/public/home/liuzj/pbslog/"

    if not SNAKEMAKE_MODE:
        if len(pbs_id) == 1:
            pbs_id = pbs_id[0]
            if '-' in pbs_id:
                pbs_id = pbs_id.split('-') | F(lambda x: [int(z) for z in x]) | F(lambda x: list(range(*x)))
            else:
                pbs_id = [pbs_id]
        for _id in pbs_id:
            delOneJob(_id)

    else:
        assert len(pbs_id) == 1, "snakemake mode only supports one jobid"
        pbs_id = pbs_id[0]
        # pbs_id = pbs_id.split('/')[-1]
        logFilePath = str(sh.realpath(pbs_id)).strip()
        configPath = "/".join(logFilePath.split("/")[:-1]) + "/config.yaml"
        # configPath = '_'.join(logFilePath.split('_')[:-3]) + '.yaml'
        print(configPath)
        with os.popen("ps -u") as fh:
            commandResults = fh.readlines()
        for lineContent in commandResults:
            if re.search(configPath, lineContent):
                killId = lineContent.split()[1]
                print("kill", killId)
                sh.kill("-9", killId)

        snakemakeLog = pbs_id
        allPbsIds = []
        submitJobId = []
        snakemakeContents = sh.grep(sh.cat(snakemakeLog), "jsub Id")
        for oneJob in snakemakeContents:
            oneJobId = oneJob.split("jsub Id: ")[1].split("'")[0]
            try:
                sh.Command("/usr/local/torque/bin/qdel")(oneJobId)
                print(f"qdel {oneJobId}")
            except:
                try:
                    sh.Command(
                        "/opt/ibm/lsfsuite/lsf/10.1/linux2.6-glibc2.3-x86_64/bin/bkill"
                    )(oneJobId)
                    print(f"bkill {oneJobId}")
                except:
                    pass


main()