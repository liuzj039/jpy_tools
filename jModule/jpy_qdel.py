#!/usr/bin/env python
'''
@Author: liuzj
@Date: 2020-05-30 16:30:18
LastEditors: Liuzj
LastEditTime: 2020-09-14 10:08:02
@Description: qdel任务同时在pbslog新建一个空文件
FilePath: /liuzj/softwares/python_scripts/jpy_qdel.py
'''
import os
import sh
import click
import re

def delOneJob(pbsId, logPath):
        content = os.popen(f"qstat -f {pbsId} | sed -n '2p' | awk '{{print $3}}'").read().strip()
        print(f"qdel {pbsId}")
        os.system(f"qdel {pbsId}")
        os.system(f"touch {logPath}{content}.del")


@click.command()
@click.option('--sm', 'SNAKEMAKE_MODE', is_flag = True, help = 'snakemake mode')
@click.argument('PBS_ID')
def main(pbs_id, SNAKEMAKE_MODE): 
    """
if not snakemake mode, PBS_ID is pbsId; else snakemake logpath
    """
    logPath = '/public/home/liuzj/pbslog/'
    if not SNAKEMAKE_MODE:
        try:
            sh.qdel(pbs_id)
            print(f"qdel {pbs_id}")
        except:
            sh.bkill(pbs_id)
            print(f"bkill {pbs_id}")
    else:
        # pbs_id = pbs_id.split('/')[-1]
        logFilePath = str(sh.realpath(pbs_id)).strip()
        configPath = '/'.join(logFilePath.split('/')[:-1]) + '/config.yaml'
        # configPath = '_'.join(logFilePath.split('_')[:-3]) + '.yaml'
        print(configPath)
        with os.popen('ps -u') as fh:
            commandResults = fh.readlines()
        for lineContent in commandResults:
            if re.search(configPath, lineContent):                
                killId = lineContent.split()[1]
                print('kill', killId)
                sh.kill('-9', killId)

        snakemakeLog = pbs_id
        allPbsIds = []
        submitJobId = []
        snakemakeContents = sh.grep(sh.cat(snakemakeLog), 'jsub Id')
        for oneJob in snakemakeContents:
            oneJobId = oneJob.split('jsub Id: ')[1].split("'")[0]
            try:
                sh.qdel(oneJobId)
                print(f"qdel {oneJobId}")
            except:
                try:
                    sh.bkill(oneJobId)
                    print(f"bkill {oneJobId}")
                except:
                    pass
                
main()