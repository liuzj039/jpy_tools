#!/public/home/liuzj/softwares/anaconda3/bin/python
import click
import datetime
import re
import random
import os
import sh
import pandas as pd
import time as tm
from more_itertools import sliced
from io import StringIO
import tempfile


def getScriptContent(snakemake, inline, contents, name, cluster, noMessage):
    if inline:
        contents = " ".join(contents)
        contents = "time " + contents

    else:
        contents = contents[0]
        with open(contents, "r") as fh:
            contents = fh.readlines()
            contents = [x.rstrip() for x in contents if x.strip() != ""]
            contents = "\n".join(contents)

        if snakemake:
            findContent = re.findall(r"touch[\s\S]+?jobfailed", contents)[-1]
            replaceContent = (
                findContent
                + f" && jpy_sendMessage.py --title '{cluster}上的☆{name}☆似乎出现了意外情况' --content '看不到我看不到我看不到我'"
            )
            contents = re.sub(r"(touch[\s\S]+?jobfailed)", replaceContent, contents)
    if snakemake:
        suffixNoti = ""
    elif noMessage:
        suffixNoti = f"|| jpy_sendMessage.py --title '{cluster}上的☆{name}☆似乎出现了意外情况哎' --content '看不到我看不到我看不到我' "
    else:
        suffixNoti = f" && jpy_sendMessage.py --title '已经完成了{cluster}上的☆{name}☆了哦' --content '看不到我看不到我看不到我' \
            || jpy_sendMessage.py --title '{cluster}上的☆{name}☆似乎出现了意外情况哎' --content '看不到我看不到我看不到我' "

    contents = contents + suffixNoti

    return contents


def getServerScriptContent(cluster, node, time, queue, name, mem, threads, gpu, condaEnv):
    homePath = os.environ["HOME"]

    if cluster == "pbs":
        serverContents = f"""
#PBS -N {name}
#PBS -o {homePath}/pbslog/$PBS_JOBNAME.o$PBS_JOBID
#PBS -e {homePath}/pbslog/$PBS_JOBNAME.e$PBS_JOBID
#PBS -l nodes={node}:ppn={threads}
#PBS -l mem={mem}
#PBS -l walltime={time}
#PBS -q {queue}
#PBS -V
cd $PBS_O_WORKDIR
"""
        serverContents = serverContents.strip() + "\n"

    elif cluster == "lsf":
        serverContents = f"""
#BSUB -J {name}
#BSUB -o {homePath}/lsflog/{name}.o%J
#BSUB -e {homePath}/lsflog/{name}.e%J
#BSUB -n {threads}
#BSUB -R "span[ptile={threads}]"
#BSUB -q {queue}
"""
        if node != "1":
            serverContents += f'#BSUB -R "select[hname == {node}]"'
        serverContents = serverContents.strip() + "\n"
        if gpu != 0:
            serverContents = (
                serverContents + f"#BSUB -gpu 'num={gpu}'\nmodule load cuda/10.0\n"
            )
    if not condaEnv is None:
        serverContents = (
            serverContents + f"source activate {condaEnv}\n"
        )
    return serverContents


def parsePbsJobStatus():
    try:
        jobStatus = str(
            sh.grep(sh.tail(sh.qstat("-f", u="liuzj"), n=1000), "Job Id:", A=1)
        )
    except:
        jobStatus = ""
    IdMap = {}
    try:
        for x in sliced(jobStatus.split("\n"), 3):
            jobName = x[1].strip().split("=")[1].strip()
            jobId = x[0].strip().split(":")[1].strip().split(".")[0]
            IdMap[jobName] = jobId
    except:
        pass

    return IdMap


def parseLsfJobStatus():
    try:
        jobStatus = sh.bjobs("-w")
        jobStatus = StringIO(str(jobStatus))
        jobStatus = pd.read_csv(jobStatus, sep="\s+")
        jobStatus = jobStatus.droplevel(1)
        jobStatus = jobStatus["FROM_HOST"]
        jobStatus = jobStatus.to_dict()
        jobStatus = {y: x for x, y in jobStatus.items()}
    except:
        jobStatus = {}
    finally:
        return jobStatus


@click.command()
# source parameters
@click.option(
    "-t", "threads", type=int, default=2, help="thread use", show_default=True
)
@click.option("-g", "gpu", type=int, default=0, help="gpu use", show_default=True)
@click.option("-m", "mem", default="0", help="memory", show_default=True)
@click.option("-n", "node", default="1", help="node name", show_default=True)
@click.option("-N", "name", default="Spica", help="job name", show_default=True)
@click.option("-q", "queue", default="batch", help="queue name", show_default=True)
@click.option(
    "-T", "time", default="831:00:00", help="max running time", show_default=True
)
# mode parameters
@click.option("--sm", "snakemake", is_flag=True, help="snakemake mode")
@click.option("-i", "inline", is_flag=True, help="inline mode")
# server choice
@click.option(
    "--auto",
    "cluster",
    flag_value="auto",
    default=True,
    show_default=True,
    help="auto found the server type",
)
@click.option("--pbs", "cluster", flag_value="pbs", help="pbs server")
@click.option("--lsf", "cluster", flag_value="lsf", help="lsf server")
# contents
@click.argument("contents", nargs=-1)
# sendMessage or not
@click.option("--no-message", "noMessage", is_flag=True, help="send message or not")
@click.option("--env", 'condaEnv', default=None, help='conda environment')
def main(
    contents,
    cluster,
    inline,
    snakemake,
    time,
    queue,
    name,
    mem,
    threads,
    gpu,
    node,
    noMessage,
    condaEnv
):
    """
    submit a job. compatible with lsf / pbs server.

    v2.0.0
    """
    # initialize
    if cluster == "auto":
        try:
            sh.bsub
            cluster = "lsf"
            if queue == "batch":
                queue = "normal"
        except:
            sh.qsub
            cluster = "pbs"
        finally:
            if not snakemake:
                print(f"cluster: {cluster}")
            else:
                pass

    if mem == "0":
        mem = int(threads * 1.5) + 1
        mem = f"{mem}gb"

    today = datetime.date.today().strftime("%m%d")
    currentTime = tm.strftime("%H%M%S", tm.localtime())
    randomId = random.randint(1, 10000)
    name = f"{today}_{randomId}_{name}_{currentTime}"

    scriptContent = getScriptContent(
        snakemake, inline, contents, name, cluster, noMessage
    )
    serverScriptContent = getServerScriptContent(
        cluster, node, time, queue, name, mem, threads, gpu, condaEnv
    )
    finalContent = serverScriptContent + scriptContent

    with tempfile.NamedTemporaryFile(mode="wt") as tmpH:
        tmpH.write(finalContent)
        tmpH.flush()
        if not snakemake:
            print(finalContent)

        if cluster == "pbs":
            beforeJobStatus = parsePbsJobStatus()
            sh.qsub(tmpH.name)
            afterJobStatus = parsePbsJobStatus()

        elif cluster == "lsf":
            beforeJobStatus = parseLsfJobStatus()
            sh.bsub(sh.cat(tmpH.name))
            afterJobStatus = parseLsfJobStatus()

    differJobStatus = {
        x: y for x, y in afterJobStatus.items() if x not in beforeJobStatus.keys()
    }
    jobId = differJobStatus[name]
    print(f"jsub Id: {jobId}")

    if snakemake:
        tm.sleep(random.uniform(0.5, 1))


main()

