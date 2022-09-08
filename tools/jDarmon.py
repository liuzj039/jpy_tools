#!/usr/bin/env python
import click
import sh
import time
from tempfile import TemporaryDirectory
import glob
import sys
import os

def createTempBashFile(contents, tempDirName):
    tempFilePath = tempDirName + '/scripts.sh'
    with open(tempFilePath, 'w') as fh:
        fh.write(contents)
    return tempFilePath

def mergeLog(dir_log, path_allLog):
    ls_log = sorted(glob.glob(dir_log + '/*.log.txt'), key=os.path.getmtime)
    with open(path_allLog, 'a') as fh:
        for log in ls_log:
            with open(log, 'r') as fh_log:
                fh.write(fh_log.read())
                fh.write('\n')
    for log in ls_log:
        os.remove(log)

@click.command()
@click.option('-t', 'timeSpace', type=int)
@click.option('-i', 'contents')
@click.option('--dir_log', 'dir_log')
@click.option('--allLog', 'path_allLog')
def main(timeSpace, contents, dir_log, path_allLog):
    tempDir = TemporaryDirectory()
    tempDirName = tempDir.name
    scriptsPath = createTempBashFile(contents, tempDirName)
    i = 0
    j = 0
    print(f"restart times {i}")
    run = sh.bash(scriptsPath, _bg=True, _err = sys.stderr, _out = sys.stdout)
    while True:
        time.sleep(timeSpace)
        j += 1
        if run.is_alive():
            pass
        else:
            tempDir = TemporaryDirectory()
            tempDirName = tempDir.name
            scriptsPath = createTempBashFile(contents, tempDirName)
            
            i += 1
            print(f"restart times {i}")
            run = sh.bash(scriptsPath, _bg=True, _err = sys.stderr, _out = sys.stdout)
        if dir_log:
            if j % 1000 == 1:
                mergeLog(dir_log, path_allLog)


main()