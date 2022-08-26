#!/usr/bin/env python
import click
import sh
import time
from tempfile import TemporaryDirectory
import sys

def createTempBashFile(contents, tempDirName):
    tempFilePath = tempDirName + '/scripts.sh'
    with open(tempFilePath, 'w') as fh:
        fh.write(contents)
    return tempFilePath

@click.command()
@click.option('-t', 'timeSpace', type=int)
@click.option('-i', 'contents')
def main(timeSpace, contents):
    tempDir = TemporaryDirectory()
    tempDirName = tempDir.name
    scriptsPath = createTempBashFile(contents, tempDirName)
    i = 0
    print(f"restart times {i}")
    run = sh.bash(scriptsPath, _bg=True, _err = sys.stderr, _out = sys.stdout)
    while True:
        time.sleep(timeSpace)
        if run.is_alive():
            pass
        else:
            tempDir = TemporaryDirectory()
            tempDirName = tempDir.name
            scriptsPath = createTempBashFile(contents, tempDirName)
            
            i += 1
            print(f"restart times {i}")
            run = sh.bash(scriptsPath, _bg=True, _err = sys.stderr, _out = sys.stdout)

main()