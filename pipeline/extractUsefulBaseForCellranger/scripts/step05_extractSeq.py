'''
Date: 2020-08-12 11:28:09
LastEditors: liuzj
LastEditTime: 2020-08-19 11:27:05
Description: file content
Author: liuzj
FilePath: /liuzj/scripts/pipeline/extractUsefulBaseForCellranger/scripts/step05_extractSeq.py
'''
import os
import sh
import re
import pyfastx
import glob
import lmdb
import click
import numpy as np
from concurrent.futures import ProcessPoolExecutor as multiP
from jpy_tools.ReadProcess import writeFastq, readFastq, getSubFastq

def processOneFastq(singleR1Path, singleR2Path, lmdbPath, outDir):
    singleR1File, singleR2File = readFastq(singleR1Path), readFastq(singleR2Path)
    singleR1OutFile, singleR2OutFile = outDir + singleR1Path.split('/')[-1], outDir + singleR2Path.split('/')[-1]
    with lmdb.open(lmdbPath, map_size=1099511627776) as mdbDataBase, open(singleR1OutFile, 'w') as fh1, open(singleR2OutFile, 'w') as fh2:
        mdbFile = mdbDataBase.begin()
        for singleRead1, singleRead2 in zip(singleR1File, singleR2File):
            singleUsefulRegion = mdbFile.get(singleRead1.name.encode())
            if singleUsefulRegion:
                singleUsefulRegion = np.frombuffer(singleUsefulRegion, dtype=int).reshape(-1, 2)
                singleRead2Corrected=getSubFastq(singleRead2, singleUsefulRegion)
                if len(singleRead2Corrected.seq) >= 75 :
                    writeFastq(singleRead1, fh1)
                    writeFastq(singleRead2Corrected, fh2)


@click.command()
@click.option('-i', 'fastqDir')
@click.option('-o', 'outDir')
@click.option('-l', 'lmdbPath', help = 'lmdbPath')
@click.option('-t', 'threads', type=int)
@click.option('-s', 'splitInput', is_flag=True)
def main(fastqDir, outDir, lmdbPath, threads, splitInput):
    os.mkdir(outDir)
    if not splitInput:
        allR1Path = glob.glob(f'{fastqDir}*R1*')
        allR2Path = [x.replace('R1', 'R2') for x in allR1Path]
    else:

        fastqTemp = fastqDir + 'tempSplited/'
        sh.mkdir(fastqTemp)

        allR1Path = glob.glob(f'{fastqDir}*R1*')
        allR2Path = [x.replace('R1', 'R2') for x in allR1Path]
        allSplitedPath = [fastqTemp + re.search(r'(?<=/)\w+?(?=_R1)', x)[0] + '/' for x in allR1Path]

        splitedNum = threads // len(allSplitedPath)
        
        if splitedNum <= 1 :
            allR1Path = glob.glob(f'{fastqDir}*R1*')
            allR2Path = [x.replace('R1', 'R2') for x in allR1Path]
        else:
            mPResults = []
            with multiP(threads//2) as mP:
                for singleR1Path, singleR2Path, singleSplitedPath in zip(allR1Path, allR2Path, allSplitedPath):
                    mPResults.append(mP.submit(sh.seqkit, "split2", "-f", "-1", singleR1Path, "-2", singleR2Path, p=splitedNum, O=singleSplitedPath, j=2))

            tempAllSplitedR1Path = glob.glob(f'{fastqTemp}*/*R1*')
            tempAllSplitedR2Path = [x.replace('R1', 'R2') for x in tempAllSplitedR1Path]
            sampleId = set([re.search(r'(?<=/)\w+?(?=_L)',x)[0] for x in tempAllSplitedR1Path])

            if len(sampleId) != 1:
                raise NameError("MORE THAN ONE INPUT SAMPLES")
            else:
                sampleId = sampleId.pop()

            i = 0
            for tempSingleSplitedR1Path, tempSingleSplitedR2Path in zip(tempAllSplitedR1Path, tempAllSplitedR2Path):
                i += 1
                sh.mv(tempSingleSplitedR1Path, f'{fastqTemp}{sampleId}_L{i:03}_R1_001.fastq')
                sh.mv(tempSingleSplitedR2Path, f'{fastqTemp}{sampleId}_L{i:03}_R2_001.fastq')

            for singleTempDir in glob.glob(f'{fastqTemp}*/'):
                sh.rmdir(singleTempDir)

            allR1Path = glob.glob(f'{fastqTemp}*R1*')
            allR2Path = [x.replace('R1', 'R2') for x in allR1Path]
        
    
    allSubProcess = []
    with multiP(threads) as mP:
        for singleR1Path, singleR2Path in zip(allR1Path, allR2Path):
            allSubProcess.append(mP.submit(processOneFastq, singleR1Path, singleR2Path, lmdbPath, outDir))
    [x.result() for x in allSubProcess]
    
    if not splitInput:
        pass
    else:
        sh.rm('-rf', fastqTemp)

        
if __name__ == '__main__':
    main()