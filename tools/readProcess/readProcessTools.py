import os
import sh
import glob
import pysam
import lmdb
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from ont_fast5_api.fast5_interface import get_fast5_file
from collections import namedtuple
from loguru import logger
import pickle


def getAntisense(seq):
    old_chars = "ACGT"
    replace_chars = "TGCA"
    transMap = str.maketrans(old_chars, replace_chars)
    return seq.translate(transMap)[::-1]


class Fastq:
    def __init__(self, name, seq, desc, qual):
        self.name = name
        self.seq = seq
        self.desc = desc
        self.qual = qual

    def __getitem__(self, key):
        sliceFastq = Fastq(self.name, self.seq[key], self.desc, self.qual[key])
        return sliceFastq

    def __str__(self):
        return f"{self.name}:\n{self.seq}"

    __repr__ = __str__

    def getAnti(self):
        return Fastq(self.name, getAntisense(self.seq), self.desc, self.qual[::-1])


class Fasta:
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq

    def __getitem__(self, key):
        sliceFasta = Fasta(self.name, self.seq[key])
        return sliceFasta

    def __str__(self):
        return f"{self.name}:\n{self.seq}"

    __repr__ = __str__

    def getAnti(self):
        return Fasta(self.name, getAntisense(self.seq))


def writeFastq(read, fh):
    """
    @description: 用于将pyfastx的read输出为fastq
    @param:
        read: pyfastx fastq
        fh: file fh mode w
    @return: None
    """
    readContent = f"@{read.name}\n{read.seq}\n{read.desc}\n{read.qual}\n"
    fh.write(readContent)


def writeFasta(read, fh):
    """
    @description: 用于将pyfastx的read输出为fasta
    @param:
        read: pyfastx fasta
        fh: file fh mode w
    @return: None
    """
    readContent = f">{read.name}\n{read.seq}\n"
    fh.write(readContent)


class FastaContent:
    """
    fasta容器
    """

    def __init__(self, path, useIndex=False):
        self.path = path
        self.useIndex = useIndex
        if self.useIndex:
            self.indexPath = f'{self.path}_lmdb/'
            if os.path.exists(self.indexPath):
                pass
            else:
                self.buildIndex()
            self.lmdbEnv = lmdb.Environment(self.indexPath, max_readers=1024, readonly=True)
            self.lmdbTxn = self.lmdbEnv.begin()
            self.keys = None

    def __len__(self):
        if not self.useIndex:
            logger.error(f"NO Index mode!")
            0 / 0
        else:
            if not self.keys:
                self.keys = pickle.loads(self.lmdbTxn.get("thisFileIndex".encode()))
            return len(self.keys)

    def __getitem__(self, readName):
        if not self.useIndex:
            logger.error(f"NO Index mode!")
            0 / 0
        # elif readName not in self.__keySt:
        #     logger.error(f"NOT Found Read name!")
        #     0 / 0
        else:
            return self.__readFastaReadFromLmdb(readName)

    def buildIndex(self):
        """
        建立lmdb数据库, lmdb中的'thisFileIndex'对应的为所有fasta的Name
        """

        def __writeFastaReadToLmdb(lmdbTxn, fastaRead):
            """
            将单条fasta写入lmdb文件
            """
            lmdbTxn.put(key=f"{fastaRead.name}_name".encode(), value=fastaRead.name.encode())
            lmdbTxn.put(key=f"{fastaRead.name}_seq".encode(), value=fastaRead.seq.encode())
            

        lmdbEnv = lmdb.open(self.indexPath, map_size=1099511627776, max_readers=1024)
        lmdbTxn = lmdbEnv.begin(write=True)
        readNameLs = []
        for i, fastaRead in enumerate(self.__readFasta()):
            readNameLs.append(fastaRead.name)
            __writeFastaReadToLmdb(lmdbTxn, fastaRead)
            if i % 1e5 == 0:
                logger.info(f"{i} reads processed")
        readNameMergedPk = pickle.dumps(readNameLs)
        lmdbTxn.put(key="thisFileIndex".encode(), value=readNameMergedPk)
        lmdbTxn.commit()
        lmdbEnv.close()

    def __readFasta(self):
        """
        读取fasta文件
        """
        with open(self.path, "r") as fh:
            i = 0
            while True:
                lineContent = fh.readline().strip()
                if lineContent == "":
                    break
                if lineContent.startswith(">"):
                    i += 1
                    if i == 1:
                        readName = lineContent[1:].split(" ")[0]
                        readSeq = ""
                    else:
                        read = Fasta(name=readName, seq=readSeq)
                        yield read
                        readName = lineContent[1:].split(" ")[0]
                        readSeq = ""
                else:
                    readSeq += lineContent
            read = Fasta(name=readName, seq=readSeq)
            yield read

    def __readFastaReadFromLmdb(self,readName):
        """
        从lmdb读取单条fasta
        """
        read = Fasta(
            name=self.lmdbTxn.get(f"{readName}_name".encode()).decode(),
            seq=self.lmdbTxn.get(f"{readName}_seq".encode()).decode(),
        )
        return read

    def __readLmdb(self):
        """
        读取lmdb文件
        """
        if not self.keys:
            self.keys = pickle.loads(self.lmdbTxn.get("thisFileIndex".encode()))        
        for readName in self.keys:
            yield self.__readFastaReadFromLmdb(readName)

    def iter(self):
        """
        迭代
        """
        if self.useIndex:
            return self.__readLmdb()
        else:
            return self.__readFasta()
    
    def close(self):
        self.lmdbEnv.close()


class FastqContent:
    """
    fastq容器
    """

    def __init__(self, path, useIndex=False):
        self.path = path
        self.useIndex = useIndex
        if self.useIndex:
            self.indexPath = f'{self.path}_lmdb/'
            if os.path.exists(self.indexPath):
                pass
            else:
                self.buildIndex()
            self.lmdbEnv = lmdb.Environment(self.indexPath)
            self.lmdbTxn = self.lmdbEnv.begin()
            self.keys = None
            # self.keys = pickle.loads(self.lmdbTxn.get("thisFileIndex".encode()))
            self.__keySt = set(self.keys)

    def __len__(self):
        if not self.useIndex:
            logger.error(f"NO Index mode!")
            0 / 0
        else:
            if not self.keys:
                self.keys = pickle.loads(self.lmdbTxn.get("thisFileIndex".encode()))
            return len(self.keys)

    def __getitem__(self, readName):
        if not self.useIndex:
            logger.error(f"NO Index mode!")
            0 / 0
        elif readName not in self.__keySt:
            logger.error(f"NOT Found Read name!")
            0 / 0
        else:
            return self.__readFastqReadFromLmdb(readName)

    def buildIndex(self):
        """
        建立lmdb数据库, lmdb中的'thisFileIndex'对应的为所有fastq的Name
        """

        def __writeFastqReadToLmdb(lmdbTxn, fastaRead):
            """
            将单条fastq写入lmdb文件
            """
            lmdbTxn.put(
                key=f"{fastaRead.name}_name".encode(), value=fastaRead.name.encode()
            )
            lmdbTxn.put(
                key=f"{fastaRead.name}_seq".encode(), value=fastaRead.seq.encode()
            )
            lmdbTxn.put(
                key=f"{fastaRead.name}_desc".encode(), value=fastaRead.desc.encode()
            )
            lmdbTxn.put(
                key=f"{fastaRead.name}_qual".encode(), value=fastaRead.qual.encode()
            )

        lmdbEnv = lmdb.open(self.indexPath, map_size=1099511627776)
        lmdbTxn = lmdbEnv.begin(write=True)
        readNameLs = []
        for i, fastqRead in enumerate(self.__readFastq()):
            readNameLs.append(fastqRead.name)
            __writeFastqReadToLmdb(lmdbTxn, fastqRead)
            if i % 1e5 == 0:
                logger.info(f"{i} reads processed")
        readNameMergedPk = pickle.dumps(readNameLs)
        lmdbTxn.put(key="thisFileIndex".encode(), value=readNameMergedPk)
        lmdbTxn.commit()
        lmdbEnv.close()

    def __readFastq(self):
        """
        从fastq读取read
        """
        with open(self.path, "r") as fh:
            i = 0
            readContent = []
            while True:
                lineContent = fh.readline()
                if lineContent == "":
                    break
                i += 1
                readContent.append(lineContent.strip())
                if i % 4 == 0:
                    read = Fastq(
                        name=readContent[0][1:].split(" ")[0],
                        seq=readContent[1],
                        desc=readContent[2],
                        qual=readContent[3],
                    )
                    yield read
                    readContent = []

    def __readFastqReadFromLmdb(self, readName):
        """
        从lmdb读取单条fasta
        """
        read = Fastq(
            name=self.lmdbTxn.get(f"{readName}_name".encode()).decode(),
            seq=self.lmdbTxn.get(f"{readName}_seq".encode()).decode(),
            desc=self.lmdbTxn.get(f"{readName}_desc".encode()).decode(),
            qual=self.lmdbTxn.get(f"{readName}_qual".encode()).decode(),
        )
        return read

    def __readLmdb(self):
        """
        读取lmdb文件
        """
        if not self.keys:
            self.keys = pickle.loads(self.lmdbTxn.get("thisFileIndex".encode()))
        for readName in self.keys:
            yield self.__readFastqReadFromLmdb(readName)

    def iter(self):
        """
        迭代
        """
        if self.useIndex:
            return self.__readLmdb()
        else:
            return self.__readFastq()

    def close(self):
        self.lmdbEnv.close()