
from ont_fast5_api.fast5_interface import get_fast5_file
from collections import namedtuple
from loguru import logger


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

    def __str__(self):
        return f"{self.name}:\n{self.seq}"

    __repr__ = __str__

    def getAnti(self):
        return Fastq(self.name, getAntisense(self.seq), self.desc, self.qual[::-1])


class Fasta:
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq

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


def readFasta(path):
    """
    @description: 读fasta
    @param {type} fasta路径
    @return: 一个迭代器
    """

    def _readFasta(path):
        with open(path, "r") as fh:
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

    return _readFasta(path)


def readFastq(path, length=False):
    """
    @description: 读fastq
    @param {type} fastq路径, 读取长度从3'算
    @return: 一个迭代器
    """

    def _readFastq(path):
        with open(path, "r") as fh:
            i = 0
            readContent = []
            while True:
                lineContent = fh.readline()
                if lineContent == "":
                    break
                i += 1
                readContent.append(lineContent.strip())
                if i % 4 == 0:
                    if not length:
                        read = Fastq(
                            name=readContent[0][1:].split(" ")[0],
                            seq=readContent[1],
                            desc=readContent[2],
                            qual=readContent[3],
                        )
                    else:
                        read = Fastq(
                            name=readContent[0][1:].split(" ")[0],
                            seq=readContent[1][:length],
                            desc=readContent[2],
                            qual=readContent[3][:length],
                        )
                    yield read
                    readContent = []

    return _readFastq(path)