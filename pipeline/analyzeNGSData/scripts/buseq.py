import os
import bupy
import collections
import gzip
from collections import defaultdict
import pysam
import gc
import sys
from bisect import bisect_right
from math import ceil
import itertools
import pandas as pd

"""
pysam的start和end是0-based，但end不被包含在内。因此转化为1-based的，是start是start+1，end还是end。
"""

codon2aa = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}


class FastaSeq():
    
    def __init__(self, seqid="", seq=""):
        self.id = seqid
        self.seq = seq.upper()
        self._tool = FastaSeqTool()
        
    def complement(self):
        return self._tool.complement(self.seq)

    def bin(self, bin_length=100):
        return self._tool.bin(self.seq)
    
    def revcom(self):
        return self._tool.revcom(self.seq)
    
    def base_stat(self):
        #seq = self.seq.upper()
        return self._tool.base_stat(self.seq)

    def gc_stat(self):
        return self._tool.gc_stat(self.seq)
         
    def extract_seq_by_pos(self, pos):
        return self._tool.extract_seq_by_pos(self.seq)
            
    def extract_seq_by_pos_rename(self, seq, seq_id, pos=[], new_name=""):
        #pos: [start, end, strand]
        seq_id = self.id
        seq = self.seq
        return self._tool.extract_seq_by_pos_rename()
    
class FastaSeqTool():

    def __init__(self):
        pass
    
    def translate(self, seq):
        end = len(seq) - (len(seq) %3) - 1
        aas = []
        for i in range(0, end, 3):
            codon = seq[i:(i+3)]
            if codon in codon2aa:
                aas.append(codon2aa[codon])
            else:
                aas.append("N")
        return "".join(aas)
    
    def complement(self, seq):
        seq = seq.upper()
        basecomplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N':'N'}
        
        def _com(base):
            try:
                return basecomplement[base]
            except:
                return "N"
        letters = list(seq)
        letters = [_com(base) for base in letters]
        return ''.join(letters)
    
    def revcom(self, seq):
        return self.complement(seq[::-1])
    
    def rna2dna(self, seq):
        return seq.replace("U", "T").replace("u", "t")
    
    def dna2rna(self, seq):
        return seq.replace("T", "U").replace("t", "u")
        
    def upper(self, seq):
        return seq.upper()
    
    def revcom_rename(self, seqid, seq, rename=1):
        
        def _rename_revcom(name):
            return name + "_revcom"
        seq = self.revcom(seq)
        if rename:
            seqid = _rename_revcom(seqid)
        return [seqid, seq]
    
    def base_stat(self, seq):
        #seq = seq.upper()
        base_count = collections.defaultdict(int)
        for s in seq:
            base_count[s] += 1
        return(base_count)

    def gc_stat(self, seq):
        base_count = self.base_stat(seq)
        gc = base_count["G"] + base_count["C"]
        at = base_count["A"] + base_count["T"]
        gc_ratio = gc * 1.0 / (at + gc)
        return(gc_ratio)
    
    def bin(self, seq, bin_length=100):
        i = 0
        bin_length = int(bin_length)
        while 1:
            start = i*bin_length
            r = seq[start:(start+bin_length)]
            if not r: break
            yield(r)
            i += 1
            
    def extract_seq_by_pos(self, seq, pos):
        #pos: [start, end, strand]
        if not pos:
            return(seq)
        else:
            start, end, strand = pos
            start = int(start)
            if end == ":" or int(end == 0) or not end:
                end == len(seq)
            else:
                end == int(end)
            if not strand:
                strand == "-"
            out_seq = ""
            if start < 1: start = 1
            out_seq = seq[(start-1):end]
            if strand == "-" or strand == "C":
                out_seq = self.revcom(out_seq)  
            return(out_seq)
    
    def pos2name(self, seq_id, pos, format=1):
        
        if format == 1:
            if not pos:
                return(seq_id)
            else:
                start, end, strand = pos
                return(seq_id + ":" + str(start) + "-" + str(end) + ":" + strand)
    
    def extract_seq_by_pos_rename(self, seq, seq_id, pos=[], new_name=""):
        #pos: [start, end, strand]
        result_seq = self.extract_seq_by_pos(seq, pos)
        if new_name:
            result_id = new_name
        else:
            result_id = self.pos2name(seq_id, pos)
        return((result_id, result_seq))

class Fastq():
    
    def __init__(self):
        pass
        
    def read(self, filein, return_seq_only = False):
        
        if os.path.splitext(filein)[1] == ".gz":
            LS_FASTQ_IN = gzip.open(filein, 'rt') #gzip.open default mode 'rb'
            is_gz_file = 1
        else:
            LS_FASTQ_IN = open(filein)
            is_gz_file = 0
        while 1: 
            #这里可能不是gz文件就不用解码了
            #if is_gz_file:
            #    seq_name = LS_FASTQ_IN.readline().decode().rstrip() #LS_FASTQ_IN.readline().rstrip() #python2.7
            #else:
            #    seq_name = LS_FASTQ_IN.readline().rstrip()
            seq_name = LS_FASTQ_IN.readline().rstrip()
            if not seq_name: break
            if is_gz_file:
                seq = LS_FASTQ_IN.readline().decode().rstrip()
                nm = LS_FASTQ_IN.readline().decode().rstrip()
                quality = LS_FASTQ_IN.readline().decode().rstrip()
            else:
                seq = LS_FASTQ_IN.readline().rstrip()
                nm = LS_FASTQ_IN.readline().rstrip()
                quality = LS_FASTQ_IN.readline().rstrip()
            if return_seq_only:
                yield(seq)
            else:
                yield((seq_name, seq, nm, quality))
        
        LS_FASTQ_IN.close()
    
    def filter(self, data, func):
        for d in data:
            if func(d):
                yield(d)
    
    def read_filter(self, filein, func):
        return(self.filter(self.read(filein), func))
    
    def read_filter_by_lengh(self, filein, min_length=0, max_length=0):
        
        def ls_filter_func(d):
            seq_length = len(d[1])
            if seq_length < min_length:
                return 0
            if max_length and seq_length > max_length:
                return 0
            return 1
            
        return(self.read_filter(filein, ls_filter_func))
    
    def read_write_filter(self, filein, fileout, func):
        self.write(self.filter(self.read(filein), fileout, func))
    
    def read_filter_write_by_length(self, filein, fileout, min_length=0, max_length=0):
        self.write(self.read_filter_by_lengh(filein, min_length, max_length), fileout)
    
    def write(self, data, fileout, no_n=True):
        #data:
        #[[seq_name, seq, nm, quality],[]]
        if os.path.splitext(fileout)[1] == ".gz":
            LS_FASTQ_OUT = gzip.open(fileout, 'wb') #gzip.open default mode 'rb'
            is_gz_file = 1
        else:
            LS_FASTQ_OUT = open(fileout, 'w')
            is_gz_file = 0
            
        for d in data:
            if no_n:
                l = "\n".join(d) + "\n"
                if is_gz_file:
                    l = l.encode()
                LS_FASTQ_OUT.write(l)
            else:
                l = "".join(d)
                if is_gz_file:
                    l = l.encode()
                LS_FASTQ_OUT.write(l)         
    
    def count_seq(self, filein):
        return(collections.Counter(self.read(filein, True)))
    
    def stat_length(self, filein):
        return(collections.Counter([len(seq) for seq in self.read(filein, True)]))
    
    def write_stat_length(self, filein, fileout):
        counts = dict(self.stat_length(filein))
        with open(fileout, 'w') as o:
            for lenght in list(counts.keys()):
                count = counts[length]
                o.write(f"{length}\t{count}\n")
    
    def write_count_seq(self, filein, fileout, fasta_or_count = True, sort = True):
        counts = dict(self.count_seq(filein))
        seqs = list(counts.keys())
        if sort: seqs.sort()
        if fasta_or_count:
            with open(fileout, 'w') as o:
                for seq in seqs:
                    o.write(">%s\n%s\n" % (seq, seq))
        with open(fileout, 'w') as o:
            for seq in seqs:
                count = counts[seq]
                o.write("%s\t%s\n" % (seq, count))

class BuPosFile():
    
    #[seq_id, new_name, [satrt, end, strand]]
    
    def __init__(self, file=""):
        self.loadfile(file)
    
    def loadfile(self, file=""):
        self._file = file
       
    def parse(self, file=""):
        if file:
            self.loadfile(file)
        filein = self._file
        #data_file
        #seq_name               
        #seq_name,out_seq_name
        #seq_name,start,end,strand
        #seq_name,start,end,strand,out_seq_name
        #seq_name,start,end,strand,out_seq_name 
        for l in open(filein):
            d = l.rstrip("\n").split(",")
            ld = len(d)
            if ld == 0:
                continue
            elif ld == 1:
                pos = [d[0]]
                new_name = ""
            elif ld == 2:
                seq_id, new_name == d
            else:
                seq_id, start, end = d[:3]
                start = int(start)
                end = int(end)
                if ld >= 4:
                    strand = d[3]
                else:
                    strand = "+"
                if ld >= 5:
                    new_name = d[4]
                else:
                    new_name = ""
                pos = [seq_id, start, end, strand]
            yield([new_name, [pos]])

class BuMergePosFile():
    
    #return [new_name, [[seqid1], [seqid2, satrt, end, strand]]]
    
    #data_file
    #seq_name
    #seq_name,out_seq_name
    #seq_name1,seq_name2:start:end,seq_name3:start:end:strand,out_seq_name
        
    def __init__(self, file=""):
        self.loadfile(file)
    
    def loadfile(self, file=""):
        self._file = file
       
    def parse(self, file=""):
        
        def extract_poss(poss):
            
            def _extract_pos(pos):
                d = pos.split(":")
                if len(d) <= 0: #equal 0 raise error
                    return [d[0]]
                else:
                    start, end = int(d[1]), int(d[2])
                    try:
                        strand = d[3]
                    except:
                        strand = "+"
                    return [d[0], start, end, strand]
    
            return [_extract_pos(pos) for pos in poss]
        
        def get_new_name(pos):
            if len(pos) == 0:
                return pos
            else:
                return "%s:%s-%s:%s" % (pos[0], pos[1], pos[2], pos[3])
        
        if file:
            self.loadfile(file)
        filein = self._file
        for l in open(filein):
            d = l.rstrip("\n").split(",")
            ld = len(d)
            if ld == 0:
                continue
            elif ld == 1:
                poss = extract_poss(d)
                new_name = get_new_name(poss[0])
            else:
                new_name = d[-1]
                poss = extract_poss(d[:-1])
            yield([new_name, poss])

class PosFileParser():
    
    def __init__(self, file, pos_file_class_name):
        self._parser = pos_file_class_name(file)
        
    def parse(self):
        return getattr(self._parser, "parse")()

class FastaSeqsTool():
    
    _tool = FastaSeqTool()
    _filter_funcs = { "revcom": _tool.revcom_rename

                    }
    
    def iter_read_fasta_from_file(self, file, split_id=0, split_func=0):
        
        def describ2id(seq_describ, split_id = 0):
            if split_func:
                seq_id = split_func(seq_describ)
            else:
                if split_id == 0:
                    seq_id = seq_describ.split()[0]
                elif split_id == 1:
                    seq_id = seq_describ
                else:
                    seq_id = seq_describ.split(split_id)[0]
            return(seq_id)
    
        def formate_seq(seq):
            return(seq)
        
        seq_id = ""
        seq_describ = ""
        seq = ""
        for line_num, l in enumerate(open(file)):
            l = l.strip()
            if l.startswith(">"):
                if seq_id:
                    #if not seq:
                    #    raise Exception("Line %s: No sequence for '%s'!" % (line_num, seq_id))
                    yield( (seq_id, seq) )
                seq = ""
                seq_describ = l[1:]
                seq_id = describ2id(seq_describ, split_id)
            else:
                seq += formate_seq(l)
        if seq_id:
            #if not seq:
            #    raise Exception("Line %s: No sequence for '%s'!" % (line_num, seq_id))
            yield( (seq_id, seq) )
            
    def fasta2dict(self, file, split_id=0, split_func=0):

        result_dict = {}
        seq_ids = []
        
        for seq_id, seq in self.iter_read_fasta_from_file(file, split_id, split_func):
            result_dict[seq_id] = seq
            seq_ids.append(seq_id)
        return([seq_ids, result_dict])
        
    def cut_fasta(self, seqs, cut=10, length=0, part=0, max_value=0):
        
        #seqs = self.iter_seqs()
        def _MG2num(s):
            if not s:
                return 0
            last_s = s[-1]
            last_s.upper()
            if last_s == "K":
                return float(s[:-1]) * 1000
            elif last_s == "M":
                return float(s[:-1]) * 1000000
            elif last_s == "G":
                return float(s[:-1]) * 1000000000
            elif last_s == "T":
                return float(s[:-1]) * 1000000000000
            elif last_s == "P":
                return float(s[:-1]) * 1000000000000000
            else:
                return float(s)
        
        def _cal_seq_length(s):
            return len(s[1])
        '''
        if part=0, use cut. if part != 0, use part and length to generate cut
        '''
        
        max_value = _MG2num(max_value)
        for num, part_seqs in bupy.iter_cuts(seqs, cut=cut, part=part, length=length, max_value=max_value, max_func=_cal_seq_length):
            yield([num, part_seqs])
    
    def write_fasta(self, seqs, fileout, line_max_word=0):
        bin_func = FastaSeqTool().bin
        with open(fileout, "w") as o:
            for seq_id, seq in seqs:
                o.write(">" + seq_id + "\n")
                if not line_max_word:
                    o.write(seq + "\n")
                else:
                    for s in bin_func(seq, line_max_word):
                        o.write(s + "\n")
    
    #def __getattr__(self, attribute):
    
    def revcom(self, seqs, print_all=0, rename=1):
        revcom_func = FastaSeqTool().revcom_rename
        retain_flag = 1
        for seqid, seq in seqs:
            new_seqid, new_seq = revcom_func(seqid, seq, rename)
            if print_all:
                yield([new_seqid, new_seq, retain_flag])
            else:
                yield([new_seqid, new_seq])
                
    def upper(self, seqs, print_all=0):
        retain_flag = 1
        for seqid, seq in seqs:
            seq = seq.upper()
            if print_all:
                yield([seqid, seq, retain_flag])
            else:
                yield([seqid, seq])
                
    def rna2dna(self, seqs, print_all=0):
        rna2dna_func = FastaSeqTool().rna2dna
        retain_flag = 1
        for seqid, seq in seqs:
            seq = rna2dna_func(seq)
            if print_all:
                yield([seqid, seq, retain_flag])
            else:
                yield([seqid, seq])

class FastaSeqs():
    
    '''
    When you use `fastas = FastaSeqs(file)`, the file name was recorded, 
    but the sequences haven't been read until you use fastas.load_seq() or 
    other method which use load_seq method.
    You also can extend seq data by load_seq(file=file_name) or load_seq(seq=seqs).
    Write a new method 
    '''
    _tool = FastaSeqsTool()
    _filter_funcs = {"upper": _tool.upper, 
                     "revcom": _tool.revcom,
                     "rna2dna": _tool.rna2dna}
    _extract_seq_by_pos_func =  FastaSeqTool().extract_seq_by_pos
    _pos_type2class = {"bu_pos_file": BuPosFile, "bu_merge_pos_file": BuMergePosFile}
    
    def __init__(self, file=""):
        self._reset()
        self.load_file(file)
        
    
    def _reset(self):
        self._file = ""
        self._filters = []
        self.seqids = []
        self.fastas = {}
    
    def load_file(self, file=""):
        self._file = file
        self.seqids = []
        self.fastas = {}
    
    def load_seq(self, file="", seqs={}, split_id=0, split_func=0):
        '''
        Extend seq data if file, seqs or read ._file .
        '''
        if self._file and not self.fastas:
            self.seqids, self.fastas = self._tool.fasta2dict(self._file, split_id, split_func)
        if file:
            seqids, fastas = self.fasta2dict(file,  split_id, split_func)
            self.seqids.extend(seqids)
            self_fastas = self.fastas
            for seqid, seq in fastas.items():
                self_fastas[seqid] = seq
        if seqs:
            try:
                self.seqids.extend(list(seqs.keys()))
                self_fastas = self.fastas
                for seqid, seq in fastas.items():
                    self_fastas[seqid] = seq
            except:
                seqids = []
                fastas = {}
                for seq_name, seq in seqs:
                    seqids.append(seq_name)
                    fastas[seq_name] = seq
                self.seqids = seqids
                self.fastas = fastas
        
    def iter_seqs(self, bydict=0, split_id=0, split_func=0):
        '''
        if .fastas iter .seqids and .fastas, 
        or iter read ._file.
        Return a generator of [seqid, seq]
        '''
        _tool = self._tool
        filter_funcs = self._filter_funcs
        if not self.fastas:
            seqs = _tool.iter_read_fasta_from_file(self._file, split_id, split_func)
            for filter_name in self._filters:
                #print(filter_funcs[filter_name](["seq","aucg"]))
                seqs = filter_funcs[filter_name](seqs)
            for seqid, seq in seqs:
                yield([seqid, seq])
            ''''
            for seqid, seq in seqs:
                print(seqid, seq)
                yield([seqid, seq])
            '''
        else:
            seqids, fastas = self.seqids, self.fastas
            if bydict:
                for seqid, seq in fastas.items():
                    yield([seqid, seq]) 
            else:
                for seq_id in seqids:
                    yield([seq_id, fastas[seq_id]])
                    
    def rna2dna(self):
        rna2dna_func = FastaSeqTool().rna2dna
        seqids = self.seqids
        fastas = self.fastas
        for seqid in seqids:
            fastas[seqid] = rna2dna_func(fastas[seqid])

    def cut_fasta(self, cut=10, part=0, length=0, max_value=0):
        #Just use for .write_cut_fasta() now.
        seqs = self.iter_seqs()
        return self._tool.cut_fasta(seqs, cut=cut, part=part, length=length, max_value=max_value)
    
    def write_cut_fasta(self, fileout_pre, cut=10, part=0, max_value=0, line_max_word=0):
        for num, seqs in self.cut_fasta(cut, part, max_value=max_value):
            fileout = fileout_pre + "." + str(num) + ".fa"
            self._tool.write_fasta(seqs, fileout, line_max_word)
    
    def write_fasta(self, fileout, line_max_word=0):
        seqs = self.iter_seqs()
        line_max_word = int(line_max_word)
        self._tool.write_fasta(seqs, fileout, line_max_word)
    
    def extract_seq_by_one_pos(self, seq_id, pos):
        #pos: [start, end, strand]
        seq = self.fastas[seq_id]
        return self._extract_seq_by_pos_func(seq, pos)
    
    def extract_seq_by_pos(self, pos_file="", fasta_file="", pos_data=None, pos_type="bu_pos_file", split_id=0, split_func=0):

        if not pos_data:
            pos_class = self._pos_type2class[pos_type]
            pos_data = PosFileParser(pos_file, pos_class).parse()
        if fasta_file:
            seqids, fastas = self._tool.fasta2dict(fasta_file, split_id, split_func)
        else:
            if self.fastas:
                fastas = self.fastas
            else:
                fastas = self.fasta2dict(self._file, split_id, split_func)
        extract_seq_by_pos = FastaSeqTool().extract_seq_by_pos
        pos2name = FastaSeqTool().pos2name
        for new_name, poss in pos_data:
            if not new_name:
                new_name = pos2name(poss[0][0], poss[0][1:])
            new_seqs = []
            no_seq_id = 0
            for pos in poss:
                seq_id = pos[0]
                if seq_id in fastas:
                    new_seqs.append(extract_seq_by_pos(fastas[seq_id], pos[1:]))
                else:
                    no_seq_id = 1
                    print(seq_id + " : not found!!!")
                    break
            if not no_seq_id:
                new_seq = new_seqs[0] if len(new_seqs) == 1 else ''.join(new_seqs)
                yield(new_name, new_seq)
        
        '''
        #pos_type: bu_pos_file, bed ......        
        if not pos_data:
            pos_class = self._pos_type2class[pos_type]
            pos_data = PosFileParser(pos_file, pos_class).parse()
        if fasta_file:
            seqids, fastas = self._tool.fasta2dict(fasta_file, split_id, split_func)
        else:
            if self.fastas:
                fastas = self.fastas
            else:
                fastas = self.fasta2dict(self._file, split_id, split_func)
        extract_seq_by_pos_rename = FastaSeqTool().extract_seq_by_pos_rename
        for seq_id, new_name, pos in pos_data:
            if seq_id in fastas:
                seq = fastas[seq_id]
                new_id, new_seq = extract_seq_by_pos_rename(seq, seq_id, pos, new_name)
                yield(new_id, new_seq)
            else:
                print(seq_id + " : not found!!!")
        '''
    def extract_and_merge_seq_by_pos(self, pos_file="", fasta_file="", pos_data=None, pos_type="bu_merge_pos_file", split_id=0, split_func=0):
        return self.extract_seq_by_pos(pos_file, fasta_file, pos_data, pos_type, split_id, split_func)

    def extract_seq_by_pos2file(self, pos_file, fasta_file, fileout, line_word_lim = 100, split=0):
        self._tool.write_fasta(self.extract_seq_by_pos(pos_file, fasta_file, split_id=split), fileout, line_word_lim)
    
    def extract_merge_seq_by_pos2file(self, pos_file, fasta_file, fileout, line_word_lim = 100, split=0):
        self._tool.write_fasta(self.extract_and_merge_seq_by_pos(pos_file, fasta_file, split_id=split), fileout, line_word_lim)
    
    def write_seq_length2file(self, fileout):
        '''
        Iter self data.
        '''
        seqs = self.iter_seqs()
        with open(fileout, 'w') as o:
            for seqid, seq in seqs:
                length = len(seq)
                o.write(''.join([seqid, "\t", str(length), "\n"]) )
    
    ######filters#############
    def _process_self_data(self, func, args):
        seqids = self.seqids
        fastas = self.fastas
        new_fastas = {}
        new_seqids = []
        oldid_to_newid = {}
        seqs = self.iter_seqs(bydict=1)
        for seqid, seq, retain_flag in func(seqs, print_all=1, **args):
            if retain_flag:
                new_fastas[new_seqid] = new_seq
                oldid_to_newid[seqid] = new_seqid
        for seqid in seqids:
            if seqid in oldid_to_newid:
                new_seqids.append(oldid_to_newid[seqid])
        self.seqids = new_seqids
        self.fastas = new_fastas
    
    def __getattr__(self, attribute):
        if attribute in self._filter_funcs:
            func = self._filter_funcs[attribute]
            #func = getattr(self._tool, attribute)
            def _filter_func(change_self=1, args={}):
                if change_self:
                    self._filters.append(attribute)
                    if self.fastas:
                        self._process_self_data(func, args)
                else:
                    seqs = self.iter_seqs()
                    return func(seqs, **args)
            return _filter_func

class BsmapMethratio():
    
    #使用完后，需用close方法关闭文件句柄
    
    def __init__(self, filein):
        self.filein = filein
        self.check_format()
        self.have_open = False
        self.open()
    
    def open(self):
        if not self.have_open:
            if self.is_gz:
                self.IN = gzip.open(self.filein, 'rt')
            else:
                self.IN = open(self.filein)
            if self.is_tbx:
                self.tbx = pysam.TabixFile(self.filein)
        self.have_open = True
    
    def close(self):
        try:
            self.IN.close()
            self.tbx.close()
        except:
            pass
        self.have_open = False
       
    def check_format(self):
        #自动检测是否为gz格式
        #自动检测是否为原始格式（列数大于等于8）或简化的格式（列数为6）
        #原始格式：
        #chr_name, pos, strand, context, ratio, eff_CT_count, C_count, CT_count
        #unmethlyated C was converted to T
        #not return eff_CT_count
        #the eff_CT_count is new and is calculated by adjusting the
        # methylation ratio for the C/T SNP, using the reverse strand
        # mapping information.
        #简化的格式：
        #chr_name, pos, strand, context, C_count, CT_count
        filein = self.filein
        if os.path.splitext(filein)[1] == ".gz":
            self.is_gz = True
            LS_IN = gzip.open(filein, 'rt')
        else:
            self.is_gz = False
            LS_IN = open(filein)
        next(LS_IN)
        l = next(LS_IN)
        d = l.rstrip("\n").split("\t")
        if len(d) >= 8:
            self.is_origin_format = True
        else:
            self.is_origin_format = False
        if os.path.exists(filein + ".tbi"):
            self.is_tbx = True
        else:
            self.is_tbx = False
        LS_IN.close()
    
    def iter_read_line(self):
        self.close()
        self.open()
        if self.is_origin_format: next(self.IN)
        #if self.is_tbx: #速度相似，暂不使用了
        #    records = self.tbx.fetch()
        #    for d in self._parse_tbx_lines(records):
        #        yield(d)
        #else:
        for l in self.IN:
            d = l.rstrip("\n").split("\t")
            if len(d) >= 8:
                chr_name, pos, strand, context, ratio, eff_CT_count, C_count, CT_count = d[:8]
            else:
                chr_name, pos, strand, context, C_count, CT_count = d
            pos = int(pos)
            C_count, CT_count = int(C_count), int(CT_count)
            yield([chr_name, pos, strand, context, C_count, CT_count])
    
    def _parse_tbx_line(self, l):
        chr_name, pos, strand, context, C_count, CT_count = l.split("\t")
        pos = int(pos)
        C_count, CT_count = int(C_count), int(CT_count)
        return([chr_name, pos, strand, context, C_count, CT_count])
        
    def _parse_tbx_lines(self, lines):
        for l in lines:
            yield(self._parse_tbx_line(l))
    
    def read_feature(self, feature, up=0, down=0, to_list=True):
        #only for tbi method
        feature_id, chr_name, feature_start, feature_end = feature[:4]
        s = feature_start - up
        e = feature_end + down
        s -= 1
        if s < 0:
            s = 0
        records = self.tbx.fetch(chr_name, s, e)
        records = self._parse_tbx_lines(records)
        if to_list:
            records = list(records)
        return(feature + [records])

    def iter_read_feature(self, features, up=0, down=0, method=1):
        #features is file or obj
        #如果method = 0, features是字典， key是染色体，value是feature的列表
        #如果method = 1，features是列表 [id_, chr_, start, end, strand]
        if method == 0:
            iter_data = iter_pos_feature(read_bsmap_methratio(self.filein), features, up, down, is_point = True)
            for chr_name, (feature_id, feature_start, feature_end, feature_strand, records) in iter_data:
                yield([feature_id, chr_name, feature_start, feature_end, feature_strand, records])
        else:
            if isinstance(features, str): features = convert_pos_feature(read_pos_file(features, _sorted = True))
            for feature in features:
                yield(self.read_feature(feature, up, down))
    
    def stat_bsm_region_bin(self, filefeature, up=1000, down=1000, bin_number = 100,
                            file_stat_bin_pos = "",
                            file_stat_feature = "",
                            file_all_pos = "", retrun_d=False, method=1):
        #resume memory
        #return
        a = self.iter_read_feature(filefeature, up, down, method)
        d = []
        for feature_id, chr_name, feature_start, feature_end, feature_strand, records in a:
            feature_is_plus = feature_strand == "+"
            feature_length = feature_end - feature_start + 1
            for na, pos, base_strand, context, c_count, ct_count in records:
                if pos < feature_start:
                    rel_pos = feature_start - pos # 1~1000
                    if feature_is_plus:
                        pos_type = 0 # 0: up, 1: in, 2: down
                    else:
                        pos_type = 2
                elif pos <= feature_end:
                    pos_type = 1 
                    if feature_is_plus:
                        rel_pos = pos - feature_start + 1 # 0 ~ n - 1
                    else:
                        rel_pos = feature_end - pos + 1 # 0 ~ n - 1
                else:
                    rel_pos = pos - feature_end
                    if feature_is_plus:
                        pos_type = 2
                    else:
                        pos_type = 1
                d.append([feature_id, feature_length, pos_type, rel_pos, context, c_count, ct_count])
        d = pd.DataFrame(d, columns=["feature_id", "feature_length", "pos_type", "rel_pos", "context", "c_count", "ct_count"])
    
        f0 = d.pos_type == 0 # 0: up, 1: in, 2: down
        f1 = d.pos_type == 1
        f2 = d.pos_type == 2
   
        d["bin_length"] = up
        d.loc[f2, "bin_length"] = down
        d.loc[f1, "bin_length"] = d["feature_length"][f1]
        d["rel_bin"] = (d["rel_pos"] - 1) * bin_number // d["bin_length"]
        d.loc[f0, "rel_bin"] = -(d["rel_bin"][f0] + 1)
        d.loc[f2, "rel_bin"] = d["rel_bin"][f2] + bin_number
                
        s1 = d.loc[d.pos_type == 1, :].groupby(["feature_id", "context"])[["c_count", "ct_count"]].sum().reset_index()
        s1["c_ratio"] = s1["c_count"]/s1["ct_count"]
        s1.loc[s1["ct_count"] == 0, "c_ratio"] = 0
        
        s2 = d.groupby(["rel_bin", "context"])[["c_count", "ct_count"]].sum().reset_index()
        s2["c_ratio"] = s2["c_count"]/s2["ct_count"]
        s2.loc[s2["ct_count"] == 0, "c_ratio"] = 0
       
        if retrun_d:
            return([s1, s2, d])
        else:
            return([s1, s2])
    
    def _cal_methratio(self, records, min_depth=1):
        this_count_stat = {"CG": [0, 0, 0] , "CHG": [0, 0, 0], "CHH": [0, 0, 0]}
        for d in records:
            context, C_count, CT_count = d[3:6]
            ls = this_count_stat[context]
            ls[0] += C_count
            ls[1] += CT_count
        for context, context_data in this_count_stat.items():
            if context_data[1] >= min_depth:
                context_data[2] = context_data[0] * 1.0 /context_data[1]
        return(this_count_stat)
    
    def _write_methratio(self, iter_data, fileout):
        with open(fileout, 'w') as o:
            o.write("id\tcg_c_count\tcg_ct_count\tcg_c_ratio\tchg_c_count\tchg_ct_count\tchg_c_ratio\tchh_c_count\tchh_ct_count\tchh_c_ratio\n")
            for feature_id, feature_stat in iter_data:
                o.write(feature_id + "\t" + "\t".join([str(i) for i in feature_stat["CG"] + feature_stat["CHG"] + feature_stat["CHH"]]) + "\n")
    
    def iter_cal_methratio(self, file_feature, up=0, down=0, min_depth=1, method=1):
        iter_data = self.iter_read_feature(file_feature, up, down, method)
        for feature_id, chr_name, feature_start, feature_end, feature_strand, records in iter_data:
            this_stat = self._cal_methratio(records, min_depth)
            yield([feature_id, this_stat])
            
    def write_feature_methratio(self, file_feature, fileout, up=0, down=0, min_depth=1, method=1):
        iter_data = self.iter_cal_methratio(file_feature, up, down, min_depth, method)
        self._write_methratio(iter_data, fileout)
        
    def write_all_chr_ratio(self, fileout):
        def iter_cal_chr_methratio():
            iter_chr_data = itertools.groupby(self.iter_read_line(), lambda x: x[0])
            for chr_name, chr_data in iter_chr_data:
                yield([chr_name, self._cal_methratio(chr_data)])
        self._write_methratio(iter_cal_chr_methratio(), fileout)
    
    def stat_chr_bin(self, bin_length=25, min_depth=4):
    
        for chr_name, bin_pos, bin_data in cut_pos2chr_bin(self.iter_read_line(), bin_length):
            bin_stat = self._cal_methratio(bin_data, min_depth)
            yield([chr_name, bin_pos, bin_data, bin_stat])
    
    def write_wig(self, fileout_pre, bin_length=25, min_depth=4):
    
        fileout1 = fileout_pre + ".CG.wig"
        fileout2 = fileout_pre + ".CHG.wig"
        fileout3 = fileout_pre + ".CHH.wig"

        with open(fileout1, 'w') as o1, open(fileout2, 'w') as o2, open(fileout3, 'w') as o3:

            def wirte_count_stat(bin_pos, bin_stat):
                if bin_stat["CG"][1] >= min_depth:
                    ratio = bin_stat["CG"][0]/bin_stat["CG"][1]
                    o1.write(f'{bin_pos}\t{ratio}\n')
                if bin_stat["CHG"][1] >= min_depth:
                    ratio = bin_stat["CHG"][0]/bin_stat["CHG"][1]
                    o2.write(f'{bin_pos}\t{ratio}\n')
                if bin_stat["CHH"][1] >= min_depth:
                    ratio = bin_stat["CHH"][0]/bin_stat["CHH"][1]
                    o3.write(f'{bin_pos}\t{ratio}\n')
            
            last_chr_name = ""
            for chr_name, bin_pos, bin_data, bin_stat in self.stat_chr_bin(bin_length, min_depth):
                if chr_name != last_chr_name:
                    if not last_chr_name:
                        output_line = f'variableStep chrom={chr_name} span={bin_length}\n'
                        o1.write(output_line)
                        o2.write(output_line)
                        o3.write(output_line)
                    last_chr_name = chr_name
                wirte_count_stat(bin_pos, bin_stat)
    
def cut_pos2chr_bin(data, bin_length, file_fasta_fai=""):
    #进阶： file_fasta_fai: 前两列为染色体名和染色体长度, 暂时不支持
    #data: 前两列分别为chr_name, pos
    #只返还含有数据的bin
    #生成器，元素为[pos, bin_data]
    #pos为该bin的start position, 1-based.
    #如bin_length为25时，pos为1，26，51, ...
    #如果data是列表，返回值的元素和data的元素共用一个内存地址
    last_chr_name = ""
    this_pos = 1
    this_data = []
    for d in data:
        chr_name, pos = d[:2]
        if chr_name != last_chr_name:
            if this_data: yield([chr_name, this_pos, this_data])
            last_chr_name = chr_name
            this_pos = 1
            this_data = []
        if pos >= bin_length + this_pos:
            if this_data: yield([chr_name, this_pos, this_data])
            this_pos = ((pos - 1) // bin_length) * bin_length + 1
            this_data = [d]
        else:
            this_data.append(d)
    if this_data: yield([chr_name, this_pos, this_data])
            
        
            
        
                            
def iter_pos_file(filein, parse_pos_func = None, header = False):
    
    """
    迭代位置文件的每一行。
    
    输入参数:
    -------
    filein : feature位置文件路径名。每行存储一个feature，具体格式由
            parse_pos_func参数指定。
    
    parse_pos_func : 函数。或"pos"或者"bed"，指定处理每行的函数名。默认为None，
            根据后缀名判断是pos还是bed。如果以".bed"结尾，则为"bed",
            否则为"pos"。
            "pos": _parse_pos  
                   空字符隔开的文件，每列对应:
                   chr_, id_, start, end, strand
            "bed": _parse_bed
                   空字符隔开的文件，每列对应:
                   chr_, start, end, id, na, strand
            这些函数输入是一行字符串，输出是一个列表：
            [chr_, id_, start, end, strand]。
    
    header : 布尔型值。如果为True，则跳过文件第一行。
        
    输出结果：
    -------
    迭代文件的每一行。每次输出为[chr_, id_, start, end, strand]
    
    """
    
    def _parse_pos(l):
        id_, chr_, start, end, strand = l.rstrip("\n").split()
        start, end = int(start), int(end)
        return [chr_, id_, start, end, strand]
        
    def _parse_bed(l):
        d = l.rstrip("\n").split()
        chr_, start, end, id_, na, strand = d[:6]
        start, end = int(start), int(end)
        return [chr_, id_, start, end, strand]
    
    _parse_pos_func_dict = {"pos": _parse_pos, "bed": _parse_bed}
    if parse_pos_func is None:
        parse_pos_func = "bed" if filein.endswith(".bed") else "pos"
    if parse_pos_func in _parse_pos_func_dict:
        parse_pos_func = _parse_pos_func_dict[parse_pos_func]
    
    with open(filein) as f:
        if header: f.readline()
        for raw_line in f:
            try:
                yield(parse_pos_func(raw_line))
            except:
                print(raw_line)
                chr_, id_, start, end, strand = parse_pos_func(raw_line)
        
def read_pos_file(filein, parse_pos_func = None, header = False, _sorted = False):
    
    """
    利用iter_pos_file处理每行feature，输出一个存储了每个染色体中每个feature位置的字典。
    
    输入参数:
    -------
    filein : feature位置文件路径名。每行存储一个feature，具体格式由
            parse_pos_func参数指定。
    
    parse_pos_func : 函数。或"pos"或者"bed"，指定处理每行的函数名。默认为空，
            根据后缀名判断是pos还是bed。如果以".bed"结尾，则为"bed",
            否则为"pos"。
            "pos": _parse_pos  
                   空字符隔开的文件，每列对应:
                   chr_, id_, start, end, strand
            "bed": _parse_bed
                   空字符隔开的文件，每列对应:
                   chr_, start, end, id, na, strand
            这些函数输入是一行字符串，输出是一个列表：
            [chr_, id_, start, end, strand]。
    
    header : 布尔型值。如果为True，则跳过文件第一行。
    
    _sorted : 布尔型值。每个染色体的feature列表是否按start位置排序。
    
    输出结果：
    -------
    一个字典。key为染色体名，value为该染色体上feature的列表。每个元素
    对应一个feature，每个feature是一个列表[id, start, end, strand].
    
    """

    iter_lines = iter_pos_file(filein, parse_pos_func, header)
    
    pos_dict = defaultdict(list)
    for chr_, id_, start, end, strand in iter_lines:
        pos_dict[chr_].append([id_, start, end, strand])
            
    if _sorted:
        #sort by start position. 1 is the index of start.
        for chr_, chr_data in pos_dict.items():
            chr_data.sort(key=lambda x: x[1])

    return pos_dict

def convert_pos_feature(d):
    
    for chr_, chr_features in d.items():
        for (id_, start, end, strand) in chr_features:
            yield([id_, chr_, start, end, strand])
       
def iter_pos_feature_by_file(filex, filey, up=0, down=0):
    
    def convert_iter_pos_for_iter_pos_feature(d):
        for chr_, id_, start, end, strand in d:
            yield([chr_, start, end, strand, id_])
    
    x_iter = convert_iter_pos_for_iter_pos_feature(iter_pos_file(filex))
    y_data = read_pos_file(filey)

    return iter_pos_feature(x_iter, y_data, up, down)    

def iter_pos_feature_by_file2file(filex, filey, fileout, up=0, down=0):
    
    with open(fileout, 'w') as o:
        for chr_, (y_id, y_start, y_end, y_strand, xs) in iter_pos_feature_by_file(filex, filey, up, down):
            for x_chr_, x_start, x_end, x_strand, x_id in xs:
                o.write("\t".join([str(s) for s in [x_id, chr_, x_start, x_end, x_strand, y_id, y_start, y_end, y_strand]]) + "\n")

def filter_bam_aligns(aligns, filter_tag=["unmapped"]):
    
    def _generate_bam_filter_func(filter_tag=["unmapped"]):
    
        """
        输入参数:
        filter_tag: 字符串或函数组成的列表。如果是字符串，则根据
        """
    
        tag2func = {
            "unmapped": lambda read: read.is_unmapped,
            "clean": lambda read: read.get_tag("TP") != "clean",
            "sRNA_map_only_one": lambda read: read.qname.split("_")[2] != 1,
            "remove_duplicated": lambda read: read.is_duplicate,
            "map_only_one": lambda read: read.get_tag("NH") != 1 #for hisat2
        }
    
        funcs = [tag2func[tag] for tag in filter_tag]
    
        def _filter(read):
            for _func in funcs:
                if _func(read):
                    return True
            return False
    
        return _filter
    
    filter_read_func = _generate_bam_filter_func(filter_tag)
    for read in aligns:                
        if filter_read_func(read): continue
        chr_ = read.reference_name
        #id_ += 0
        start = read.reference_start + 1
        strand = "-" if read.is_reverse else "+" 
        end = read.reference_end
        yield([ chr_, start, end, strand, read]) #str(id_),
    
def iter_bam(bam_file, filter_tag=["unmapped"]):
    
    aligns = pysam.AlignmentFile(bam_file, "rb").fetch()
    return(filter_bam_aligns(aligns, filter_tag))

def iter_chr_bam(bam_file, filter_tag=["unmapped"]):
    
    bam_iter = iter_bam(bam_file, filter_tag)
    return bupy.cut_iter(bam_iter)

def filter_sRNA_align_shortstack(aligns, filter_tag=[], min_len=20, max_len=30, no_map_time=False):
    #filter_tag和no_map_time not be used, 为以后使用。也为了方便和其他函数兼容。
    for read in aligns:
        
        is_unique = read.get_tag("XY") == "U"
        #if "unmapped" in filter_tag and read.is_unmapped: continue
        ##if "clean" in filter_tag and read.get_tag("TP") != "clean": continue
        #if "sRNA_map_only_one" in filter_tag and not is_unique: continue
        map_time = read.get_tag("XX")

        chr_ = read.reference_name
        start = read.reference_start + 1
        strand = "-" if read.is_reverse else "+" 
        end = read.reference_end
        length = end - start + 1
        if length < min_len or length > max_len: continue 
        yield([chr_, start, end, strand, length, map_time])

def filter_sRNA_align(aligns, filter_tag, min_len=20, max_len=30, no_map_time=False):
    
    last_seq, last_start = "", -1
    for read in aligns:
    
        if "unmapped" in filter_tag and read.is_unmapped: continue
        if "clean" in filter_tag and read.get_tag("TP") != "clean": continue
    
        name = read.qname
        if no_map_time:
            seq, count = name.split("_")
            count = int(count)
            map_time = 1
        else:
            seq, count, map_time = name.split("_")
            count, map_time = int(count), int(map_time)
        
        if "sRNA_map_only_one" in filter_tag and map_time != 1: continue   
        chr_ = read.reference_name
        start = read.reference_start + 1
        strand = "-" if read.is_reverse else "+" 
        end = read.reference_end
    
        length = end - start + 1
        if length < min_len or length > max_len: continue
    
        if start == last_start and seq == last_seq: continue
        last_seq, last_start = seq, start
    
        yield([chr_, start, end, strand, seq, length, count, map_time])

def iter_sRNA_bam(file_bam, filter_tag=["clean"], min_len=20, max_len=30, no_map_time=False):
    
    """
    输入参数:
    -------
    bam_file : 小RNA比对文件路径。
    
    """
    filter_tag = set(filter_tag)
    
    bam_obj = pysam.AlignmentFile(file_bam, "rb")

    aligns = bam_obj.fetch()
    for d in filter_sRNA_align(aligns, filter_tag, min_len=20, max_len=30, no_map_time=no_map_time):
        yield(d)
                 
def iter_pos_feature(aligns, features, up=0, down=0, is_point=0, simple_chr=False):

    """
    如果algins是一个点，也可以用这个。效率应该差不太多。
    
    Parameters
    ----------
    aligns : iterator. each element is a `align_data`:
            if is_point == 0: [chr_, start, end, ....] 
            if is_point == 0: [chr_, start, ...] 
            按照chr和起始位置进行排序。不管strand是"+"还是"-",
            start位置小于end位置。计算过程中也未考虑strand信息。
    
    features : dict. key is `chr_`, value is the list of feature 
            (`feature_data`) sorted by start position. each feature is represented by a list. 
            the function only use the second and the third element as start 
            and end position, 一个经典的`feature_data`是[id, start, end, strand].
            The start is less than end no matther what 
            the strand is.
            feature also can be file (txt or bed)
    
    up : 碱基数，默认值0。如果align与feature以及feature上游up碱基有重叠，即
            该align就属于该feature。
    
    down : 碱基数，默认值0。如果align与feature以及feature下游down碱基有重叠，即
            该align就属于该feature。
    
    pos_format: if is_point == 0: [chr_, start, end, ....] 
                if is_point == 0: [chr_, start, ...]
        
    Return values
    ----------
    迭代器。每个元素代表一个feature及其包含的align。[chr_, feature_data]. 
    `feature_data`是拷贝`features`中存储的`feature_data`，后面再添加一个
    元素，存储这个feature含有的`align_data`的列表。实际计算过程中是先直接在
    `features`中的`feature_data`中添加元素，当某feature被扫面完后，在将
    其值拷贝，输出出来，然后再去把原来`features`中存储的值的最后一个元素去掉。
    
    2019.05.05之前对于feature有重叠的处理的不好。需要重新更改。
    """
    
    if isinstance(features, str):
        features = read_pos_file(features, _sorted = True)
    
    this_chr = ""
    this_feature_aligns = []
    scaned_chr = set()
    scan_this_chr_feature = 0
    ls_test_i = 0
    for align_data in aligns:
        ls_test_i += 1
        if is_point:
            align_chr, align_start = align_data[:2]
            align_end = align_start
        else:
            align_chr, align_start, align_end = align_data[:3]
        if align_chr != this_chr:
            scaned_chr.add(align_chr)
            if this_chr:
                #扫描完一个染色体后，将剩余还没输出的feature输出出去
                for d in this_feature_data[feature_indicator:]:
                    yield([this_chr, d.copy()])
                    d.pop()
            this_chr = align_chr
            this_feature_data = features[this_chr]
            for d in this_feature_data:
                d.append([])
            feature_indicator = 0
            scan_this_chr_feature = 0
            feature_num = len(this_feature_data)
        if scan_this_chr_feature: continue
        pad_align_start = align_start - down
        pad_align_end = align_end + up
        for i in range(feature_indicator, feature_num):
            feature_info = this_feature_data[i]
            if pad_align_end < feature_info[1]:
                break
            elif pad_align_start > feature_info[2]:
                #/2019.05.05修改
                #feature_indicator = i + 1
                #yield([this_chr, feature_info.copy()])
                #this_feature_data[i].pop()
                #if feature_indicator == feature_num:
                #    scan_this_chr_feature = 1
                #continue
                if i == feature_indicator:
                    feature_indicator = i + 1
                    yield([this_chr, feature_info.copy()])
                    this_feature_data[i].pop()
                    if feature_indicator == feature_num:
                        scan_this_chr_feature = 1
                    continue
                #2019.05.05修改/
            else:
                feature_info[-1].append(align_data)
    if this_chr:
        #扫描完后，将最后一个染色体剩余还没输出的feature输出出去
        for d in this_feature_data[feature_indicator:]:
            yield([this_chr, d.copy()])
            d.pop()
    for chr_, feature_data in features.items():
        #将没有被扫描到的染色体的feature输出出去
        if chr_ not in scaned_chr:
            for d in feature_data:
                d1 = d.copy()
                d1.append([])
                yield([chr_, d1])
                
def iter_bam_feature(file_bam, features_or_bed_file, filter_bam_func=filter_bam_aligns, 
                    filter_bam_params={"filter_tag":["unmapped"]}, up=0, down=0, method=1):
    #如果method == 0， features_or_bed_file只能为file
    if isinstance(features_or_bed_file, str):
        origin_features = read_pos_file(features_or_bed_file, _sorted = True)
        features = None
    else:
        origin_features = None
        features = features_or_bed_file
    
    bam_obj = pysam.AlignmentFile(file_bam, "rb")
    
    if method == 1:
        bam_obj = pysam.AlignmentFile(file_bam, "rb")
        if features is None:
            features = convert_pos_feature(origin_features)
        for feature in features:
            id_, chr_, start, end, strand = feature
            aligns = bam_obj.fetch(contig=chr_, start=start, end=end)
            aligns = filter_bam_func(aligns, **filter_bam_params)
            yield([feature, aligns])
    else:
        aligns = bam_obj.fetch()
        aligns = filter_bam_func(aligns, **filter_bam_params)       
        for chr_name, (gene_id, start, end, block_strand, this_aligns) in iter_pos_feature(aligns, origin_features):
            yield([[gene_id, chr_name, start, end, block_strand], this_aligns])

def iter_sRNA_feature(file_bam, features_or_bed_file, filter_tag=["clean"], min_len=20, max_len=30, up=0, down=0, method=1, no_map_time=False, bam_format ="sim_bam"):
    #用method=2是用自己写的方法输出数据
    #注意输出顺序和原先的file_bed顺序不一定完全一致。
    #因为没有小RNA比对上的chr（例如一些contig）会最后输出。
    #因此不同样品的输出顺序也不一定一样。
    
    if bam_format == "short_stack":
        read_bam_func = filter_sRNA_align_shortstack
        filter_bam_params = {"min_len": min_len,
                             "max_len": max_len}
    else:
        read_bam_func = filter_sRNA_align
        filter_bam_params={"filter_tag": filter_tag, 
                           "min_len": min_len,
                           "max_len": max_len,
                           "no_map_time": no_map_time}
    
    return iter_bam_feature(file_bam, features_or_bed_file, 
                     filter_bam_func=read_bam_func, 
                     filter_bam_params=filter_bam_params,
                     up=up, down=down, method=method)
 
def iter_sRNA_feature_old(file_bam, features_or_bed_file, filter_tag=["clean"], min_len=20, max_len=30, up=0, down=0, method=1):
    #2019.05.04以前的，仍然可用。只是没必要这样写了。
    
    if isinstance(features_or_bed_file, str):
        origin_features = read_pos_file(features_or_bed_file, _sorted = True)
        features = None
    else:
        origin_features = None
        features = features_or_bed_file
    
    if method == 1:
        bam_obj = pysam.AlignmentFile(file_bam, "rb")
        if features is None:
            features = convert_pos_feature(origin_features)
        for feature in features:
            id_, chr_, start, end, strand = feature
            aligns = bam_obj.fetch(contig=chr_, start=start, end=end)
            aligns = filter_sRNA_align(aligns, filter_tag, min_len=min_len, max_len=max_len)
            yield([feature, aligns])
    else:
        sRNA_bam_aligns = iter_sRNA_bam(file_bam, ["clean"], min_len=min_len, max_len=max_len)
        for chr_name, (gene_id, start, end, block_strand, aligns) in iter_pos_feature(sRNA_bam_aligns, origin_features):
            yield([[gene_id, chr_name, start, end, block_strand], aligns])

def write_region_sRNA(file_bam, file_bed, fileout, filter_tag=["clean"], min_len=20, max_len=30, up=0, down=0, method=2):
    iter_feature_align_data = iter_sRNA_feature(file_bam, file_bed, filter_tag=filter_tag, min_len=min_len, max_len=max_len, up=up, down=down, method=method)
    
    with open(fileout, 'w') as o:
        o.write("\t".join(["cluster_id", "chr", "pos", "end", "strand",  "seq",  "length", "count", "map_time"]) + "\n")
        for (gene_id, chr_name, start, end, block_strand), aligns in iter_feature_align_data:
            for sRNA_align in aligns:
                o.write(gene_id + "\t" + "\t".join([str(s) for s in sRNA_align]) + "\n")

def cluster_sRNA(sRNAs, chr_column_index=0, start_column_index=1, end_column_index=2, window=100, return_sRNA=False):
    
    """
    输入参数:
    sRNAs : 小RNA位置组成的迭代器或列表。每个元素是一个列表，表示小RNA的位置。需含有
        染色体名称，起始位置和终止位置。
    chr_column_index : 输入sRNAs的每个元素中，染色体名称所在的索引，0-based。默认0。
    start_column_index : 输入sRNAs的每个元素中，起始位置所在的索引，0-based。默认1。
    end_column_index : 输入sRNAs的每个元素中，终止位置所在的索引，0-based。默认2。
    window : 同一个cluster中紧邻的两个sRNA中，前一个小RNA的终止位置和
        后一个小RNA的起始位置相差不超过window大小。如当window设为100时，sRNA1的
        终止位置是100，sRNA2的起始位置是200。则两个sRNA是一个cluster。而如果sRNA2
        的起始位置是201。则两个sRNA不是一个cluster。默认100。
    """
    
    this_cluster_chr = ""
    this_cluster_end = 0
    for sRNA in sRNAs:
        chr_, start, end = sRNA[chr_column_index], sRNA[start_column_index], sRNA[end_column_index]
        if chr_ != this_cluster_chr:
            if this_cluster_chr:
                if return_sRNA:
                    yield([this_cluster_chr, this_cluster_start, this_cluster_end, ls_sRNAs])
                else:
                    yield([this_cluster_chr, this_cluster_start, this_cluster_end])
            this_cluster_chr = chr_
            this_cluster_start = start
            this_cluster_end = end
            ls_sRNAs = [sRNA]
        elif this_cluster_end + window < start:
            if return_sRNA:
                yield([this_cluster_chr, this_cluster_start, this_cluster_end, ls_sRNAs])
            else:
                yield([this_cluster_chr, this_cluster_start, this_cluster_end])
            this_cluster_chr = chr_
            this_cluster_start = start
            this_cluster_end = end
            ls_sRNAs = [sRNA]
        else:
            #原先是this_cluster_end = end是错误的，201912修改
            ls_sRNAs.append(sRNA)
            if end > this_cluster_end:
                this_cluster_end = end
    #201912修改
    if this_cluster_chr:
        if return_sRNA:
            yield([this_cluster_chr, this_cluster_start, this_cluster_end, ls_sRNAs])
        else:
            yield([this_cluster_chr, this_cluster_start, this_cluster_end])

def sRNA_cluster2bed(clusters, fileout):
    
    with open(fileout, 'w') as o:
        for chr_, start, end in clusters:
            start, end = str(start), str(end)
            id_ = chr_ + ":" + start + "-" + end
            o.write("\t".join([chr_, start, end, id_, ".", "+"]) + "\n")

def compair_chr_pos(x, y, chr_order, chr_column_index=0, start_column_index=1, end_column_index=1):
    
    """
    比较x和y，返回1和0。x小于或等于y时返回1。
    x和y都是一个存储了染色体名和开始位置的列表。也可以存储终止位置。
    chr_column_index，start_column_index， end_column_index分别表示染色体名、
    开始位置和终止位置在列表中的索引位置。如果end_column_index小于0，则表示不考虑
    终止位置。
    如果染色体不同，则染色体排序靠前的较小。否则，start较小的较小，
    如果start相同，而且end_column_index大于等于0，则end较小的较小。
    
    chr_order是一个字典，键是染色体名称，值是该染色体对应的顺序。
    当chr相同时，则比较pos。x的pos不大于y，则返回1。
    """
    
    if x[chr_column_index] == y[chr_column_index]:        
        if x[start_column_index] > y[start_column_index]:
            return 0
        elif end_column_index >= 0 and x[start_column_index] == y[start_column_index] and x[end_column_index] > y[end_column_index]:
            return 0
        else:
            return 1
    else:
        if chr_order[x[chr_column_index]] <= chr_order[y[chr_column_index]]:
            return 1
        else:
            return 0
            
def iter_sort_multi_item(xs, key_func=None, indexs=[]):
    
    """
    有多个列表（或迭代器）。想将它们的元素排序后输出出来。这里多个列表已经都排过序了。就设计了该函数
    迭代这些列表，按顺序迭代输出列表中的元素。具体排序函数由key_func指定，默认是比较两个元素大小，
    小的先输出。
    
    xs就是一个由多个列表组成的列表。
    
    key_func的格式形如：返回为1时，x会先输出，返回为0时，y会先输出。
    def _compare(x, y):
        if x <= y:
            return 1
        else:
            return 0
    
    indexs ：列表。默认为空。为空时每次迭代返回的仅是原来的列表中存储的元素。如果想每次返回元素时，
    也返回它是从哪个列表产生的，以及是这个列表的第几个元素。那么可以就定义每个列表的名称，也就是indexs。
    因此indexs的个数应该和xs中存储的列表的个数是一样的。这时每次迭代返回的是[d, index, i]。d是元素，
    index是indexs中存储的值，就是d来源的那个列表对应的index。i是d在它来源的那个列表的位置，0-based。
    """
    
    def iter_sort_two_item(x, y, key_func=None):

        """
        核心算法部分。
        x和y代表两个迭代器，算法中遍历x使用的是next方法，对list和tuple不兼容，因此先将list
        和tuple转化为迭代器。
        key_func用于比较两个元素。
        
        """

        def _compare(x, y):
            if x <= y:
                return 1
            else:
                return 0


        if key_func is None: key_func = _compare
        
        if isinstance(x, list) or isinstance(x, tuple): x = iter(x)
        if isinstance(y, list) or isinstance(y, tuple): y = iter(x)
        
        #算法中now_index控制当前迭代的是哪个迭代器。now_index=1-now_index
        #用于切换当前迭代器。
        a = [x, y]
        now_index = 0
        try:
            n1 = next(a[now_index])
            now_index = 1-now_index
            while 1:
                try:
                    n2 = next(a[now_index])
                    if key_func(n1, n2):
                        yield(n1)
                        n1 = n2
                        now_index = 1 - now_index
                    else:
                        yield(n2)
                except StopIteration:
                    yield(n1)
                    for n1 in a[1-now_index]:
                        yield(n1)
                    break
        except StopIteration:
            yield(n1)
            for n1 in a[1-now_index]:
                yield(n1)
                
    if key_func is None: key_func = _compare
    
    def conver_key_func2_with_index(key_func):
        def _with_index_func(x, y):
            return key_func(x[0], y[0])
        return _with_index_func
    
    def iter_with_index(x, index):
        for i, d in enumerate(x):
            yield([d, index, i])
        
    if indexs:
        key_func = conver_key_func2_with_index(key_func)
        xs = [iter_with_index(x, index) for x, index in zip(xs, indexs)]
    
    if len(xs) == 1:
        return xs[0]
    elif len(xs) >= 2:
        x = xs.pop()
        return iter_sort_two_item(x, iter_sort_multi_item(xs, key_func), key_func)
    
def read_chr_order(filein, chr_column_index=0):
    
    """
    读取第一列（可用chr_column_index设置，默认为第一列，即0）是
    排过序的染色体名的文件filein，返回一个字典。
    字典的键是染色体名，字典的值是染色体的出现的顺序，0-based。
    染色体列的染色体可以重复，但必须是排过序的。
    
    可以用samtools faidx genome.fa产生的genome.fa.fai文件。
    """
    
    chr_list = []
    last_chr = ""
    with open(filein) as f:
        for l in f:
            chr_ = l.rstrip("\n").split("\t")[0]
            if chr_ != last_chr:
                chr_list.append(chr_)
                last_chr = chr_
    chr_order = {}
    for i, chr_ in enumerate(chr_list):
        chr_order[chr_] = i
    return chr_order

def read_rna_seq2dict(filein):
    seqs = FastaSeqs(filein)
    seqs.load_seq()
    seqs.rna2dna()
    return seqs.fastas
   
def write_seq_length2file(filefasta, fileout):
    fastas = FastaSeqs(filefasta)
    fastas.write_seq_length2file(fileout)
          
def extract_seq_by_pos2file(pos_file, fasta_file, fileout, line_word_lim = 100, split=0):
    FastaSeqs().extract_seq_by_pos2file(pos_file, fasta_file, fileout, line_word_lim, split)
 
def extract_merge_seq_by_pos2file(pos_file, fasta_file, fileout, line_word_lim = 100, split=0):
    FastaSeqs().extract_merge_seq_by_pos2file(pos_file, fasta_file, fileout, line_word_lim, split)
  
def stat_genome_gc_by_bin():
    #filein, fileout, bin_size = 100
    filein, fileout, bin_size = sys.argv[1:]
    with open(fileout, "w") as o:
        for seq_id, seq in FastaSeqs().iter_read_fasta_from_file(filein):
            for i, s in enumerate(FastaSeq(seq_id, seq).bin(bin_size)):
                gc_ratio = FastaSeq("", s).gc_stat()
                o.write("%s\t%s\t%s\n" % (seq_id, i, gc_ratio))

def test_iter_fasta():
    filein = sys.argv[1]
    for seq_id, seq in FastaSeqs().iter_read_fasta_from_file(filein):
        my_seq = FastaSeq(seq_id, seq)
        gc_ratio = my_seq.gc_stat()
        print(seq_id, "\t", gc_ratio)
        for i, s in enumerate(my_seq.bin(100)):
            print(i, s)

class SeqBlocks():
    
    def __init__(self, file_length = "", bin = 100, file_seq = ""):
        '''
        NOW: not support file_seq, you must give file_length.
        The class read file_length file and init self.blocks data by bin.
        blocks is a dict with chr name as key. each element is a list which index
        is the block num.
        '''
        if file_length:
            self.load_length(file_length)
        if file_seq:
            self.load_seq(file_seq)
        self.bin = bin
        self.block_lengths, self.blocks = self._init_seqs_block()
        
    def load_seq(self, file_seq = ""):
        self._file_seq = file_seq
        if file_seq:
            pass
    
    def load_length(self, file_length=""):
        self._file_length = file_length
        if file_length:
            self.seq_ids, self.seq_lengths = self._read_seq_length(file_length)
            
    def _read_seq_length(self, filein):
        seq_lengths = {}
        seq_ids = []
        for l in open(filein):
            d = l.rstrip().split("\t")
            if len(d) < 2:
                continue
            seq_id, seq_length = d[:2]
            seq_length = int(seq_length)
            seq_ids.append(seq_id)
            seq_lengths[seq_id] = seq_length
        return seq_ids, seq_lengths
           
    def _init_seqs_block(self):
        
        def _init_seq_block(length, bin=100):
            total_block_num = (length - 1)//bin
            return total_block_num, [0] * total_block_num
            
        data = {}
        block_length = {}
        bin = self.bin
        seq_lengths = self.seq_lengths
        for seq_id, seq_length in seq_lengths.items():
            total_block_num, block_data = _init_seq_block(seq_length, bin)
            block_length[seq_id] = total_block_num
            data[seq_id] = block_data
        return block_length, data
            
    def extend(self, data):
        blocks = self.blocks
        bin = self.bin
        for seq_id, pos, count in data:
            bin_num = (pos - 1)//bin - 1
            if seq_id in blocks:
                blocks[seq_id][bin_num] += count
    
    def _bin2pos(self, bin_num, bin_length):
        start = bin_length*bin_num + 1
        end = start + bin_length - 1
        return [start, end]
    
    def write(self, fileout, header=0):
        bin = self.bin
        seq_ids = self.seq_ids
        blocks = self.blocks
        block_lengths = self.block_lengths
        bin2pos_func = self._bin2pos
        with open(fileout, 'w') as o:
            if header:
                o.write("seq_id\tbin_num\tstart\tend\tcount\n")
            for seq_id in seq_ids:
                block_data = blocks[seq_id]
                block_length = block_lengths[seq_id]
                for i in range(block_length):
                    count = block_data[i]
                    start, end = bin2pos_func(i, bin)
                    o.write( '\t'.join([str(s) for s in [seq_id, i, start, end, count]]) + "\n" )
                    

class ExtractPos():
    
    '''
    Exampele:
    ---------
    extract_pos_obj = ExtractPos(filedb, fileindex, header=False, 
                        chr_column_index, start_column_index, end_column_index,
                        block_length, sep))
    extract_pos_obj.extract_region(pos)
    
    属性：
    default_index_suffix    默认为".buindex"
    filedb
    fileindex
    index_configs
    index_data
    
    
    __init__()方法：
    --------------
    描述： 创建实例。如果fileindex为空，则fileindex的值是filedb + self.default_index_suffix，
    默认default_index_suffix为.buindex。如果fileindex文件不存在，则会调用index_file方法生成
    fileindex文件。
    
    参数：
    ----
    filedb : 参数sep隔开的文本文件，需按照染色体和起始位置排序从小到达排序，每行代表一个feature。
            文本开头以#开头的行会当做注释，不进行处理。但中间内容不能以#开头，否则需修改_extract_region
            方法中读取filedb相应的代码。
            示例文件格式如下:
             chr1   101 101 ...
    
    fileindex : 索引文件。
    
    header : filedb文件第一行是否为列名，如果是，则跳过第一行。默认为False。
    chr_column_index : filedb中染色体名所在列的索引，0-based。默认0。
    
    start_column_index : filedb中起始位置所在列的索引，0-based。默认1。
    
    end_column_index : filedb中终止位置所在列的索引，0-based。默认1。即默认与start_column_index相同。
    
    block_length : 建立索引时多大长度建立一个block。
    
    sep : filedb隔开每列的符号，默认为"\t"
    
    
    
    extract_region()方法：
    ————————————————————
    利用fileindex，返回filedb中与指定区域的有重叠（包含，被包含或一端重叠）的features。
    
    参数:
    ____
    pos : [chr, start, end]指定的一个区域。
    
    结果：
    ____
    迭代器。每一个元素代表一个feature，是filedb中该feature对应行利用sep（默认为"\t")分割后产生的列表，
    注意起始位置和输入位置仍是字符串格式。按照feature在filedb中出现的顺序输出。
    
    
    index原理：
    index文件以#开头的前几行为meta信息，提供了chr_column_index，end_column_index，
    start_column_index，block_length，sep。
    首先将染色体按特定的长度（默认为10000）分成紧紧相连无重叠的block。index文件存储的是与第一个
    与某个block重叠的feature所出现的位置（即其相对于文件开头的字符长度）。这样给定一个区域
    [chr, start, end]，就知道这个start所在的block，然后从index文件中记录的这个block对应的位置
    出开始读取feature，判断其是否与该区域重叠，直到某feature的start大于给定区域的end为止。
    '''
    
    default_index_suffix = ".buindex"
    
    def __init__(self, filedb="", fileindex="", header=False, chr_column_index=0, start_column_index=1, end_column_index=1, block_length=10000, sep="\t"):
        if not fileindex:
            fileindex = filedb + self.default_index_suffix
        if not os.path.exists(fileindex):
            self.index_file(filedb, fileindex, header, chr_column_index, start_column_index, end_column_index, block_length, sep)
        self.filedb = filedb
        self.fileindex = fileindex
        self.index_configs = None
        self.index_data = None
        
    def index_file(self, filedb, fileindex, header=False, chr_column_index=0, start_column_index=1, end_column_index=1, block_length=10000, sep="\t"):
    
        with open(filedb) as f, open(fileindex, 'w') as o:
        
            o.write("#chr_column_index\t{}\n".format(chr_column_index))
            o.write("#end_column_index\t{}\n".format(end_column_index))
            o.write("#start_column_index\t{}\n".format(start_column_index))
            o.write("#block_length\t{}\n".format(block_length))
            o.write("#sep\t{}\n".format(sep))
            
            if header: next(f)
            
            this_chr = ""
            this_block_end = block_length
            total_text_length = 0
            for l in f:
                if l.startswith("#"):
                    total_text_length += len(l)
                    continue
                d = l.rstrip("\n").split(sep)
                chr_, end  = d[chr_column_index], int(d[end_column_index])
                if chr_ != this_chr or end >= next_block_length:
                    this_chr = chr_
                    this_block_start = (end - 1) // block_length * block_length + 1
                    next_block_length = this_block_start + block_length
                    o.write("\t".join([chr_, str(this_block_start), str(total_text_length)]) + "\n")
                total_text_length += len(l)
                
    def read_index(self, fileindex):
    
        def _read_header(fileindex):
            header_total_length = 0
            header_configs = {}
            for l in open(fileindex):
                if not l.startswith("#"): break
                key, value = l.rstrip('\n').split("\t")
                key = key[1:]
                header_configs[key] = value
                header_total_length += len(l)
            return [header_configs, header_total_length]
    
        index_configs, header_total_length = _read_header(fileindex)
    
        index_data = {}
    
        with open(fileindex) as f:
            f.seek(header_total_length, 0)
            for l in f:
                chr_, start, total_length = l.rstrip("\n").split("\t")
                start, total_length = int(start), int(total_length)
                try:
                    index_data[chr_][0].append(start)
                    index_data[chr_][1].append(total_length)
                except:
                    index_data[chr_] = [[start], [total_length]]
            
        return [index_configs, index_data]
        
    def _extract_region(self, filedb, index_configs, index_data, pos):
    
        chr_column_index = int(index_configs["chr_column_index"])
        start_column_index = int(index_configs["start_column_index"])
        end_column_index = int(index_configs["end_column_index"])
        block_length = int(index_configs["block_length"])        
        sep = index_configs["sep"]
        if sep == "\\t": sep = "\t"
        
        chr_, start, end = pos
    
        try:
            chr_index_data_starts, chr_index_data_length = index_data[chr_]
        except:
            return []
        
        
        start_index = bisect_right(chr_index_data_starts, start) - 1
        if start_index == -1: start_index = 0
        left_index_start = chr_index_data_starts[start_index]
        left_index_total_length = chr_index_data_length[start_index]
        if end < left_index_start: return []
        if start >=  left_index_start + block_length:
            try:
                left_index_start = chr_index_data_starts[start_index+1]
                left_index_total_length = chr_index_data_length[start_index+1]
            except:
                return []
        
        with open(filedb) as f:
            f.seek(left_index_total_length, 0)
            for l in f:
                d = l.rstrip("\n").split(sep)
                line_chr, line_start, line_end = d[chr_column_index], int(d[start_column_index]), int(d[end_column_index])
                if line_chr != chr_: break
                if line_start > end:
                    break
                elif line_end >= start:
                    yield(d)
                    
    def extract_region(self, pos):
        if self.index_configs is None:
            self.index_configs, self.index_data = self.read_index(self.fileindex)
        
        return self._extract_region(self.filedb, self.index_configs, self.index_data, pos)


class ReadClipTool():

    """
    注意方法get_clip_seq中定义了name的格式
    使用方法：
    read_clip_obj = ReadClipTool(PAD_LENGTH)
    read_clip_obj.load_origin_seqs(filein_seq)
    
    for read in bam_obj.fetch():
        read_strand, seq_length, read_name, left_name, left_seq, right_name, right_seq = read_clip_obj.get_clip_seq(read)
    """
    def __init__(self, pad_length=20, sep=","):
        self.pad_length = pad_length
        self.sep = sep
    
    def load_origin_seqs(self, filein_seq):
        import buseq
        origin_seqs_obj = buseq.FastaSeqs(filein_seq)
        origin_seqs_obj.load_seq()
        self.origin_seqs_obj = origin_seqs_obj
        self.get_seq = origin_seqs_obj.extract_seq_by_one_pos
        self.rev_com = buseq.FastaSeqTool().revcom
    
    def get_clip_seq(self, read):
        read_strand = "-" if read.is_reverse else "+"
        left_clip_type, left_clip_length = read.cigartuples[0]
        right_clip_type, right_clip_length = read.cigartuples[-1]
        if left_clip_type == 0:
            left_clip_length = 0
        if right_clip_type == 0:
            right_clip_length = 0
        left_clip_length += self.pad_length
        right_clip_length += self.pad_length
        
        #如果是Hard clip，需查询原来的序列。
        #如果比对方向是反向的，需把原来的序列反向互补        
        seq = read.query_sequence
        if left_clip_type == 5 or right_clip_type == 5:
            seq = self.get_seq(read.query_name, [])
            if read_strand == "-": seq = self.rev_com(seq)
        seq_length = len(seq)

        if left_clip_type in [0, 4, 5]:
            left_clip_seq = seq[:left_clip_length]
            left_clip_seq = self.rev_com(left_clip_seq)
        else:
            1/0
        
        if right_clip_type in [0, 4, 5]:
            right_clip_seq = seq[(-right_clip_length):]
        else:
            1/0
        
        #if (not left_clip_seq) or (not right_clip_seq): continue
        
        read_name = self.sep.join([read.query_name, 
                              read.reference_name,
                              str(read.reference_start + 1), 
                              str(read.reference_end),
                              read_strand,
                              str(seq_length),
                              str(left_clip_length),
                              str(right_clip_length),
                              str(self.pad_length)])
        left_name, right_name = self.generate_left_right_name(read_name)
        return [read_strand, seq_length, read_name, left_name, left_clip_seq, right_name, right_clip_seq]
    
    def get_read_name(self, left_or_right_name):
        record_id_split = left_or_right_name.split(self.sep)
        read_name = left_or_right_name[:-(len(record_id_split[-1]) + len(self.sep))]
        return [read_name, record_id_split]
    
    def get_read_info(self, record_id_split):
        #[read_strand, str_read_length, \
        #str_left_length, str_right_length, \
        #str_clip_inner_length, first_end_type]
        return record_id_split[4:10]

    def generate_left_right_name(self, read_name):
        return [read_name + self.sep + "5", read_name + self.sep + "3"]

def iter_bam_clip_seq(filein_bam, filein_seq, pad_length=20, sep=","):
    '''
    输入minimap2比对的bam文件和原始read的fasta序列文件。
    输出clip两端的序列，各加上20bp（pad_length)。
    注意左端和右端是基因组位置的5’端和3‘端。

    1）read核心名字：
    默认SEP为","，用于生成read名。

    因为一个read可能比对到多个位置，因此需要将read名加上比对位置，用于表示这个比对。
    基本格式为read_name,chr_name,start,end,strand

    这个名字对应着extract_read_exon_pos的名字，只不过extract_read_exon_pos
    生成名字还多一个block数量。而该脚本生成名字还多了
    seq_length, left_clip_length, right_clip_length, pad_seq_length。
    left_clip_length也是基因组5’端的clip长度，而不是read的5‘端。

    2）clip序列名字
    左边clip序列是read的核心名字加上",5",右边的是加上",3"。
    '''

    #每个元素是[read, [read_strand, seq_length, read_name, left_name, left_clip_seq, right_name, right_clip_seq]]
    read_clip_obj = ReadClipTool(pad_length, sep)
    read_clip_obj.load_origin_seqs(filein_seq)
    with pysam.AlignmentFile(filein_bam, "rb") as bam_obj:
        for read in bam_obj.fetch():
            yield([read, read_clip_obj.get_clip_seq(read)])
            
def extract_chrpos(filedb, pos, fileindex="", header=False, chr_column_index=0, start_column_index=1, end_column_index=1, block_length=10000, sep="\t"):
    extract_pos_obj = ExtractPos(filedb, fileindex, header, chr_column_index, start_column_index, end_column_index, block_length, sep)
    return extract_pos_obj.extract_region(pos)

def polyA_finder(seq="", base="A",
                 match = 1,
                 mis = -1.5):
    """
    给定序列，寻找该序列的polyA
    （可指定base，寻找其他poly，如polyT）。
    
    结果示例：[[start, end, score]]
    返回结果是找到的polyA位置列表。没找到则为空列表。
    列表按polyA发现顺序排序，每个元素也是一个列表[start, end, score]
    注意start和end是0-based index，可以通过seq[start:end]获得polyA的序列。
    因此end并不是polyA的末端，而是末端的索引+1。
    
    score是polyA打分，给定一个序列，可以用如下打分系统评价其polyA打分：
    A的个数 * match - 非A碱基的个数 * mis。
    
    必须有至少5个碱基连续的A才行。
    """
      
    def align_find_max(query, direction="+", base="A", match=1, mis=-1.5, tanlan_mode=0):
        """
        给一个序列query，选择从一段锚定，拓展获得最大的多聚A比对打分。从左端向右拓展，则
        为direction="+",从右端向左拓展，direction="-"。
        返回结果是[max_score_index, max_score, total_score]
        max_score_index是获得最大打分的query的索引（0-based)，此时的打分是max_score。
        total_score是query全长的打分。
        """
        if not query: return [0, 0, 0]
        #累加计算打分，求最大值，获得最大值所在的index
        indexs = range(len(query)) if direction == "+" else range(len(query)-1, -1, -1)
        scores = []
        last_score = 0
        for i in indexs:
            s = query[i]
            if s == base:
                last_score += match
            else:
                last_score += mis
            scores.append(last_score)
        max_score = max(scores)
        total_score = scores[-1]
        if tanlan_mode:
            max_score_index = scores[::-1].index(max_score)
        else:
            max_score_index = scores.index(max_score)
        if direction == "-":
            max_score_index = len(query) - max_score_index - 1
        return [max_score_index, max_score, total_score]
        
    min_poly_count = 5
    window_length = 5
    min_poly_seq = "A" * min_poly_count
    
    #1. 如果长度小于min_poly_count，返回[]
    if len(seq) < min_poly_count: 
        return []
    
    #2. 将序列分成长度为5的窗口，计算每个窗口中的A含量
    #存放到window_counts列表中，每个元素是[start, end, count]
    #start, end是该区间的索引，注意区间不包含end
    max_window_num =  ceil(len(seq)/window_length)
    window_counts = []
    for window_num in range(max_window_num):
        start = window_num * window_length
        end = start + window_length if window_num < max_window_num - 1 else len(seq)
        window_counts.append([start, end, seq[start:end].count(base)])
    
    #3. 获得含有min_poly_count个A的种子区域
    #seeds是按顺序排列的种子区域列表，每个元素是[start, end, score]
    #start, end是种子区域位置索引，注意区间不包含end
    #score是该种子区域的比对打分
    seeds = []
    for i, (start, end, count) in enumerate(window_counts):
        if count == 5:
            if seeds and seeds[-1][1] == start:
                seeds[-1][1] = end
                seeds[-1][2] = seeds[-1][2] + 5*match
            else:
                seeds.append([start, end, 5*match])
        elif count > 0 and i > 0:
            last_start, last_end, last_count = window_counts[i - 1]
            if last_count + count >= 5 and last_count < 5:
                poly_index = seq[last_start:end].find(min_poly_seq)
                if poly_index > -1:
                    poly_start = start + poly_index
                    poly_end = start + len(min_poly_seq)
                    for ls_i in range(poly_end, end):
                        if seq[ls_i] == base:
                            poly_end = ls_i + 1
                        else:
                            break
                    seeds.append([poly_start, poly_end, (poly_end-poly_start)*match])
    
    #4. 种子区域拓展和合并
    if not seeds:
        return []
    else:
        #第一个种子向前拓展
        start, end, score = seeds[0]
        if start > 0:
            max_score_index, max_score, total_score = align_find_max(seq[:start], "-", base, match, mis)
            if max_score > 0:
                seeds[0][0] = max_score_index
                seeds[0][2] = score + max_score
        #最后一个种子向后拓展
        start, end, score = seeds[-1]
        if end < len(seq) - 1:
            max_score_index, max_score, total_score = align_find_max(seq[end:], "+", base, match, mis)
            if max_score > 0:
                seeds[-1][1] = max_score_index + end + 1
                seeds[-1][2] = score + max_score
        #中间的种子拓展
        if len(seeds) > 1:
            start, end, score = seeds[0]
            merge_seeds = [[start, end, score]]
            for next_start, next_end, next_score in seeds[1:]:
                start, end, score = merge_seeds[-1]    
                max_score_index, max_score, total_score = align_find_max(seq[end:next_start], "+", base, match, mis)
                if total_score > -min(next_score, score):
                    merge_seeds[-1][1] = next_end
                    merge_seeds[-1][2] = score + next_score + total_score
                else:
                    if max_score > 0:
                        merge_seeds[-1][1] = max_score_index + end + 1
                        merge_seeds[-1][2] = score + max_score
                    max_score_index, max_score, total_score = align_find_max(seq[end:next_start], "-", base, match, mis)
                    merge_seeds.append([next_start, next_end, next_score])
                    if max_score > 0:
                        merge_seeds[-1][0] = max_score_index + end
                        merge_seeds[-1][2] = next_score + max_score
            seeds = merge_seeds
        return(seeds)
        
def remove_seq_des(filein, fileout, sep=""):
    with open(fileout, 'w') as o:
        for l in open(filein):
            if l.startswith(">"):
                if sep:
                    name = l.rstrip().split(sep)[0]
                else:
                    name = l.rstrip().split()[0]
                o.write(name + "\n")
            else:
                o.write(l)

def extend_depth_file(file_length, file_depth, fileout, bin=100):
    
    def _iter_depth_file(file):
        with open(file) as f:
            for l in f:
                d = l.rstrip().split("\t")
                if len(d) != 3:
                    continue
                chr_, length, count = d
                length, count = int(length), int(count)
                yield([chr_, length, count])
                
    seq_blocks = SeqBlocks(file_length, bin=bin)
    depth_data = _iter_depth_file(file_depth)
    seq_blocks.extend(depth_data)
    seq_blocks.write(fileout)

def filter_fastq_by_length(filein, fileout, min_length=0, max_length=0):
    Fastq().read_filter_write_by_length(filein, fileout, min_length, max_length)

def stat_fastq_length(filein, fileout):
    Fastq().write_stat_length(filein, fileout)      
    
def main():
    stat_genome_gc_by_bin()

if __name__ == '__main__':
    main()
