# -*- coding: utf-8 -*-

import sys
import collections
import gzip
import os
from collections import Counter
import pandas as pd


def get_header(filein):
    header = ""
    with open(filein) as f:
        header = next(f)
    return header

def get_header_index(filein, select_column_names, sep="\t"):
    """
    输入列名列表，返回它们在文件中的列的index。
    注意如果有多列同名，返回最小的那个index。
    """
    select_indexs = []
    column_names = get_header(filein).rstrip("\n").split(sep)
    select_indexs = [column_names.index(column_name) for column_name in select_column_names]
    return select_indexs

def read_by_column_names(filein, select_column_names, sep="\t"):
    select_indexs = get_header_index(filein, select_column_names, sep)
    with open(filein) as f:
        next(f)
        for l in f:
            d = l.rstrip("\n").split(sep)
            yield([d[i] for i in select_indexs]) 

def read_ids(filein, id_index, header=False, split_sep="\t"):
    ids = set()
    with open(filein) as f:
        if header: next(f)
        for l in f:
            id_ = l.rstrip("\n").split(split_sep)[id_index]
            ids.add(id_)
    return ids


def _iter_id(d, id_index=0):
    '''
    输入d为[["id1", 1, 2], ["id2", 2, 3], ["id2", 4, 5], ["id3", 4, 5]]
    注意两个id2必须是连续的，否则不会合并为一个。有点类似linux uniq的特性。
    将id_index设为0，则输出一个迭代器，每个元素依次为：
    ["id1", [["id1", 1, 2]]],
    ["id2", [["id2", 2, 3], ["id2", 4, 5]]],
    ["id3", [["id3", 4, 5]]]
    '''
    record_id = ""
    record_d = []
    for l in d:
        if record_id != l[id_index]:
            if record_d:
                yield([record_id, record_d])
            record_d = []
            record_id = l[id_index]
        record_d.append(l)
    yield([record_id, record_d])

def iter_id(d, id_index=0, sort_flag=False):
    '''
    封装_iter_id，实现对id不连续的支持。但是这样如果d是迭代器，需要把d的数据全部读入内存。
    '''
    if sort_flag: d = sorted(d, key=lambda x: x[id_index])
    return _iter_id(d, id_index)
        

def trans_command_line_sep(sep):
    if sep == "\\t":
        sep = "\t"
    return(sep)

def extract_by_index(filedb, fileid, fileindex, fileout, filedbheader_flag=0,
                            fileidheader_flag=0, fileoutheader_flag = 0, 
                            db_id_column_num=0, id_id_column_num=0,
                            filedb_sep="\t", fileid_sep="\t", fileout_seq="\t"):

        extract_db = ExtractByIndex(filedb, fileindex, filedbheader_flag,
                                    db_id_column_num, filedb_sep)
        
        extract_db.extract_id_to_file(fileid, fileout, fileoutheader_flag, fileout_seq,
                                        fileidheader_flag, id_id_column_num, fileid_sep)

        extract_db.close()

class ExtractByIndex():
    
    '''
    use for:
    1) index file
    2) extract row by id quickly
    3) extract rows by ids quickly
    4) extract rows by ids file quickly
    
    To use the class, pleas set the information of file_db and file_index:
    for __init__ function:
    file_db_name or file_db_obj: text file or text file stream.
    file_db_header: Whether the filedb has header [default 0]
    file_db_id_column_num: The id column num [default 0]
    file_db_sep: The separator of file_db [default: "\t"]
    file_index_name: the index file for extract id from file_db quickly.
        If the file doesn't exsit, then create it. So you can init the class to
        create a index file.
    
    other important function:
    close(): you must use run self.close to close db file stream.!!!!!
    extract_by_id(id): extract id. return the row which has id as key in filedb
    extract_by_ids(ids): ids is a list of id. return a generator.
    extract_by_idfile(fileid, header, id_column_num, fileid_sep):
            read ids from fileid. and run extract_by_ids(ids).
    extract_by_idfile(fileid, fileout, fileoutheader_flag,id_header_flag,
            id_id_column_num, fileid_sep):
            run extract_by_idfile. and write the result to fileout.
            fileid: id file. one row one id.
            id_header_flag: whether id file has header, if yes, skip the first
                            line of file id.
            id_id_column_num: the id column num
            fileid_sep: The separator of id file.
            fileout: the output file
            fileoutheader_flag[0]: 0 or 1. If 1: print the header of filedb to fileout.
    
    Don't use gzip file as data for now. I havn't write the gzip reader object.
    '''
    
    def __init__(self, file_db_name=None, file_index_name=None, 
                    file_db_header=0, file_db_id_column_num = 0,
                    file_db_sep = "\t", file_db_obj=None):
        
        if file_db_obj is None:
            if not file_db_name:
                raise IOError("No file_db_name or file_db_obj was supported!")
            file_db_obj = open(file_db_name)
        self.file_db_obj = file_db_obj
        self.file_db_name = self.file_db_obj.name
        self.file_db_header = file_db_header
        self.file_db_sep = file_db_sep
        self.file_db_id_column_num = file_db_id_column_num
        self.file_index_name = file_index_name

        if not os.path.exists(file_index_name):
            self._index_file()

        self._readindex() #to: self.index_data
        self._readhead() #to: self.header()
                                  
    def _index_file(self):
        
        file_index_name = self.file_index_name
        file_db_name = self.file_db_name
        file_db_header = self.file_db_header
        file_db_id_column_num = self.file_db_id_column_num
        file_db_sep = self.file_db_sep
        
        with open(file_db_name) as f, open(file_index_name, 'w') as o:
            for l in f:
                id = l.rstrip("\n").split(file_db_sep)[file_db_id_column_num]
                o.write("{id}\t{length}\n".format(id=id, length=len(l)))
        
    def _readindex(self):
        
        file_index_name = self.file_index_name
        index_sep = "\t"
        
        index_data = {}
        total_length = 0
        for l in open(file_index_name):
            id, length = l.rstrip("\n").split(index_sep)
            length = int(length)
            index_data[id] = [total_length, length]
            total_length += length
        self.index_data = index_data
            
    def _readhead(self, ):
        header = ""
        if self.file_db_header:
            header = next(self.file_db_obj)
        self.header = header
    
    def _readids(self, file, header = False, id_column_num = 0, sep = "\t"):
        ids = []
        with open(file) as f:
            if header: next(f)
            for l in f:
                l = l.rstrip("\n")
                if not l:
                    continue
                this_id = l.split(sep)[id_column_num]
                ids.append(this_id)
        return(ids)
    
    def extract_by_id(self, id=""):
        
        if id not in self.index_data:
            return None
        start, length = self.index_data[id]

        try:
            self.file_db_obj.seek(start,os.SEEK_SET)
            value = self.file_db_obj.read(length)
            return(value)
        except:
            self.close()
            raise IOError("read id: {id} failure. in byte {start} {length}!\n"
                              .format(id=id, start=start, length=length))
    
    def extract_by_ids(self, ids=[]):
        for id in ids:
            start, length = self.index_data[id]
            self.file_db_obj.seek(start,os.SEEK_SET)
            value = self.file_db_obj.read(length)
            yield(value)
    
    def get_header(self):
        return self.header
    
    def write_stream(self, results, fileoutheader_flag = 0, fileoutsep = "\t", filein_sep = "\t"):
        if fileoutheader_flag:
            header = self.get_header()
            if filein_sep != fileoutsep:
                header = header.replace(filein_sep, fileoutsep)
            yield(header)
        for value in results:
            if filein_sep != fileoutsep:
                value = value.replace(filein_sep, fileoutsep)
            yield(value)
    
    def write(self, results, fileout, fileoutheader_flag = 0, fileout_sep = "\t", filein_sep = "\t"):
        with open(fileout, 'w') as o:
            for line in self.write_stream(results, fileoutheader_flag, fileout_sep, filein_sep):
                o.write(line)
                                    
    def extract_by_idfile(self, fileid, header = False, id_column_num = 0, fileid_sep = "\t"):
        ids = self._readids(fileid, header, id_column_num, fileid_sep)
        return self.extract_by_ids(ids)
                                                
    def extract_id_to_file(self, fileid, fileout, fileoutheader_flag = 0, fileout_sep = "\t", 
                            id_header_flag = False, id_id_column_num = 0, fileid_sep = "\t"):
        file_db_sep = self.file_db_sep
        iter_query_results = self.extract_by_idfile(fileid, id_header_flag, id_id_column_num, fileid_sep )
        self.write(iter_query_results, fileout, fileoutheader_flag, fileout_sep, file_db_sep)

    def close(self):
        try:
            self.file_db_obj.close()
        except:
            pass

class FileIndex_Old():
    
    def __init__(self):
        pass
    
    def setdb(self, filedb, header_flag=0, id_column_num=0, sep="\t"):
        self.filedb = filedb
        self.dbheaderflag = header_flag
        self.db_id_column_num = id_column_num
        self.db_sep = sep
        
    def setindex(self, fileindex, header_flag=0):
        #index sep is "\t"
        self.fileindex = fileindex
        self.indexheaderflag = header_flag
        
    def open_db(self):
        try:
            self.close_db()
        except:
            pass
        self.dbhandle = open(self.filedb)
        
    def close_db(self):
        try:
            self.dbhandle.close()
        except:
            pass

    def readindex(self, return_data_flag=False):
        
        filein = self.fileindex
        header_flag = self.indexheaderflag
        index_data = {}
        total_length = 0
        with open(filein) as f:
            if header_flag:
                id_, length = next(f).rstrip("\n").split()
                length = int(length)
                total_length += length
            for l in f:
                id_, length = l.rstrip("\n").split()
                length = int(length)
                index_data[id_] = [total_length, length]
                total_length += length
        self.index_data = index_data
        if return_data_flag:
            return(index_data)
    
    def indexfile(self):
        
        sep = self.db_sep
        filein = self.filedb
        fileout = self.fileindex
        id_column_num = self.db_id_column_num
        with open(filein) as f, open(fileout, 'w') as o:
            for l in f:
                d = l.rstrip("\n").split(sep)[id_column_num]
                o.write("{id}\t{length}\n".format(id=d, length=len(l)))

    def get_by_id(self, id_, close_handle=False):
        
        start, length = self.index_data[id_]
        try:
            self.dbhandle.seek(start,os.SEEK_SET)
            value = self.dbhandle.read(length)
            return(value)
        except:
            self.close_db()
            raise IOError("read file {file} by id: {id} failure. in byte {start} {length}!\n"
                          .format(file=self.filedb, id=id_, start=start, length=length))
        if close_handle:
            self.close_db()
    
    def get_by_ids(self, ids, close_handle = False):
        
        for id_ in ids:
            yield(self.get_by_id(id_))
            
        if close_handle:
            self.close_db()

    def extract_fiter(self, filedb, fileid, fileindex, fileout, filedbheader_flag=0, 
                      fileidheader_flag=0, fileoutheader_flag = 0, 
                      db_id_column_num=0, id_id_column_num=0,
                      filedb_sep="\t", fileid_sep="\t"):
        
        def readids(file, header = False, id_column_num = 0):
            split_char = "\t"
            ids = set()
            with open(file) as f:
                if header: next(f)
                for l in f:
                    l = l.rstrip("\n")
                    if not l:
                        continue
                    this_id = l.split()[id_column_num]
                    ids.add(this_id)
            return(ids)
        
        def readhead(file):
            header = ""
            with open(file) as f:
                header = f.readline()
            return(header)

        self.setdb(filedb, filedbheader_flag, db_id_column_num, filedb_sep)
        self.setindex(fileindex, filedbheader_flag)
        if (not fileindex) or (not os.path.exists(fileindex)):
            self.indexfile()
        self.readindex()
        ids = readids(fileid, fileidheader_flag, id_id_column_num)
        self.open_db()
        with open(fileout, 'w') as o:
            if fileoutheader_flag and filedbheader_flag:
                o.write(readhead(filedb))
            for id_ in ids:
                o.write(self.get_by_id(id_))
        self.close_db()

def iter_cuts(iters, cut=10, part=0, length=0, max_value=0, max_func=0):
    
    #give a iterable object, cut it by cut.
    #return a iterable object, each element is [time, list]
    #cut是每部分多少个元素。part是所有元素分割成多少份。用part的话，如果iters不是列表，需要提供length，
    #也就是总元素个数。有时候每个元素的权重不一样，可以用max_value和max_func配套使用。
    #max_func来对每个元素计算出一个值，使分割后每部分各元素的值加起来恰好不大于max_value。
    
    cut = int(cut)

    if max_value != 0:
        if not max_func:
            max_func = lambda x: x
        time = 0
        ls_d = []
        value = 0
        for i, d in enumerate(iters):
            ls_d.append(d)
            value += max_func(d)
            if value >= max_value:
                yield([time, ls_d])
                ls_d = []
                time += 1
                value = 0
        if ls_d:
            yield([time, ls_d])
    else:
        data = iters
        if part != 0:
            if length == 0:
                data = list(iters)
                length = len(data)
            cut = (length - 1)//part + 1
        time = 0
        ls_d = []
        for i, d in enumerate(data):
            ls_d.append(d)
            if i % cut == cut - 1:
                yield([time, ls_d])
                ls_d = []
                time += 1
        if ls_d:
            yield([time, ls_d])

def iter_split_file_by_row(filein, cut=10):
    
    cut = int(cut)
    for time, data in iter_cuts(open(filein), cut):
        yield([time, data])

def cut_iter(iters, id_index=0):
    
    """
    给定一个每个元素是一个列表的迭代器。每个列表的第一个元素是其id。使用类似下面：
    for id_, id_data in cut_iter(iters, id_index):
        for d in id_data:  #d是原始的迭代器的一个元素
            print(d)
    """
    
    def iter_this_id():
        nonlocal last_d, last_id, id_index, not_iter_completed
        yield(last_d)
        while 1:
            try:
                d = next(iters)
            except:
                not_iter_completed = 0
                break
            if d[id_index] == last_id:
                yield(d)
            else:
                last_id = d[id_index]
                last_d = d
                break
    try:
        last_d = next(iters)
        last_id = last_d[id_index]
        not_iter_completed = 1
        while not_iter_completed:
            yield([last_id, iter_this_id()])
    except:
        pass

def iter_file(filedata, sep="\t"):
    
    if isinstance(filedata, str):
        filedata = open(filedata)
        
    for l in filedata:
        yield(l.rstrip().split(sep))
        
def cut_iter_file(filedata, id_index=0, sep="\t"):
    
    return cut_iter(iter_file(filedata, sep), id_index)
        
    
def file2list(filedata):

    sep ="\t"
    data = []
    for l in filedata:
        data.append(l.rstrip().split(sep))
    return(data)

def iter_wid2long(data):
    nrow = len(data)
    ncol = len(data[0])
    for i in range(ncol):
        yield([data[j][i] for j in range(nrow)])    

class DataDict():
    
    def __init__(self, filein = ""):
        if filein: 
            self.readdict(filein)
        
    def readdict(self, filein = "", key_column = 0, header_flag = False, delete_key = True):
        data = {}
        header = []
        with open(filein) as f:
            if header_flag: header = next(f).rstrip().split("\t")
            def readline(l):
                d = l.rstrip().split("\t")
                if len(d) - 1 < key_column: raise ValueError("col length are not same:\n" + l + "\n") 
                key = d[key_column]
                if delete_key:
                    d = d[:key_column] + d[(key_column+1):]
                return(key, d)
            key, d = readline( next(f) )
            collen = len(d)
            data[key] = d            
            for l in f:
                key, d = readline(l)
                if len(d) != collen: raise ValueError("col length are not same:\n" + l + "\n") 
                data[key] = d
        self.changedata(data, collen)
    
    def setdefautvalue(self):
        d = ["0"] * self.collen
        self.defautvalue = d
        return(d)
        
    def changedata(self, data, collen, header=[]):
        self.data = data
        self.collen = collen
        if len(header) != self.collen:
            self.header = []
        else:
            self.header = header
        self.setdefautvalue()
        
    def merge(self, data2, new_header="", all="all", changeself = True):
        #all: "all", "left", "right", "both"
        new_data = {}
        x = set(self.data.keys())
        y = set(data2.data.keys())
        xy_both = x & y
        x_only = x - y
        y_only = y - x
        for key in xy_both:
            new_data[key] = self.data[key] + data2.data[key]
        if all in ["all", "left"]:
            for key in x_only:
                new_data[key] = self.data[key] + data2.defautvalue
        if all in ["all", "right"]:
            for key in y_only:
                new_data[key] =  data2.defautvalue + data2.data[key]
        collen = self.collen + data2.collen
        if changeself:
            self.changedata(new_data, collen)
            self.header = new_header
            return(self)
        else:
            result = DataDict()
            result.changedata(new_data, collen)
            result.header = new_header
            return(result)
    
    def write(self, fileout):
        with open(fileout, 'w') as o:
            if self.header: o.write('\t'.join(self.header) + "\n")
            for key, d in self.data.items():
                o.write(key + "\t" + "\t".join(d) + "\n")
                
                       
                
class LabelLeading_onlyone():
    
    def add_leading_line_write(self, filein_handle, fileout, label_prex="#Chr_name:\t"):

        last_label = ""
        with open(fileout, 'w') as o:
            for d in filein_handle:
                this_label = d[0]
                if this_label != last_label or (not last_label):
                    last_label = this_label
                    o.write(label_prex + last_label + "\n")
                o.write('\t'.join(d[1:]) + "\n")
    
    def _read_leading_line_file_all_line(self, filein, label_prex="#Chr_name:\t"):
        len_label_prex = len(label_prex)
        
        this_label = ""
        for l in open(filein):
            if l.startswith(label_prex):
                this_label = l[len_label_prex:-1]
            else:
                yield([this_label, l])
       
    def read_leading_line_file(self, filein, label_prex="#Chr_name:\t"):
        
        for label, l in self._read_leading_line_file_all_line(filein, label_prex):
            d = l.rstrip("\n").split("\t")
            yield([this_label] + d)

    def read_leading_line_parse_by_leading(self, filein, label_prex="#Chr_name:\t"):
        len_label_prex = len(label_prex)
        this_label = ""
        this_label_data = []
        for l in open(filein):
            if l.startswith(label_prex):
                if this_label:
                    yield([this_label, this_label_data])
                this_label_data = []
                this_label = l[len_label_prex:-1]
            else:
                d = l.rstrip("\n").split("\t")
                this_label_data.append(d)
        if this_label:
            yield([this_label, this_label_data])
            
    def convert_leading_line_file(self, filein, fileout, label_prex="#Chr_name:\t"):
        with open(fileout, 'w') as o:
            for label, l in self._read_leading_line_file_all_line(filein, label_prex):
                o.write(label + "\t" + l)

class LabelLeading():
    
    def add_leading_line_write(self, filein_handle, fileout, leading_label_data=[[0, "#Chr_name:\t"]]):
        
        def _delete_list_elements_by_indexs(d, indexs):
            indexs_set = set(indexs)
            return_indexs = []
            for i in range(len(d)):
                if i not in indexs:
                    return_indexs.append(i)
            return(return_indexs)
            
        leading_indexs = [x for x, y in leading_label_data]
        leading_labels = [y for x, y in leading_label_data]
        last_labels = [""] * len(leading_label_data)
        return_indexs = []        
        with open(fileout, 'w') as o:
            for d in filein_handle:
                for i, (leading_index, leading_label, last_label) in enumerate(zip(leading_indexs, leading_labels, last_labels)):
                    this_label = d[leading_index]
                    if this_label != last_label or (not last_label):
                        o.write("%s%s\n" % (leading_label, this_label))
                        last_labels[i] = this_label
                if not return_indexs:
                    return_indexs = _delete_list_elements_by_indexs(d, leading_indexs)
                r = [d[i] for i in return_indexs]
                o.write('\t'.join(r) + "\n")
                
    def read_leading_line_file(self, filein, leading_label_data=[[0, "#Chr_name:\t"]]):
        
        def _steps_merge_data_labels(d_length, indexs):
            
            indexs_2pos = {}
            for pos, index in enumerate(indexs):
                indexs_2pos[index] = pos
                
            total_length = d_length + len(indexs)
            insert_nums = 0
            steps = []
            for i in range(total_length):
                if i in indexs_2pos:
                    steps.append([1, indexs_2pos[index]])
                    insert_nums += 1
                else:
                    steps.append([0, i - insert_nums])
            return(steps)
            
        def _merge_data_labels(d, labels, steps):
            data = [d, labels]
            r = []
            for i, index in steps:
                r.append(data[i][index])
            return(r)
            
        leading_indexs = [x for x, y in leading_label_data]
        leading_labels = [y for x, y in leading_label_data]
        #length_leading_labels = [len(s) for s in leading_labels]
        last_labels = [""] * len(leading_label_data)
        steps = []
        for l in open(filein):
            l = l.rstrip()
            for i, leading_label in enumerate(leading_labels):
            #for i, (leading_label, length_leading_label) in enumerate(zip(leading_labels, length_leading_labels)):
                if l.startswith(leading_label):
                    last_labels[i] = l[len(leading_labels):]
                    #last_labels[i] = l[length_leading_label:]
            d = l.split("\t")
            if not steps:
                steps = _steps_merge_data_labels(len(d), leading_indexs)
            r = _merge_data_labels(d, last_labels, steps)
            yield(r)

    
def main():
    stat_genome_gc_by_bin()

if __name__ == '__main__':
    main()
    
        
