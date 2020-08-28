import pysam
from buseq import iter_bam_feature
import sys

def main():
    file_bam, file_intron_pos, fileout = sys.argv[1:4]

    try:
        strand_flag = int(sys.argv[4])
    except:
        strand_flag = 0  #1 is RF, 2 is FR, 0 is no. #主要用RF

    try:
        min_overlap = int(sys.argv[5])
    except:
        min_overlap = 6
    
    try:
        write_read_info = int(sys.argv[6])
    except:
        write_read_info = 0
    
    fileout_read_info = fileout + ".readinfo.txt" if write_read_info else ""
    
    with open(fileout, 'w') as OUT:
        OUT.write("intron_id\tchr_name\tintron_start\tintron_end\tintron_strand\tintron_coverage_ratio\ta\tb\tab\tc\to\tt\tiratio\tsratio\toratio\to1ratio\to2ratio\to1_type\to1_count\to2_type\to2_count\tother_o\n")
        for [intron_id, chr_name, intron_start, intron_end, intron_strand], this_intron_data, intron_reads in iter_intron_read_type(file_bam, file_intron_pos, min_overlap, strand_flag, fileout_read_info):
            intron_output = "\t".join([intron_id, chr_name, str(intron_start), str(intron_end), intron_strand])
            intron_coverage_ratio = stat_intron_coverage(intron_reads, intron_start, intron_end)
            readname2flag = {}
            for flag, read_name, str_blocks in this_intron_data:
                if flag.startswith("p"): continue
                if read_name in readname2flag:
                    f = readname2flag[read_name]
                    #不等的情况
                    #a/b o    --> o   #a可能属于o
                    #a/b a/b    --> ab
                    #a/b ab   --> ab
                    #a/b c    --> c   #比对错误？
                    #o   ab   --> ab  #比对错误？
                    #o   c    --> c   #比对错误？
                    #o   o    --> o   #比对错误？选择junction更短的o
                    if f == flag:
                        pass
                    elif f.startswith("ab") or flag.startswith("ab"):
                        flag = "ab"
                    elif f.startswith("c") or flag.startswith("c"):
                        flag = "c"
                    elif f.startswith("o") or flag.startswith("o"):
                        #一部分，如a，b的。是因为被包含关系。
                        #其他的主要是错误比对。如o和o，o和c，Cyto4中共几千个read。
                        #当o与c，o与ab存在时，用c或ab。
                        #其中AT1G07930和AT1G07920就占了很大一部分比例。
                        #因为这两个基因序列很相近。
                        #一个read被拆分比对到两个基因上了。
                        #这里出现两个o时，返回junction短的那个o。
                        if (f[0] == "o" and flag[0] == "o"):                        
                            junction_length1 = cal_junction_length(f)
                            junction_length2 = cal_junction_length(flag)
                            if junction_length2 > junction_length1:
                                flag = f
                        else: #主要是a或者b
                            if f[0] == "o": 
                                flag = f                    
                    elif (f[0]=="a" and flag[0]=="b") or (f[0]=="b" and flag[0]=="a"):
                        flag = "ab"
                    readname2flag[read_name] = flag
                else:
                    readname2flag[read_name] = flag
            flag_stat = {"a": 0, "b": 0, "ab": 0, "c": 0, "o": 0, "i": 0}
            flag_o_stat = {}
            for flag in readname2flag.values():
                if flag[0] == "o":
                    flag_stat["o"] += 1
                    if flag not in flag_o_stat:
                        flag_o_stat[flag] = 1
                    else:
                        flag_o_stat[flag] += 1
                else:
                    flag_stat[flag] +=1
            #intron_id chr_name start end strand a b ab c o o1_type o1_count o2_type o2_count  other_o(type=count;)
            o_stat_list = sorted(flag_o_stat.items(), key=lambda x: x[1], reverse=True)
            if len(o_stat_list) >= 1:
                o1_type, o1_count = o_stat_list[0]
            else:
                o1_type, o1_count = "", 0
            if len(o_stat_list) >= 2:
                o2_type, o2_count = o_stat_list[1]
            else:
                o2_type, o2_count = "", 0
            if len(o_stat_list) > 2:
                other_o = ";".join([s + "=" + str(v) for s, v in o_stat_list[2:]])
            else:
                other_o = ""
            a = flag_stat["a"]
            b = flag_stat["b"]
            ab = flag_stat["ab"]
            c = flag_stat["c"]
            o = flag_stat["o"]
            t = a + b + ab + c + o
            if t == 0: continue
            ls_t = a + b + 2*(ab + c + o)
            iratio = (a + b + 2*ab)/ls_t
            sratio = 2*c/ls_t
            oratio = 2*o/ls_t
            o1ratio = 2*o1_count/ls_t
            o2ratio = 2*o2_count/ls_t
            r = "\t".join(str(i) for i in [a, b, ab, c, o, t, iratio, sratio, oratio, o1ratio, o2ratio])
            OUT.write("\t".join([intron_id, chr_name, str(intron_start), str(intron_end), intron_strand, 
                               str(intron_coverage_ratio),
                               str(r), 
                               o1_type, str(o1_count),
                               o2_type, str(o2_count),
                               other_o]) + "\n")
                               
def get_read_blocks(read):
    #输入pysam的read比对对象
    #输出该read比对由于splicing junction拆分成的block。
    #过滤掉小的删除、插入和错配等。
    #按染色质位置从小到大排列。
    #是一个列表，每一个元素是一个起始位置和终止位置（均是1-based）组成的列表。
    blocks = []
    start = read.reference_start + 1
    end = start - 1
    for (flag, length) in read.cigartuples:
        if flag == 4 or flag == 5 or flag == 1: continue
        if flag == 0 or flag == 2:
            end += length
        if flag == 3:
            blocks.append([start, end])
            start = end + length + 1
            end = start - 1
    blocks.append([start, end])
    return blocks

def trim_read_blocks(read_blocks, start, end):
    #将read_blocks分成三部分，返回这三部分组成的列表
    #每一部分本身是一个列表，每个元素是[block_start, block_end]
    #第一部分是start-end区间左侧的blocks，
    #第二部分是与start-end区间重叠的blocks
    #第三部分是start-end区间右侧的blocks
    
    block_total_num = len(read_blocks)
    #计算每个read block是不是在start-end的闭区间内。
    in_flags = []
    for s, e in read_blocks:
        if e < start: 
            #block在区间左侧或右侧
            in_flag = 0
        elif s > end:
            in_flag = 2
        else:
            in_flag = 1
        in_flags.append(in_flag)
    
    #寻找中间连续的满足在闭区间的block
    blocks_left_end_index = -1
    blocks_start_index = block_total_num
    blocks_end_index = -1
    blocks_right_start_index = block_total_num
    for i, in_flag in enumerate(in_flags):
        if in_flag == 0:
            blocks_left_end_index = i
        elif in_flag == 1:
            if blocks_start_index == block_total_num: #说明是第一个出现的1
                blocks_end_index = blocks_start_index = blocks_left_end_index + 1
            else:
                blocks_end_index = i
        elif in_flag == 2:
            blocks_right_start_index = i
            break
    
    left_blocks = read_blocks[:(blocks_left_end_index+1)] if blocks_left_end_index != -1 else []
    inner_blocks = read_blocks[blocks_start_index:(blocks_end_index+1)] if blocks_start_index != block_total_num else []
    right_blocks = read_blocks[blocks_right_start_index:] if blocks_right_start_index != block_total_num else []
    return([left_blocks, inner_blocks, right_blocks])

def iter_intron_read_type(file_bam, file_intron_pos, min_overlap=6, strand_flag=0, fileout_read_info=""):
    
    filter_tag=["unmapped", "remove_duplicated"]
    feature_aligns = iter_bam_feature(file_bam, file_intron_pos, 
                     filter_bam_params={"filter_tag": filter_tag}, up=0, down=0, method=2)
    if fileout_read_info:
        READ_INFO_OUT = open(fileout_read_info, 'w')
        READ_INFO_OUT.write("intron_id\tchr_name\tintron_start\tintron_end\tintron_strand\tflag\tstr_blocks\tis_read1\tstrand\tread_name\n")
    
    for [intron_id, chr_name, intron_start, intron_end, intron_strand], aligns in feature_aligns:

        #边界位置（包含）
        #第一个1表示是intron的左端，2表示intron的右端。
        #第二个1表示左区间，2表示右区间。
        #判定overlap是否够用的：
        xover11 = intron_start - min_overlap
        xover12 = intron_start + min_overlap - 1
        xover21 = intron_end - min_overlap + 1
        xover22 = intron_end + min_overlap
        
        intron_reads = [] #intron内部read，没有任何junction跨该内含子
        this_intron_data = []
        for na, start, end, strand, read in aligns:
            
            #1.如果是链特异性的，过滤掉方向不一致的
            if strand_flag == 1:
                if read.is_read1:
                    if strand == intron_strand: continue
                else:
                    if strand != intron_strand: continue
            elif strand_flag == 2:
                if read.is_read1:
                    if strand != intron_strand: continue
                else:
                    if strand == intron_strand: continue
            origin_read_blocks = get_read_blocks(read)
            
            #2.将block分离成三个部分
            left_blocks, blocks, right_blocks = trim_read_blocks(origin_read_blocks, intron_start, intron_end)
                            
            #3.提取覆盖intron的junction信息
            #junction, junction_start, junction_end用于后面的计算
            left_junction = ""
            right_junction = ""
            inner_junction = ""
            if left_blocks:
                junction_start = left_blocks[-1][1] + 1
                if blocks and (blocks[0][0] >= intron_start): left_junction = str(junction_start) + "-" + str(blocks[0][0] - 1)
            if right_blocks:
                junction_end = right_blocks[0][0] - 1
                if blocks and (blocks[-1][1] <= intron_end): right_junction = str(blocks[-1][1] + 1) + "-" + str(junction_end)
            if left_blocks and right_blocks and (not blocks):
                inner_junction = str(junction_start) + "-" + str(junction_end)
            if len(blocks) >= 2:
                for i in range(len(blocks)-1):
                    if inner_junction: inner_junction += ":"
                    inner_junction = str(blocks[i][1] + 1) + "-" + str(blocks[i+1][0] - 1)
            junction = left_junction
            if inner_junction:
                if junction: junction += ":"
                junction += inner_junction
            if right_junction:
                if junction: junction += ":"
                junction += right_junction
            
            #4. 计算read剪切类型
            if blocks:
                block_start = blocks[0][0]
                block_end = blocks[-1][1]
                
            if not blocks:
                #没有blocks，肯定有left_blocks和right_blocks。
                #肯定是有junction跨这个intron。有可能恰好是这个intron。
                #有可能比这个intron长。
                if junction_start == intron_start:
                    if junction_end == intron_end:
                        flag = ["c"]   #正确的剪切
                    else:
                        flag = ["o", "", "1"] #3'外侧剪切
                else:
                    if junction_end == intron_end:
                        flag = ["o", "1"] #5’ 外侧剪切
                    else:
                        #5'和3’外侧剪切 （注意这种有可能是Exon skipping）
                        flag = ["o", "1", "1"]
            elif len(blocks) == 1:
                if block_start < intron_start:
                    #表明a也可能是o|0,o|0|1的片段
                    #--------...............--------- gene structure
                    #      -------------------------  ["ab"]
                    #      ------------.....--------- ["o", "0"]  5' 内侧剪切
                    #      ------------.........----  ["o", "0", "1"] 5内3外
                    #      --------                   ["a"]
                    if block_end > intron_end:
                        flag = ["ab"]  #block跨整个intron
                    else:
                        if right_blocks:
                            if junction_end == intron_end:
                                flag = ["o", "0"]  #5' 内侧剪切
                            else:
                                flag = ["o", "0", "1"]
                        else:
                            flag = ["a"]
                else:
                    #o||0u除了可能是o||0外，也可能是o|||e1, o||1|e1
                    #o|1|0u除了可能是o|1|0外，也可能是o|1||e1, o|1|1|e1 
                    #o|0u|除了可能是o|0外，也可能是o|||e1, o|1||e1
                    #o|0u|1除了可能是o|0外，也可能是o||1|e1, o|1|1|e1
                    #i可以是多种可能
                    #不过后两者的概率较小
                    #--------...............--------- gene structure
                    #  ------.....--------------      ["o","", "0"] 3内
                    #  ------.....-----.....------    ["o","","","e1"] ES
                    #  ------.....-----.......-----   ["o","", "1", "e1"]
                    #  ------.....-----               ["o","", "0u"]
                    #
                    #  ---........-------------       ["o","1", "0"] 5外3内
                    #  ---........-----.....------    ["o","1","", "e1"]
                    #  ---........-----.......-----   ["o","1", "1", "e1"]
                    #  ---........-----               ["o","1", "0u"]
                    #
                    #             ---------------     ["b"]
                    #             -----.....------    ["o","0u"]
                    #             -----........---    ["o","0u", "1"]
                    #             ------              ["i"]
                    if left_blocks:
                        if junction_start == intron_start:
                            if block_end > intron_end:
                                flag = ["o","", "0"]
                            else:
                                if right_blocks:
                                    if junction_end == intron_end:
                                        flag = ["o","","","e1"] #exon skipping
                                    else:
                                        flag = ["o","", "1", "e1"]
                                else:
                                    flag = ["o","", "0u"]#表示intron 3'没覆盖到                                    
                        else:
                            if block_end > intron_end:
                                flag = ["o","1", "0"]
                            else:
                                if right_blocks:
                                    if junction_end == intron_end:
                                        flag = ["o","1","", "e1"]
                                    else:
                                        flag = ["o","1", "1", "e1"]
                                else:
                                    flag = ["o","1", "0u"]#表示intron 3’没覆盖到
                    else:
                        if block_end > intron_end:
                            flag = ["b"]
                        else:
                            if right_blocks:
                                if junction_end == intron_end:
                                    flag = ["o","0u"]
                                else:
                                    flag = ["o","0u", "1"]
                            else:
                                flag = ["i"]
            elif len(blocks) >= 2: 
                #--------...............--------- gene structure
                #     ------.......--------
                #  ------...------......---
                #  ----.....------........----
                e_num = len(blocks)
                flag = ["o", "", ""]
                if block_start < intron_start:
                    e_num -= 1
                    flag[1] = "0"
                else:
                    if left_blocks:
                        if junction_start < intron_start:
                            flag[1] = "1"
                        else:
                            flag[1] = ""
                    else:
                        flag[1] = "u"
                if block_end > intron_end:
                    e_num -= 1
                    flag[2] = "0"
                else:
                    if right_blocks:
                        if junction_end > intron_end:
                            flag[2] = "1"
                        else:
                            flag[2] = ""
                    else:
                        flag[2] = "u"
                if e_num:
                    flag.append("e" + str(e_num))
            
            #5. 计算两端覆盖是否满足条件
            #如果两侧有无关的block，说明比对可信。不用考虑覆盖问题。
            flag_overlap = True
            if flag[0] == "a":
                flag_left_block_overlap = (len(left_blocks) > 0) or (blocks[0][0] <= xover11)
                flag_right_block_overlap = blocks[0][1] >= xover12
                flag_overlap = flag_left_block_overlap and flag_right_block_overlap
            elif flag[0] == "b":
                flag_left_block_overlap =  blocks[0][0] <= xover21
                flag_right_block_overlap = (len(right_blocks) > 0) or (blocks[0][1] >= xover22)
                flag_overlap = flag_left_block_overlap and flag_right_block_overlap
            elif flag[0] == "ab":
                flag_left_block_overlap = (len(left_blocks) > 0) or (blocks[0][0] <= xover11)
                flag_right_block_overlap = (len(right_blocks) > 0) or (blocks[0][1] >= xover22)
                if flag_left_block_overlap:
                    if not flag_right_block_overlap:
                        flag[0] = "a"
                    else:
                        flag[0] = "ab"
                else:
                    if not flag_right_block_overlap:
                        flag[0] = "ab"
                        flag_overlap = False
                    else:
                        flag[0] = "b"
            elif flag[0] == "i":
                flag_overlap = True
            else:
                flag_left_block_overlap = False
                flag_right_block_overlap = False
                if left_blocks:
                    flag_left_block_overlap = (len(left_blocks)>1) or (left_blocks[0][1] - left_blocks[0][0] + 1 >= min_overlap)
                else:
                    if block_start < intron_start:
                        if block_start <= xover11:
                            flag_left_block_overlap = True
                    else:
                        if (blocks[0][1] - blocks[0][0] + 1) >= min_overlap:
                            flag_left_block_overlap = True
                if right_blocks:
                    flag_right_block_overlap = (len(right_blocks)>1) or (right_blocks[0][1] - right_blocks[0][0] + 1 >= min_overlap)
                else:
                    if block_end > intron_end:
                        if block_end >= xover22:
                            flag_right_block_overlap = True
                    else:
                        if (blocks[-1][1] - blocks[-1][0] + 1) >= min_overlap:
                            flag_right_block_overlap = True
                flag_overlap = flag_left_block_overlap and flag_right_block_overlap
            
            #6. 如果intron_strand为负，调整flag信息
            if flag[0] == "o":
                #一定不要放到下面的if语句里，因为intron_strand为+也要这样操作。要不然会导致不一致
                flag = flag + [""]*(4-len(flag))
            if intron_strand == "-":
                if flag[0] == "a":
                    flag = ["b"]
                elif flag[0] == "b":
                    flag = ["a"]
                elif flag[0] == "ab":
                    pass
                elif flag[0] == "c":
                    pass
                elif flag[0] == "i":
                    pass
                else:
                    ls5, ls3 = flag[2], flag[1]
                    flag[1] = ls5
                    flag[2] = ls3
            
            #7. 将flag由列表转化为字符串，暂时不考虑u，因此去掉u             
            flag = "|".join(flag)
            flag = flag.replace("u", "")
            
            #8. 将flag上添加上junction信息
            if junction and (flag[0] != "c"):
                flag = flag + ":" + junction
            
            #9. 记录intron_reads。用于计算intron coverage
            if flag == "a" or flag == "b" or flag == "ab" or flag == "i":
                intron_reads.append([block_start, block_end])
            
            #10. 如果两侧overlap达不到要求，则加上p
            if not flag_overlap:
                flag = "p" + flag
            # str(start), str(end), strand, 
            str_blocks = ":".join([",".join([str(s) + "-" + str(e) for s, e in left_blocks]),
                                   ",".join([str(s) + "-" + str(e) for s, e in blocks]),
                                   ",".join([str(s) + "-" + str(e) for s, e in right_blocks])])
            if fileout_read_info:
                READ_INFO_OUT.write("\t".join([intron_id, chr_name, str(intron_start), str(intron_end), intron_strand, 
                                    flag, str_blocks, str(read.is_read1), strand, read.query_name]) + "\n") 
            this_intron_data.append([flag, read.query_name, str_blocks])
            
            
        yield([intron_id, chr_name, intron_start, intron_end, intron_strand], this_intron_data, intron_reads)
    READ_INFO_OUT.close()

def stat_intron_coverage(intron_reads, intron_start, intron_end):
    if not intron_reads: return 0
    intron_length = intron_end - intron_start + 1
    intron_reads.sort(key=lambda x: x[0])
    start, end = intron_reads[0]
    if start < intron_start: start = intron_start
    if end >= intron_end:
        return((intron_end - start + 1)/intron_length)
    total_coverage = 0
    if len(intron_reads) > 1:
        for s, e in intron_reads[1:]:
            if s <= end:
                if e > end:
                    if e >= intron_end:
                        return((total_coverage + (intron_end - start +1))/intron_length)
                    end = e
            else:
                total_coverage += (end - start + 1)
                start = s
                if e >= intron_end:
                    return((total_coverage + (intron_end - start +1))/intron_length)
                else:
                    end = e
    total_coverage += end - start + 1
    return(total_coverage/intron_length)

def cal_junction_length(flag):
    d = flag.split(":")
    if len(d) == 1: return 0
    d = d[1:]
    length = 0
    for j in d:
        s, e = j.split("-")
        s, e = int(s), int(e)
        length += abs(e - s) + 1
    return length
 
main()
                           
                           
"""
#暂时使用不上
    #有些外显子才4个bp如
    #AT1G02305
    #AT1G63290
    #AT1G63770
    #AT2G39780
    #AT2G41520
    #AT3G01850
    #AT4G01610
    #AT5G51700
    #除此之外，鉴定到oa只有这两个，他们虽然基因组未注释，但read确实支持
    #AT2G35020  基因组未注释，但read确实支持这有个可变剪切
    #AT3G47560  基因组未注释，但read确实支持这有个可变剪切
    #因此，不再关心oa的问题

                           
def get_introns(blocks, return_all_featurs=0):
    
    if return_all_featurs:
        featurs = [blocks[0]]
        featurs[0].append(1)
    else:
        introns = []
        
    if len(blocks) > 1:
        last_start = blocks[0][1] + 1
        for start, end in blocks[1:]:
            if return_all_featurs:
                featurs.append([last_start, start-1, 0])
                featurs.append([start, end, 1])
            else:
                introns.append([last_start, start-1])
            last_start = end + 1
    if return_all_featurs:
        return featurs
    else:
        return introns

def get_read_introns(read):
    return get_introns(get_read_blocks(read))
    
def get_read_exon_introns(read):
    exons = get_read_blocks(read)
    introns = get_introns(exons)
    return([exons, introns])
    
def get_read_features(read):
    exons = get_read_blocks(read)
    return get_introns(exons, return_all_featurs=1)
    

def filter_read_exons(read_exons, min_overlap=6):
    ##过滤read_exons的两端的block，如果小于min_overlap则过滤掉
    filter5, filter3 = 0, 0
    s, e = read_exons[0]
    if e - s + 1 < min_overlap: filter5 = 1
    if len(read_exons) > 1:
        s, e = read_exons[-1]
        if e - s + 1 < min_overlap: filter3 = 1
    if filter5 or filter3:
        filter_exons = read_exons.copy()
        if filter5: filter_exons.pop(0)
        if filter3: filter_exons.pop(-1)
        return filter_exons
    else:
        return read_exons
"""
