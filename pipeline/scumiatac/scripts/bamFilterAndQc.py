import pysam
import click
import pandas as pd
from collections import defaultdict, Counter
import time
import os

# 定义统计数据的容器
class RunStats:
    def __init__(self):
        self.total_pairs = 0      # 总 Fragment/Pair 数
        self.total_reads = 0      # 总 Read 数
        self.retained_pairs = 0   # 保留的 Fragment/Pair 数
        self.retained_reads = 0   # 保留的 Read 数
        self.start_time = time.time()

    @property
    def filtered_pairs(self):
        return self.total_pairs - self.retained_pairs

    @property
    def filtered_reads(self):
        return self.total_reads - self.retained_reads

    @property
    def retention_rate(self):
        if self.total_pairs == 0: return 0.0
        return (self.retained_pairs / self.total_pairs) * 100

def is_good_read(read):
    """
    检查单条 Read 是否符合质量要求 (XA/SA/Unmapped)
    注意：is_secondary 的检查已经移到主循环中，这里只检查主要比对的质量
    """
    if read.is_unmapped:
        return False
    # 过滤掉有多重比对标签 (BWA specific)
    if read.has_tag('XA'): 
        return False
    # 过滤掉 Split Alignment
    if read.has_tag('SA'): 
        return False
    return True

def process_read_group(read_list, outfile, bc_stats, umi_global_stats, stats):
    """
    处理一组同名的 Reads。
    逻辑：同生共死。如果组内所有 Read 都合格，则全部输出；否则全部丢弃。
    """
    if not read_list:
        return

    # 1. 记录统计信息 (Total)
    stats.total_pairs += 1
    stats.total_reads += len(read_list)

    # 2. 检查整组是否都合格
    all_passed = True
    for r in read_list:
        if not is_good_read(r):
            all_passed = False
            break
    
    # 如果不合格，直接返回 (丢弃)
    if not all_passed:
        return

    # 3. 检查 UMI 格式 (N content)
    # 取第一条 read 解析名称即可
    first_read = read_list[0]
    try:
        parts = first_read.query_name.split('_')
        if len(parts) != 3:
            # 格式不对，视为不合格 (或者你可以选择保留但计入 log)
            return
        
        base_id = parts[0]
        barcode = parts[1]
        umi = parts[2]

        if 'N' in umi:
            return
    except Exception:
        return

    # 4. 通过所有检查 -> 执行写入和统计
    stats.retained_pairs += 1
    stats.retained_reads += len(read_list)

    # --- 统计业务数据 ---
    # Read Count: 列表里有几条就加几
    for _ in read_list:
        bc_stats[barcode]['read_count'] += 1
    
    # UMI Count: 一个 Pair 贡献一个 UMI 计数
    bc_stats[barcode]['umis'].add(umi)
    umi_global_stats[f"{barcode}_{umi}"] += 1

    # --- 修改名称并写入 ---
    # 构造新名称: V...:0:0:0:UMI_BarcodeUMI
    id_prefix = base_id.rsplit(':', 1)[0]
    new_name = f"{id_prefix}:UMI_{barcode}{umi}"

    for r in read_list:
        r.query_name = new_name
        r.set_tag('CB', barcode)
        r.set_tag('RX', umi)
        outfile.write(r)

def write_log(log_path, stats, input_path, output_path):
    """
    输出日志文件
    """
    elapsed_time = time.time() - stats.start_time
    
    log_content = [
        "=======================================================",
        "               BAM Filtering & UMI Processing Log",
        "=======================================================",
        f"Input File       : {input_path}",
        f"Output File      : {output_path}",
        f"Processing Time  : {elapsed_time:.2f} seconds",
        "-------------------------------------------------------",
        "Statistics:",
        f"  Total Input Pairs (Fragments) : {stats.total_pairs:,}",
        f"  Total Input Reads             : {stats.total_reads:,}",
        "",
        f"  Retained Pairs                : {stats.retained_pairs:,}",
        f"  Retained Reads                : {stats.retained_reads:,}",
        "",
        f"  Filtered Pairs                : {stats.filtered_pairs:,}",
        f"  Filtered Reads                : {stats.filtered_reads:,}",
        "-------------------------------------------------------",
        f"  Retention Rate (Pairs)        : {stats.retained_pairs / stats.total_pairs * 100:.2f}%" if stats.total_pairs > 0 else "  Retention Rate : 0.00%",
        "======================================================="
    ]
    
    # 打印到屏幕
    print("\n".join(log_content))
    
    # 写入文件
    with open(log_path, 'w') as f:
        f.write("\n".join(log_content) + "\n")

def process_bam_and_stats(input_bam_path, output_bam_path):
    
    bc_stats = defaultdict(lambda: {'read_count': 0, 'umis': set()})
    umi_global_stats = Counter()
    stats = RunStats()

    print(f"[*] 开始处理文件: {input_bam_path}")
    print("    注意: 必须输入 Name-Sorted 的 BAM 文件！")

    with pysam.AlignmentFile(input_bam_path, "rb") as infile, \
         pysam.AlignmentFile(output_bam_path, "wb", template=infile) as outfile:

        read_buffer = []
        last_query_name = None

        for read in infile:
            # 预过滤：次级比对和补充比对直接丢弃，不进入 Buffer，也不参与“同生共死”判断
            # 我们只关心 Primary Alignment 对是否合格
            if read.is_secondary or read.is_supplementary:
                continue

            current_name = read.query_name

            if last_query_name is not None and current_name != last_query_name:
                process_read_group(read_buffer, outfile, bc_stats, umi_global_stats, stats)
                read_buffer = []
            
            read_buffer.append(read)
            last_query_name = current_name
            
            # 简单的进度打印
            if stats.total_pairs > 0 and stats.total_pairs % 500000 == 0:
                # 注意：这里的 total_pairs 只有在 process_read_group 被调用后才更新
                # 所以这个进度条可能稍微有一点滞后，但无伤大雅
                pass 

        # 处理最后一组
        if read_buffer:
             process_read_group(read_buffer, outfile, bc_stats, umi_global_stats, stats)

    # 整理 DataFrame
    print("[*] 正在生成统计报表...")
    
    bc_data_list = []
    for bc, data in bc_stats.items():
        bc_data_list.append({
            'Barcode': bc,
            'Read_Count': data['read_count'],
            'UMI_Count': len(data['umis'])
        })
    
    df_barcode = pd.DataFrame(bc_data_list)
    if not df_barcode.empty:
        df_barcode = df_barcode.sort_values(by='UMI_Count', ascending=False)

    df_umi = pd.DataFrame(umi_global_stats.items(), columns=['UMI', 'Read_Count'])
    if not df_umi.empty:
        df_umi = df_umi.sort_values(by='Read_Count', ascending=False)

    return df_barcode, df_umi, stats


@click.command()
@click.argument('bam_path', type=click.Path(exists=True))
@click.argument('bam_output_path', type=click.Path())
def main(bam_path, bam_output_path):
    df_barcode, df_umi, stats = process_bam_and_stats(bam_path, bam_output_path)
    
    # 保存表格
    df_barcode.to_csv(bam_output_path + '.barcode_stats.tsv', sep='\t', index=False)
    df_umi.to_csv(bam_output_path + '.umi_stats.tsv', sep='\t', index=False)
    
    # 保存并打印 Log
    log_path = bam_output_path + '.log'
    write_log(log_path, stats, bam_path, bam_output_path)

if __name__ == "__main__":
    main()