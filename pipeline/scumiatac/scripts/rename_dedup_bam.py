import pysam
import click
from tqdm import tqdm
import os

@click.command()
@click.argument('input_bam', type=click.Path(exists=True))
@click.argument('output_bam', type=click.Path())
def main(input_bam, output_bam):
    """
    读取 BAM 文件，提取 CB 和 RX 标签，将其添加到 Read 名称的前缀中。
    格式: {CB}_{RX}_{OriginalName}
    """
    
    # 1. 预读取以获取进度条的总数 (可选，如果文件巨大可以跳过这一步以节省时间)
    print(f"[*] 正在读取输入文件索引信息: {input_bam}")
    try:
        # 尝试快速获取总 reads 数 (需要索引)
        idx_stats = pysam.idxstats(input_bam)
        total_reads = sum([int(line.split('\t')[2]) for line in idx_stats.split('\n') if line])
    except Exception:
        # 如果没有索引或获取失败，则不显示总数
        total_reads = None
        print("    提示: 未找到索引文件 (.bai)，进度条将不显示剩余时间。")

    print(f"[*] 开始处理...")
    
    # 2. 打开输入输出文件
    with pysam.AlignmentFile(input_bam, "rb") as infile, \
         pysam.AlignmentFile(output_bam, "wb", template=infile) as outfile:
        
        # 使用 tqdm 显示进度
        pbar = tqdm(infile, total=total_reads, unit=" reads", mininterval=1.0)
        
        processed_count = 0
        renamed_count = 0
        
        for read in pbar:
            processed_count += 1
            
            # 获取原始名称
            original_name = read.query_name
            
            # 提取标签，如果没有则返回 'None'
            # 注意: 根据你的数据情况，CB/RX 可能是字符串，也可能是其他类型，通常是字符串
            try:
                cb = read.get_tag("CB")
            except KeyError:
                cb = "None"
            
            try:
                rx = read.get_tag("RX")
            except KeyError:
                rx = "None"
            
            # 这里按照你的需求：强行改名保持格式统一
            
            # 构造新名称: {CB}_{RX}_{ReadName}
            new_name = f"{cb}_{rx}_{original_name}"
            
            # 修改 Read 名称
            read.query_name = new_name
            
            # 写入输出文件
            outfile.write(read)
            renamed_count += 1

    print(f"\n[*] 处理完成!")
    print(f"    输出文件: {output_bam}")
    print(f"    共处理 Reads: {processed_count}")
    print(f"    已重命名 Reads: {renamed_count}")

if __name__ == "__main__":
    main()