import pysam
import lmdb
import click
import pickle
import os
from tqdm import tqdm

# 定义需要提取的 Tag 列表
TARGET_TAGS = ['cD', 'cM', 'cE', 'cd', 'ce']

@click.command()
@click.argument('bam_path', type=click.Path(exists=True))
@click.argument('lmdb_path', type=click.Path())
@click.option('--map-size', default=1099511627776, help='LMDB 最大虚拟容量 (默认 1TB)')
@click.option('--batch-size', default=100000, help='批量写入的条数 (默认 10w)')
def main(bam_path, lmdb_path, map_size, batch_size):
    """
    读取 BAM 文件，提取指定 Tag (cD, cM, cE, cd, ce)，
    并将 {ReadName: TagsDict} 存入 LMDB 数据库。
    """
    
    # 1. 检查并清理旧数据库 (LMDB 是文件夹或文件，取决于系统，但在 Python binding 中通常指定路径)
    # 如果路径存在且是文件夹，lmdb 会在里面建立 data.mdb
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    print(f"[*] Input BAM  : {bam_path}")
    print(f"[*] Output LMDB: {lmdb_path}")
    
    # 2. 打开 LMDB 环境
    # map_size 设置得很大是安全的，因为 OS 只会分配实际使用的物理空间
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    # 3. 读取 BAM 统计总数 (为了进度条)
    print("[*] 读取 BAM 索引以计算总 Reads 数...")
    try:
        idx_stats = pysam.idxstats(bam_path)
        total_reads = sum([int(line.split('\t')[2]) for line in idx_stats.split('\n') if line])
    except Exception:
        total_reads = None
        print("    提示: 无索引，无法显示进度百分比。")

    # 4. 开始处理
    print("[*] 开始提取 Tags 并写入数据库...")
    
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as infile:
        
        # 使用 tqdm 显示进度
        pbar = tqdm(infile, total=total_reads, unit="reads", mininterval=1.0)
        
        # 开启 LMDB 写事务
        txn = env.begin(write=True)
        count = 0
        inserted_count = 0
        
        for read in pbar:
            read_name = read.query_name
            
            # 构建 Tag 字典
            tag_data = {}
            for tag in TARGET_TAGS:
                try:
                    val = read.get_tag(tag)
                    tag_data[tag] = val
                except KeyError:
                    tag_data[tag] = None
            
            # 序列化 (Pickle)
            # key 必须是 bytes，value 必须是 bytes
            k = read_name.encode('utf-8')
            v = pickle.dumps(tag_data)
            
            # 写入事务缓冲区
            txn.put(k, v)
            
            count += 1
            inserted_count += 1
            
            # 批量提交 (Batch Commit)
            if count % batch_size == 0:
                txn.commit()
                # 开启下一轮事务
                txn = env.begin(write=True)
                pbar.set_description(f"Committed {count} records")

        # 处理剩余的数据
        txn.commit()
        
    env.close()
    
    print(f"\n[*] 完成！共存入 {inserted_count} 条记录。")
    print(f"    数据库位置: {lmdb_path}")

# ==========================================
# 附赠：如何读取这个数据库的示例代码
# ==========================================
def example_read_lmdb(lmdb_path, read_name_query):
    """
    演示如何从生成的 LMDB 中查询数据
    """
    print(f"\n[Demo] 查询 Read: {read_name_query}")
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # 查询
        cursor = txn.cursor()
        val_bytes = txn.get(read_name_query.encode('utf-8'))
        
        if val_bytes:
            # 反序列化
            data = pickle.loads(val_bytes)
            print(f"    Found Tags: {data}")
        else:
            print("    Read not found in DB.")
    env.close()

if __name__ == "__main__":
    main()