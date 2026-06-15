#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -ne 3 ]; then
    echo "用法: $0 <输入的FASTA文件(支持.gz)> <分割份数> <输出文件夹>"
    echo "示例: $0 input.fasta.gz 5 ./output_dir"
    exit 1
fi

FASTA_FILE=$1
NUM_PARTS=$2
OUT_DIR=$3

# 检查输入文件是否存在
if [ ! -f "$FASTA_FILE" ]; then
    echo "错误: 找不到文件 '$FASTA_FILE'"
    exit 1
fi

# 检查分割份数是否为正整数
if ! [[ "$NUM_PARTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: 分割份数必须是正整数！"
    exit 1
fi

# 创建输出文件夹
mkdir -p "$OUT_DIR"

# 判断是否为 gzip 压缩文件，动态选择读取命令
# 使用 gunzip -c 可以兼容不同系统（macOS/Linux），且不会修改源文件
if [[ "$FASTA_FILE" == *.gz ]]; then
    READ_CMD="gunzip -c"
    echo "检测到 .gz 压缩文件，将以流式读取..."
else
    READ_CMD="cat"
fi

# 统计总序列数
echo "正在统计序列总数..."
TOTAL_SEQS=$($READ_CMD "$FASTA_FILE" | grep -c "^>")

if [ "$TOTAL_SEQS" -eq 0 ]; then
    echo "错误: 在 '$FASTA_FILE' 中没有找到任何序列（缺少 '>' 开头的行）。"
    exit 1
fi

echo "共发现 $TOTAL_SEQS 条序列，准备分割为 $NUM_PARTS 份..."

# 调整份数逻辑
if [ "$NUM_PARTS" -gt "$TOTAL_SEQS" ]; then
    echo "警告: 分割份数 ($NUM_PARTS) 大于序列总数 ($TOTAL_SEQS)。自动将份数调整为 $TOTAL_SEQS。"
    NUM_PARTS=$TOTAL_SEQS
fi

# 使用管道将读取的数据直接传递给 awk 处理
$READ_CMD "$FASTA_FILE" | awk -v total="$TOTAL_SEQS" -v parts="$NUM_PARTS" -v out="$OUT_DIR" '
BEGIN {
    seq_count = 0;
    file_idx = 1;
    seqs_per_part = int(total / parts);
    remainder = total % parts;
    current_limit = seqs_per_part + (file_idx <= remainder ? 1 : 0);
}
/^>/ {
    seq_count++;
    if (seq_count > current_limit && file_idx < parts) {
        file_idx++;
        seq_count = 1;
        current_limit = seqs_per_part + (file_idx <= remainder ? 1 : 0);
    }
}
{
    print $0 > (out "/" file_idx ".fa")
}
'

echo "分割完成！文件已保存至: $OUT_DIR/"
