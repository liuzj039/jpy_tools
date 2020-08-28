library("DESeq2")

#4个参数，前三个输入文件，第四个输出文件
#输入文件1是各基因（行）在各文库（列）的count值。是逗号隔开的csv文件。
#输入文件2是各文库的组名。具有文件header，需要两列，一列是sample，一列是group
#输入文件3要比较的两个组，不具有文件header，第一个列是第一个组，第二列是第二个组，
#第三列可选，如果有第三列，则为0和1的值，0表示这一列数据不用，如果没有第三列，则
#所有数据都用
#输出文件是每个基因每一对组用DESeq2计算的padj值，可以根据需要选取0.1，0.05和0.01作为阈值
#输出文件的列名由输入文件3指定，如WT和M1,则列名为"WT_VS_M1_padj"

args <- commandArgs(T)

filedata = args[1]
filecondition = args[2]
filepair = args[3]
fileout =  args[4]
tab_or_csv = 0

if (tab_or_csv == 1){
	data = read.table(filedata, header=T)
} else {
	data = read.csv(filedata)
}
colnames(data)[1] = "id"
rownames(data) = data$id

condition_data = read.table(filecondition,header=T,stringsAsFactors=F, sep="\t")
conditions = unique(condition_data$group)
pair_conditions = read.table(filepair)
if (ncol(pair_conditions) == 3) {
  pair_conditions = pair_conditions[pair_conditions[,3] ==1, ]
}
pair_conditions = t(pair_conditions[, 1:2])


my_deseq_core <- function(countData, colData) {
  dds <- DESeqDataSetFromMatrix(countData = countData,
                                colData = colData,
                                design = ~ condition)
  dds <- DESeq(dds)
  res <- results(dds)
  res["id"] <- rownames(res)
  res <- res[c("id","pvalue","padj")]
  res$pvalue[is.na(res$pvalue)] <- 1
  res$padj[is.na(res$padj)] <- 1
  return(res)
}

my_deseq <- function(data, condition_data, condition_pair ) {
  #condition_pair  <- c("WT","M1")
  ls_conditions = as.character(condition_data$group[condition_data$group %in% condition_pair])
  ls_samples = as.character(condition_data$sample[condition_data$group %in% condition_pair])
  ls_colData = data.frame("condition" = ls_conditions)
  rownames(ls_colData) = ls_samples
  ls_countData = data[ls_samples]
  res = my_deseq_core(ls_countData, ls_colData)
  return(res)
}

deseq_results = data.frame("id" = rownames(data))

nm = apply(pair_conditions,2,function(x) deseq_results[paste(x[1],"_VS_",x[2],"_padj",sep="")] <<- my_deseq(data,condition_data,x)$padj)
write.table(deseq_results, fileout, quote=F, sep="\t", row.names=F)

