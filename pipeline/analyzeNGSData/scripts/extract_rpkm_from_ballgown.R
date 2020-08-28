library("ballgown")

args <- commandArgs(T)

sample_label <- args[1]
sample_path <- args[2]
file_mRNA_out <- args[3]
file_gene_out <- args[4]

sample_paths <- c(sample_path)

#sample_label <- "GSM1571988"
#sample_paths <- c("GSM1571988")
#file_mRNA_out <- "a.txt"
#file_gene_out <- "b.txt"


bg <- ballgown(sample_paths)
mRNA_names <- transcriptNames(bg)
#gene_names <- geneIDs(bg)  # gene names of mRNAs

mRNA_fpkm <- data.frame(id=mRNA_names)
mRNA_fpkm[sample_label] <- texpr(bg)[,1]

gene_fpkm_values <- gexpr(bg)
gene_fpkm <- data.frame(id=rownames(gene_fpkm_values))
gene_fpkm[sample_label] <- gene_fpkm_values[,1]

write.table(mRNA_fpkm, file_mRNA_out, quote=F, sep="\t", row.names=F)
write.table(gene_fpkm, file_gene_out, quote=F, sep="\t", row.names=F)