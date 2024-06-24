# ChIP-seq Pipelines

ChIP-seq is a method used to analyze protein interactions with DNA. ChIP-seq combines chromatin immunoprecipitation with DNA sequencing to infer the possible binding sites of DNA-associated proteins. 

The histone ChIP-seq pipeline, described here, is suitable for proteins that associate with DNA over longer regions or domains.

The transcription factor ChIP-seq (TF ChIP-seq) pipeline is suitable for proteins that are expected to bind in a punctate manner, and may be needed to modified.


## Inputs
| File format | Information contained in file | File description | 
|---------- |---------- |---------- |
| fastq | reads | G-zipped reads, paired-ended or single ended, stranded or unstranded. | 
| fasta | genome indices (bowtie2) | Indices are dependent on the assembly being used for mapping | 

## Output
| File format | Information contained in file | File description |
|---------- |---------- |---------- |
| bigWig | raw signal (bw_files), fold change over control (bw_compare) | signal coverage tracks |
| narrowPeak | peaks | Peak calls for each replicate individually. |
| png | figure | metaplot around the gene body |

## Notes

To compare the signal between, the [sample_table.tsv](./sample_table.tsv) is required.

Due to issue with MASC2 installation, it is recommended to use singularity to run conda:

You can also pass additional arguments to singularity, including bind points, like this:

`snakemake --use-singularity --singularity-args "-B /path/outside/container/:/path/inside/container/"`

More details, you can see the [Snakemake + docker example, how to use volumes](https://stackoverflow.com/questions/52742698/snakemake-docker-example-how-to-use-volumes)

## File details

```
.
├── aligned_data
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.sorted.rmdup.bam
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.sorted.rmdup.bam.bai
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.sorted.rmdup.bam.markdump.txt
│   ├── col_input_ChIPseq_GSM4275149.1.sorted.rmdup.bam
│   ├── col_input_ChIPseq_GSM4275149.1.sorted.rmdup.bam.bai
│   └── col_input_ChIPseq_GSM4275149.1.sorted.rmdup.bam.markdump.txt
├── bw_compare
│   └── col_H3K4me3_ChIPseq_GSM4275144.1.compare.bw
├── bw_files
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.sorted.rmdup.CPM.bw
│   └── col_input_ChIPseq_GSM4275149.1.sorted.rmdup.CPM.bw
├── config.yml
├── deeptools_profile
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.scale.png
│   └── col_input_ChIPseq_GSM4275149.1.scale.png
├── macs2_result
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1_control_lambda.bdg
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1_model.r
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1_peaks.narrowPeak
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1_peaks.xls
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1_summits.bed
│   └── col_H3K4me3_ChIPseq_GSM4275144.1_treat_pileup.bdg
├── raw_data
│   ├── col_H3K4me3_ChIPseq_GSM4275144.1.fastq.gz
│   └── col_input_ChIPseq_GSM4275149.1.fastq.gz
├── sample_table.tsv
├── Snakefile
└── snakemake
    ├── callpeak.smk
    ├── chipseq_pe.smk
    └── chipseq_se.smk
```