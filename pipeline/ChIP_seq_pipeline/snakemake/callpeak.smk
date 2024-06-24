sample_table = config['sample_table']
treatment_vs_control = {}
with open(sample_table, 'r') as f:
    for line in f:
        if line[0] == '#':
            continue
        l = line.rstrip().split()
        treatment_vs_control[l[0]] = l[1]


def macs2_callpeak_input(wildcard):
    return {
        'treatment': expand(
            'aligned_data/{treatment}.sorted.rmdup.bam',
            treatment=wildcard.treatment),
        'control': expand(
            'aligned_data/{control}.sorted.rmdup.bam',
            control=treatment_vs_control[wildcard.treatment])
    }

rule macs2_callpeak:
    input:
        unpack(macs2_callpeak_input)
    output:
        treat='macs2_result/{treatment}_peaks.narrowPeak'
    threads: 1
    params:
        name='{treatment}',
        out_dir='macs2_result/',
        gsize=config['gsize']
    container:
        "/public/home/mowp/test/singularity/macs.sif"
    shell:
        '''
macs2 callpeak -t {input.treatment} -c {input.control} -f BAM -g {params.gsize} -n {params.name} -B --SPMR -q 0.01 --outdir {params.out_dir}
        '''

rule bamCompare:
    input:
        unpack(macs2_callpeak_input)
    output:
        'bw_compare/{treatment}.compare.bw'
    threads: 32
    shell:
        '''
bamCompare -b1 {input.treatment} -b2 {input.control} -o {output} --binSize 10 --skipNAs --centerReads --scaleFactorsMethod SES -p {threads}
        '''