# %%
import jpy_tools.parseSnake as jps

# %%
snakeFile = jps.SnakeMakeFile()
# %%
snakeHeader = jps.SnakeHeader(
    snakeFile,
    "/public/home/liuzj/scripts/pipeline/combineSnuupyAndSiceloreResult/snakemake/config.yaml",
)
snakeHeader.addFeature('combineScriptDir')
snakeHeader.generateContent()
# %%
combineResult = jps.SnakeRule(snakeFile, "combineResult", 1, 1)
combineResult.setInput(b=["siceloreBam", "siceloreFasta", "snuupyFea", "snuupyFasta"])
combineResult.setOutput(a=["combineFasta.fa", "combineFea.feather"])
combineResult.setShell(
    """
cd {combineScriptDir}
python ./mergeSnuupyAndSiceloreResults.py --si-bam {input.siceloreBam} --si-fasta {input.siceloreFasta} \
--sn-fasta {input.snuupyFasta} --sn-fea {input.snuupyFea} --out-fasta {output.combineFasta} \
--out-fea {output.combineFea}
"""
)
combineResult.generateContent()
# %%
minimapMappingPolished = jps.SnakeRule(snakeFile, "minimapMappingPolished", 2, 56)
minimapMappingPolished.setInput(a=["combineFasta"])
minimapMappingPolished.setOutput(a=["polishedMappingResult.bam"])
minimapMappingPolished.setParams(b=["minimap2Path", "genomeFa", "geneAnnoBed"])
minimapMappingPolished.setShell(
    """
{params.minimap2Path} -ax splice --secondary=no -uf --MD --sam-hit-only -t {threads} \
--junc-bed {params.geneAnnoBed} {params.genomeFa} {input.combineFasta} |\
samtools sort - -o {output.polishedMappingResult} && samtools index {output.polishedMappingResult}
"""
)
minimapMappingPolished.generateContent()
# %%
addGeneName = jps.SnakeRule(snakeFile, "addGeneName", 3, 2)
addGeneName.setInput(a=["polishedMappingResult"])
addGeneName.setOutput(
    a=["polishedReadsAddGNPickle.pickle", "polishedReadsAddGNBam.bam"]
)
addGeneName.setParams(b=["geneAnnoBed", "bedtoolsPath"])
addGeneName.setShell(
    """
python ./snuupy/snuupy.py addGeneName -i {input.polishedMappingResult} --bed {params.geneAnnoBed} --out-pickle {output.polishedReadsAddGNPickle} --out-bam {output.polishedReadsAddGNBam} --bedtools {params.bedtoolsPath}
"""
)
addGeneName.generateContent()
# %%
getSpliceInfo = jps.SnakeRule(snakeFile, "getSpliceInfo", 4, 2)
getSpliceInfo.setInput(a=["polishedMappingResult", "polishedReadsAddGNPickle"])
getSpliceInfo.setOutput(a=["splicingInfo.tsv"])
getSpliceInfo.setParams(b=["bedtoolsPath", "geneAnnoRepreBed"])
getSpliceInfo.setShell(
    """
python ./snuupy/snuupy.py getSpliceInfo -i {input.polishedMappingResult} -b {params.geneAnnoRepreBed} -o {output.splicingInfo} -g {input.polishedReadsAddGNPickle} --bedtools {params.bedtoolsPath}
"""
)
getSpliceInfo.generateContent()
# %%
addPolyATag = jps.SnakeRule(snakeFile, "addPolyATag", 5, 56)
addPolyATag.setInput(
    a=["combineFea", "polishedReadsAddGNBam"],
    b=["rawNanoporeFa", "nanoporeWorkspace", "nanoporeSeqSummary", "geneAnnoBed"],
)
addPolyATag.setOutput(a=["polishedReadsAddGNPABam.bam"])
addPolyATag.setParams(
    a=["polyACallerTemp/"], b=["genomeFa", "geneAnnoBed", "minimap2Path"]
)
addPolyATag.setShell(
    """
    python ./snuupy/snuupy.py addPolyATag --in-fasta {input.rawNanoporeFa} \
    --genome {params.genomeFa} -t {threads} --in-f5-workspace {input.nanoporeWorkspace} \
    --in-f5-summary {input.nanoporeSeqSummary} --bed {params.geneAnnoBed} \
    --tempDir {params.polyACallerTemp} --feather {input.combineFea} \
    --in-bam {input.polishedReadsAddGNBam} --out-bam {output.polishedReadsAddGNPABam} \
    --minimap {params.minimap2Path}
"""
)
addPolyATag.generateContent()
# %%
polyAClusterDetected = jps.SnakeRule(snakeFile, "polyAClusterDetected", 6, 56)
polyAClusterDetected.setInput(a=["polishedReadsAddGNPABam"])
polyAClusterDetected.setOutput(a=["polyAClusterDetectedFinished.empty"])
polyAClusterDetected.setParams(a=["polyACluster/"], b=["geneNot12Bed", "genomeFa"])
polyAClusterDetected.setShell(
    """
python ./snuupy/snuupy.py polyAClusterDetected --infile {input.polishedReadsAddGNPABam} --gene-bed {params.geneNot12Bed} --out-dir {params.polyACluster} -t {threads} --fasta {params.genomeFa} && touch {output.polyAClusterDetectedFinished}
"""
)
polyAClusterDetected.generateContent()
# %%
generateMtx = jps.SnakeRule(snakeFile, "generateMtx", 7, 2)
generateMtx.setInput(
    a=["splicingInfo", "polishedReadsAddGNPABam", "polyAClusterDetectedFinished"]
)
generateMtx.setOutput(a=["generateMtxFinished.empty"])
generateMtx.setParams(
    a=["IlluminaMultiMat/", "NanoporeMultiMat/"],
    b=["usedIntron", "cellRangerH5"],
    d=dict(
        step6=['polyACluster/polya_cluster.filtered.bed']
    ),
)
generateMtx.setShell(
    """
python ./snuupy/snuupy.py generateMtx -i {input.splicingInfo} --in-illumina {params.cellRangerH5} \
--apa-pac {params.polya_clusterFiltered} --apa-bam {input.polishedReadsAddGNPABam} \
--ir --ir-list {params.usedIntron} --out-nanopore {params.NanoporeMultiMat} \
--out-illumina {params.IlluminaMultiMat} && touch {output.generateMtxFinished}
"""
)
generateMtx.generateContent()
# %%
snakeAll = jps.SnakeAll(snakeFile)
snakeAll.generateContent(generateMtxFinished=0)
# %%
snakeFile.generateContent("/public/home/liuzj/scripts/pipeline/combineSnuupyAndSiceloreResult/snakemake/snakefile")
# %%

# %%
