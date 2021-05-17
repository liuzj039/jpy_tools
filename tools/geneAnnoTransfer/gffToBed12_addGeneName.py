import sh
import pyranges as pr
import tempfile
import click

@click.command()
@click.option('-i', 'inPath')
@click.option('--gtfToGenePred', 'gtfToGenePred', default='~/scripts/tools/geneAnnoTransfer/gtfToGenePred', show_default=True)
@click.option('--genePredToBed', 'gpToBedPath', default='~/scripts/tools/geneAnnoTransfer/genePredToBed', show_default=True)
def main(inPath, gtfToGpPath, gpToBedPath):
    """
    transform gtf to bed12
    """
    inPr = pr.read_gtf(inPath)
    inPr.transcript_id = inPr.transcript_id  + '||' + inPr.gene_id
    with tempfile.TemporaryDirectory('/') as tempDir:
        inName = inPath.split('/')[-1]
        changeTrsGtfPath = f"{tempDir}{inName}.changeTrsId.gtf"
        gpPath = f"{tempDir}{inName}.changeTrsId.gp"
        bedPath = f"{inPath}.changeTrsId.bed"

        inPr.to_gtf(changeTrsGtfPath)
        sh.Command(gtfToGpPath)(changeTrsGtfPath, gpPath)
        sh.Command(gpToBedPath)(gpPath, bedPath)

main()