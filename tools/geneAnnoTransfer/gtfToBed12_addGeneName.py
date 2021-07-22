import sh
import pyranges as pr
import pandas as pd
import tempfile
import click

@click.command()
@click.option('-i', 'inPath')
@click.option('--gtfToGenePred', 'gtfToGpPath', default='~/scripts/tools/geneAnnoTransfer/gtfToGenePred', show_default=True)
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
    
    df_bed = pr.read_bed(bedPath, as_df=True)
    df_bed = df_bed.assign(Gene = lambda df:df['Name'].str.split("\|\|").str[1])
    df_geneBed = df_bed.groupby("Gene").agg(
        {
            "Chromosome": lambda x: x.iat[0],
            "Start": "min",
            "End": "max",
            "Strand": lambda x: x.iat[0],
        }
    )
    df_geneBed = df_geneBed.reset_index().pipe(lambda df:df.assign(
        Name=df['Gene'] + '||' + df['Gene'],
        Score=0,
        ThickStart=df['Start'],
        ThickEnd=df['End'],
        ItemRGB=0.0,
        BlockCount=1,
        BlockSizes=(df['End']-df['Start']).astype(str) + ',',
        BlockStarts='0,'
    ))
    df_bed = pd.concat([df_bed, df_geneBed]).sort_values(['Chromosome', 'Start'])
    df_bed = df_bed.reindex(
        columns=[
            "Chromosome",
            "Start",
            "End",
            "Name",
            "Score",
            "Strand",
            "ThickStart",
            "ThickEnd",
            "ItemRGB",
            "BlockCount",
            "BlockSizes",
            "BlockStarts",
        ]
    )
    df_bed['ItemRGB'] = df_bed['ItemRGB'].astype(int)
    df_bed.to_csv(f"{bedPath}.addedGene.bed", sep='\t', index=None, header=None)

main()
