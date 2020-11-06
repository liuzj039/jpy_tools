import pyranges as pr
import pandas as pd
import pyfastx
import click



@click.command()
@click.option('-f', 'fastaPath', help='genome fa')
@click.option('-b', 'bedpath', help='bed with polyA cluster pos')
@click.option('-s', 'bedSummitPath', help='bed with polyA summit pos')
@click.option('-o', 'fillterPolyASitePath', help='out bed')
def main(fastaPath, bedPath, bedSummitPath, fillterPolyASitePath):
    genomeFa = pyfastx.Fasta(fastaPath)
    polyAClusterBed = pr.read_bed(bedSummitPath, True)
    polyAClusterBed['seq'] = polyAClusterBed.apply(lambda x:genomeFa[x['Chromosome']][x['Start'] - 10 : x['End'] + 10].seq, axis=1)
    polyAClusterBed['seqLength'] = polyAClusterBed['seq'].map(len)
    polyAClusterBed['Ratio'] = polyAClusterBed.apply(lambda x:x['seq'].count('A') if x['Strand'] == '+' else x['seq'].count('T'), axis=1) / polyAClusterBed['seqLength']
    usePolyASite = polyAClusterBed.query('Ratio <= 0.5')['Name']
    polyAClusterRawRangeBed = pr.read_bed(bedPath, True)
    polyAClusterPassedRangeBed = polyAClusterRawRangeBed.query('Name in @usePolyASite')
    polyAClusterPassedRangeBed.to_csv(fillterPolyASitePath, sep='\t', header=None, index=None)

main()