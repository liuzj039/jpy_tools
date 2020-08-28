#!/usr/bin/perl
use strict;
#use warnings;

# A script to convert BED12 into a GTF.
# Input: piped BED12 file.
# Example: cat clark.gtfs.lift_to_hg38/Clark_DataS3.NoncodingTranscriptsAll.lift_to_hg38.bed | perl /users/rg/rjohnson/science/scripts/bed12ToGTF/bed12ToGTF.1.pl

my @line=();

my $trxchr;
my $trxstart;
my $trxend;
my $trxstrand;
my $trxid;
my $trxblank;
my @blocksizes;
my @blockstarts;

my $thisstart;
my $thisend;

my $exontotal;

while (<STDIN>) {
	chomp $_;
	@line=split("\t",$_);

	($trxchr, $trxstart, $trxend, $trxid, $trxblank,$trxstrand)=($line[0],$line[1],$line[2],$line[3],$line[4], $line[5]);

	$trxstart+=1;   # Conversion from BED to GTF!!

	# Print the Transcript Line
	print "$trxchr\tblank\ttranscript\t$trxstart\t$trxend\t.\t$trxstrand\t.\ttranscript_id \"$trxid\;\n";
	
	@blocksizes=split(",", $line[10]);
	@blockstarts=split(",", $line[11]);
	$exontotal=scalar(@blockstarts);
	

	#print "\n @blocksizes hi @blockstarts";

	my $exon_count=0;
	my $rev_exon_count=$exontotal+1;

	for (my $i=0; $i < $exontotal; $i++) {
		$exon_count++;
		$rev_exon_count--;
		

		if ($trxstrand eq "+"){
		$thisstart=$trxstart+$blockstarts[$i];
		$thisend=$trxstart+$blockstarts[$i]+$blocksizes[$i]-1;    # The -1 added empirically after browser inspection
		#Print the Exon lines.
		print "$trxchr\tblank\texon\t$thisstart\t$thisend\t.\t$trxstrand\t.\ttranscript_id \"$trxid\; exon_number $exon_count\;\n";	
		}

		elsif ($trxstrand eq "-"){
		#$thisend=$trxend-$blockstarts[$i];
		#$thisstart=$thisend-$blocksizes[$i]+1;   # The +1 added empirically after browser inspection
		$thisstart=$trxstart+$blockstarts[$i];
		$thisend=$trxstart+$blockstarts[$i]+$blocksizes[$i]-1;    # The -1 added empirically after browser inspection
		#Print the Exon lines.
		print "$trxchr\tblank\texon\t$thisstart\t$thisend\t.\t$trxstrand\t.\ttranscript_id \"$trxid\; exon_number $rev_exon_count\;\n";	
		}
}

}

