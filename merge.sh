
# $1: SRS ID e.g. SRS075963

echo "join paired end reads"
src/SeqPrep -f $1/$1.denovo_duplicates_marked.trimmed.1.fastq \
  -r ../shotgunClassification/$1/$1.denovo_duplicates_marked.trimmed.2.fastq \
  -1 $1.first.clean.fastq -2 $1.second.clean.fastq -s $1.fastq.gz
gunzip $1.fastq.gz

echo "finished"