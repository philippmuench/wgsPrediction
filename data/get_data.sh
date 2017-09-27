
# $1: SRS ID e.g. SRS075963

id='SRS075963'

# download 
wget http://downloads.hmpdacc.org/data2/hhs/genome/microbiome/wgs/analysis/hmwgsqc/v2/$id.tar.bz2

# unpack
tar -xjvf $id.tar.bz2 && rm $id.tar.bz2

# join paired end reads
src/SeqPrep -f $1/$1.denovo_duplicates_marked.trimmed.1.fastq \
  -r ../shotgunClassification/$id/$id.denovo_duplicates_marked.trimmed.2.fastq \
  -1 $id.first.clean.fastq -2 $id.second.clean.fastq -s $id.fastq.gz

# extract fasta from fastq
gunzip -c $id.fastq.gz | paste - - - - | cut -f 1,2 | sed 's/^/>/' | tr "\t" "\n" > $id.fasta 
