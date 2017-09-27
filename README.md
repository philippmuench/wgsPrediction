# 2 step read prediction for WGS data

## method

this script will extract high-confidence reads based on unique kmers (e.g. extracted by https://github.com/philippmuench/uniqueK) which are listed in `chromosome_unique.txt` and `plasmid_unique.txt`. Based on these reads a RF will be employed to predict remaining reads based on kmer count. 

## installation

```
virtualenv env
source env/bin/activate
git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
cd Multicore-TSNE/
pip install --no-cache-dir .
cd ..
pip install matplotlib pandas numpy Biopython sklearn argparse 
```


## usuage

```
python 2step.py --file test.fastq --is_fastq 
```

## visulaization

tSNE will be created for (1) reads based on kmer counts (2) with highlighed high-confidence reads used for classification (3) colored by classification by RF
