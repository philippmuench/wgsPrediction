# 2 step read prediction for WGS data

## method

this script will extract high-confidence reads based on unique kmers (e.g. extracted by https://github.com/philippmuench/uniqueK) which are listed in `database/chromosome_unique.txt` and `database/plasmid_unique.txt`. Based on these reads a random forest will be employed to predict remaining reads based on kmer count. 

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

this will output `*.chromosomes.fastq` and `*.plasmids.fastq`

## evaluation

1. download HMP sample reads `data/get_data.sh`
2. run `python 2step.py --file data/SRS075963.fastq --is_fastq`
3. run assembly and assembly evaluation `tool/evaluate_assembly.sh` this tool will also perform a standard assembly without 2step prediction as well as plasmidSPADES assembly

## visulaization

if you use the `--tnse` argument, tSNE plots will be created for (1) reads based on kmer counts (2) with highlighed high-confidence reads used for classification (3) colored by classification by RF (not fully implemented)
