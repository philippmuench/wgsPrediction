# 2 step read prediction for WGS data

this script will extract high-confidence reads based on unique kmers (e.g. extracted by https://github.com/philippmuench/uniqueK) which are listed in `chromosome_unique.txt` and `plasmid_unique.txt`. Based on these reads a RF will be employed to predict remaining reads based on kmer count. 

