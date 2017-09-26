import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, csv
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import array
from Bio.Seq import Seq
from Bio import Entrez, SeqIO
from MulticoreTSNE import MulticoreTSNE as TSNE

def countKmerMatch(fas, kmer_list, outfile):
	match = []
	with open(kmer_list,'r') as fin:
		with open(outfile, "w") as output_handle:
			lines =  fin.read().splitlines() 
			for record in SeqIO.parse(fas, "fasta"):
				read = record.seq
				num = 0
				for plasmid_sequence in lines:
					sequence = Seq(plasmid_sequence)
					num = num + int(read.count(sequence))
				if num > 0:
					match.append('1')
					# write to fasta file
					SeqIO.write(record, output_handle, "fasta")
				else:
					match.append('0')
	return match

def shapeKmerMatch2(fas, kmer_list1, kmer_list2):
	match = []
	with open(kmer_list2,'r') as fin2:
		lines_list2 =  fin2.read().splitlines() 
		with open(kmer_list1,'r') as fin:
			lines_list1 =  fin.read().splitlines() 
			for record in SeqIO.parse(fas, "fasta"):
				read = record.seq
				num_list1 = 0
				num_list2 = 0
				for plasmid_sequence in lines_list1:
					sequence = Seq(plasmid_sequence)
					num_list1 = num_list1 + int(read.count(sequence))
				for plasmid_sequence in lines_list2:
					sequence = Seq(plasmid_sequence)
					num_list2 = num_list2 + int(read.count(sequence))
				if num_list1 > 0:
					match.append(',')
				elif num_list2 > 0:
					match.append('D')
				else:
					match.append('P')
		return match

def creatematrix(kmer):
	"""Generates feature matrix from raw data"""
	kmer = pd.read_csv(kmer, sep=",", header=None, engine='python')
	X = kmer.iloc[:, :-1]
	return X

def extractkmers(data):
	s = ""
	cmd = ('src/fasta2kmers2 -i ', data, ' -f ', data, '.kmer -j 5 -k 5 -s 0 -r 1 -R 1 -n 1')
	os.system(s.join( cmd ))
	fname =  data +  ".kmer"
	fname2 = data + ".kmer.csv"
	with open(fname, "r") as inp, open(fname2 , "w") as out:
		w = csv.writer(out, delimiter=",")
		w.writerows(x for x in csv.reader(inp, delimiter="\t"))

def print_tsne(X, perplexity = 10, colors=None,  cmap='hsv', marker=None):
	X_tsne = TSNE(n_jobs = 4, n_components = 2, perplexity=perplexity, init='pca', verbose=1).fit_transform(X)
	plt.figure()
	plt.margins(0)
	plt.axis('off')
	if len(marker) > 1:
		unique_markers = set(marker) 
		for um in unique_markers:
			mask = marker == um 
			plt.scatter(X_tsne[mask, 0],
				X_tsne[mask, 1],
				marker=um,
				cmap=cmap,
				edgecolor='', # don't use edges 
				lw=0,  # don't use edges
				s=1
				)
	else:
		plt.scatter( X_tsne[:, 0], X_tsne[:, 1], 
			marker=',',
			alpha = 1, 
			c=colors,
			label='test',
			cmap=cmap,
			edgecolor='', # don't use edges 
			lw=0,  # don't use edges
			s=0.5) # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
	plt.title('tSNE perplexity:' + str(perplexity))
#	plt.colorbar()
	plt.savefig('tsne' + str(perplexity) + '.pdf')
	return(X_tsne)

def getGC(seq):
	# get GC content of DNA string
	return(str(int(round((sum([1.0 for nucl in seq if nucl in ['G', 'C']]) / len(seq)) * 100))))

def extractGC(fas):
	# iterate over fasta file and print GC content
	gc_vals = []
	for record in SeqIO.parse(fas, "fasta"):
		gc_vals.append(getGC(str(record.seq)))
	gc_vals = map(int, gc_vals)
	gc = array(gc_vals)
	return gc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', action='store', dest='file', help='path to input fasta file', default='data')
	parser.add_argument('--read_num', action='store', dest='read_num', help='number of reads for subset', default='1000')
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')

	args = parser.parse_args()

	print('extract kmer profile')
	extractkmers(args.file)

	print('extract GC content')
	gc = extractGC(args.file)

	# get counts to kmer table
	shapes = shapeKmerMatch2(args.file, 'plasmid_unique.txt', 'chromosome_unique.txt')
	is_plasmid = countKmerMatch(args.file, 'plasmid_unique.txt', 'found_plasmids.fasta')
	is_chromosome = countKmerMatch(args.file, 'chromosome_unique.txt', 'found_chromosomes.fasta')

	# print input data
	print('generate matrix')
	X = creatematrix(args.file + ".kmer.csv")
	X = X.as_matrix()

	# resaling
	print('scale data')
	sc = StandardScaler()
	X_sc = sc.fit_transform(X)

	print('crate tsne')
	print_tsne(X_sc, perplexity = 25, colors = gc, cmap='viridis', marker=np.array(shapes))
	#print_tsne(X_sc, perplexity = 5, colors = gc, cmap='viridis')
#	print_tsne(X_sc, perplexity = 25, colors = gc, cmap='viridis')
	#print_tsne(X_sc, perplexity = 50, colors = gc, cmap='viridis')
