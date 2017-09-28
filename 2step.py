import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, csv, sys, re
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import array
import progressbar
from Bio.Seq import Seq
from Bio import SeqIO
from StringIO import StringIO
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def fastq2fasta(fastq_path, fasta_path):
	with open(fasta_path, "w") as fasta, open(fastq_path, "r") as fastq:
		for record in SeqIO.parse(fastq, "fastq"):
			SeqIO.write(record, fasta, "fasta")

def sequence_cleaner(fasta_file, min_length=0, por_n=100):
	# Create our hash table to add the sequences
	sequences={}
	for seq_record in SeqIO.parse(fasta_file, "fasta"):
		sequence = str(seq_record.seq).upper()
		if (len(sequence) >= min_length and 
			(float(sequence.count("N"))/float(len(sequence)))*100 <= por_n):
			if sequence not in sequences:
				sequences[sequence] = seq_record.id
			else:
				sequences[sequence] += "_" + seq_record.id
	with open("clear_" + fasta_file, "w+") as output_file:
		# Just read the hash table and write on the file as a fasta format
		for sequence in sequences:
			output_file.write(">" + sequences[sequence] + "\n" + sequence + "\n")

def countKmerMatchFast(fas, kmer_list, outfile):
	total_count = 0
	with open(kmer_list,'r') as fin:
		lines =  fin.read().splitlines()
		for marker_sequence in lines:
			print("check " + marker_sequence)
			output_file = open(outfile, 'ab')
			input_file = open(fas, "r")
			matches = []
			current_line = ""
			for line in input_file:
				previous_line = current_line
				current_line = line
				if re.search(marker_sequence, line):
					total_count += 1
					#print previous_line,
					output_file.write(">" + previous_line) # write fasta header
					output_file.write(current_line) # write sequence
					matches.append(previous_line)
	output_file.close()
	return total_count

def countKmerMatch(fas, kmer_list, outfile, is_fastq):
	match = []
	total_count = 0
	num_found_markers = 0
	handle = StringIO("")
	num_elem = SeqIO.convert(fas, "fastq", handle, "fasta") # there must be an easyer way to get the number of sequences?
	bar = progressbar.ProgressBar(redirect_stdout=True, max_value=num_elem)
	with open(kmer_list,'r') as fin:
		with open(outfile + '.fasta', "w") as output_fasta_handle:
			lines =  fin.read().splitlines()
			if is_fastq:
				with open(outfile + '.fastq', "w") as output_fastq_handle:
					i = 0
					for record in SeqIO.parse(fas, "fastq"):
						read = record.seq
						i = i + 1
						num = 0
						bar.update(i)
						for plasmid_sequence in lines:
							sequence = Seq(plasmid_sequence)
							num = num + int(read.count(sequence))
						if num > 0:
							print("marker found at read number %d" %(i))
							num_found_markers = num_found_markers + 1
							match.append('1')
							# write to fasta file
							SeqIO.write(record, output_fastq_handle, "fastq") 
							SeqIO.write(record, output_fasta_handle, "fasta")
							total_count = total_count + 1
						else:
							match.append('0')
			else:
				for record in SeqIO.parse(fas, "fasta"):
					read = record.seq
					num = 0
					for plasmid_sequence in lines:
						sequence = Seq(plasmid_sequence)
						num = num + int(read.count(sequence))
					if num > 0:
						match.append('1')
						# write to fasta file
						SeqIO.write(record, output_fasta_handle, "fasta")
						total_count = total_count + 1
					else:
						match.append('0')
	return match, total_count

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
		match = array(match)
		return match

def creatematrix(kmer):
	"""Generates feature matrix from raw data"""
	kmer = pd.read_csv(kmer, sep=",", header=None, engine='python')
	X = kmer.iloc[:, :-1]
	return X

def extractkmers(data):
	s = ""
	cmd = ('src/fasta2kmers2 -i ', data, ' -f ', data, '.kmer -j 4 -k 4 -s 0 -r 1 -R 1 -n 1')
	os.system(s.join( cmd ))
	fname =  data +  ".kmer"
	fname2 = data + ".kmer.csv"
	with open(fname, "r") as inp, open(fname2 , "w") as out:
		w = csv.writer(out, delimiter=",")
		w.writerows(x for x in csv.reader(inp, delimiter="\t"))

def tsne_w_marker(X_tsne, marker, title, colors=None,  cmap='hsv'):
	plt.figure()
	plt.margins(0)
	plt.axis('off')
	unique_markers = set(marker) 
	for um in unique_markers:
		mask = marker == um 
		plt.scatter(X_tsne[mask, 0],
			X_tsne[mask, 1],
			marker=um,
			cmap=cmap,
			edgecolor='', # don't use edges 
			lw=0,  # don't use edges
			s=1)
	plt.title('tSNE perplexity:' + str(title))
	plt.savefig('tsne_class_' + str(title) + '.pdf')
	return(plt)

def tsne_wo_marker(X_tsne, title, colors=None,  cmap='hsv'):
	plt.figure()
	plt.margins(0)
	plt.axis('off')
	plt.scatter( X_tsne[:, 0], X_tsne[:, 1], 
			marker=',',
			alpha = 1, 
			c=colors,
			label='test',
			cmap=cmap,
			edgecolor='', # don't use edges 
			lw=0,  # don't use edges
			s=0.5) # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
	plt.title('tSNE perplexity:' + str(title))
	plt.savefig('tsne_gc_' + str(title) + '.pdf')
	return(plt)

def print_tsne(X, perplexity = 10, colors=None,  cmap='hsv', marker=None):
	sc = StandardScaler()
	X_sc = sc.fit_transform(X)
	X_tsne = TSNE(n_jobs = 4, n_components = 2, perplexity=perplexity, 
		init='pca', verbose=1).fit_transform(X_sc)
	tsne_wo_marker(X_tsne, title='test1', colors=colors, cmap=cmap)
	tsne_w_marker(X_tsne, marker=marker, title='test1', colors=colors, cmap=cmap)
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

def createKmerMatrix(kmer):
	X = pd.read_csv(kmer, sep=",", header=None)
	X = X.iloc[:, :-1]
	return X

def creatematrix(kmer_pos, kmer_neg):
	"""Generates feature matrix from raw kmer data"""
	# load kmer matrix
	kmer_pos = pd.read_csv(kmer_pos, sep=",", header=None)
	kmer_neg = pd.read_csv(kmer_neg, sep=",", header=None)
	kmer_pos = kmer_pos.iloc[:, :-1]
	kmer_neg = kmer_neg.iloc[:, :-1]
	X = kmer_pos.append(kmer_neg)
	lab = np.array([1, 0])
	y = np.repeat(lab, [kmer_pos.shape[0], kmer_neg.shape[0]], axis=0)
	return X, y

def runRF(X, y, X_val, y_val):
	clf = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, class_weight='balanced')
	#clf = RandomForestClassifier(n_estimators = 2000,
#		max_depth=param_dist['clf__max_depth'], 
#		max_features=param_dist['clf__max_features'],
#		min_samples_split=param_dist['clf__min_samples_split'],
#		min_samples_leaf=param_dist['clf__min_samples_leaf'],
#		criterion=param_dist['clf__criterion'],
#		n_jobs = -1)
	pipe = Pipeline([['sc', MaxAbsScaler()],['clf', clf]])
	scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc' : 'roc_auc', 'average_precision' : 'average_precision', 'f1' : 'f1', 'f1_micro' : 'f1_micro','f1_macro' : 'f1_macro','f1_weighted' : 'f1_weighted'}
	scores = cross_validate(pipe, X, y, scoring=scoring, cv=3, return_train_score=False)
	label = 'final model'
	print("\ntest_accuracy: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_accuracy"].mean(), scores["test_accuracy"].std(), label))
	print("test_recall: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_recall"].mean(), scores["test_recall"].std(), label))
	print("test_precision: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_precision"].mean(), scores["test_precision"].std(), label))
	print("test_roc_auc: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_roc_auc"].mean(), scores["test_roc_auc"].std(), label))
	print("test_average_precision: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_average_precision"].mean(), scores["test_average_precision"].std(), label))
	print("test_f1: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_f1"].mean(), scores["test_f1"].std(), label))
	print("test_f1_micro: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_f1_micro"].mean(), scores["test_f1_micro"].std(), label))
	print("test_f1_macro: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_f1_macro"].mean(), scores["test_f1_macro"].std(), label))
	print("test_f1_weighted: %0.2f (+/- %0.2f) [%s]" %
		(scores["test_f1_weighted"].mean(), scores["test_f1_weighted"].std(), label))
	fitted = pipe.fit(X, y)
	filename = 'model/rf.pkl'
	with open(filename, 'wb') as fid:
		joblib.dump(fitted, fid, compress=9)
	return fitted

def subsetFastaBasedOnPrediction(file, predictions):
	"""seperates plasmid and chromosomal fragments based on prediction"""
	i = 0
	pla_num = 0
	chr_num = 0
	with open('test.fastq' + ".plasmids.fastq", "w") as output_fastq_plasmid_handle:
		with open('test.fastq' + ".chromosomes.fastq", "w") as output_fastq_chromosomes_handle:	
			for seq_record in SeqIO.parse('test.fastq', "fastq"):
				if (predictions[i] == 1):
					SeqIO.write(seq_record, output_fastq_plasmid_handle, "fastq")
					pla_num += 1
				else:
					SeqIO.write(seq_record, output_fastq_chromosomes_handle, "fastq")
					chr_num += 1
				i += 1
	return pla_num, chr_num

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', action='store', dest='file', help='path to input fasta file', default='data')
	parser.add_argument('--read_num', action='store', dest='read_num', help='number of reads for subset', default='1000')
	parser.add_argument('-m', '--model_param_file', action='store', dest='model_param',
						help='path to file where .pkl models is located')
	parser.add_argument('--tsne', action='store_true', help='output tsne plot')
	parser.add_argument('--assemble', action='store_true', help='use SPADES to assemble plasmid and chromosome reads independently')
	parser.add_argument('--is_fastq', action='store_true', help='input is fastq file')
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	args = parser.parse_args()

	""" predict high-confidence reads """
	print('searching reads for plasmid markers')
	#is_plasmid, plasmid_matches = countKmerMatch(args.file, 'database/set_a_specific_20_100.txt', 'high_confidence_plasmid_reads', args.is_fastq)
	plasmid_matches = countKmerMatchFast(args.file, 'database/set_a_specific_20_100.txt', 'high_confidence_plasmid_reads')
	chromosome_matches = countKmerMatchFast(args.file, 'database/set_b_specific_20_100.txt', 'high_confidence_chromosome_reads')
	print('searching reads for non-plasmid markers')
	#is_chromosome, chromosome_matches = countKmerMatch(args.file, 'database/set_b_specific_20_100.txt', 'high_confidence_chromosome_reads', args.is_fastq)

	print("number of high-confidence plasmid reads found: %d" %(plasmid_matches))
	print("number of high-confidence chromsome reads found: %d" %(chromosome_matches))

	if plasmid_matches > 0 and chromosome_matches > 0:
		print('extract kmer information of high-confidence reads')
		extractkmers('high_confidence_plasmid_reads.fasta')
		extractkmers('high_confidence_chromosome_reads.fasta')

		print('build learning matrix')
		X, y = creatematrix('high_confidence_plasmid_reads.fasta.kmer.csv', 'high_confidence_chromosome_reads.fasta.kmer.csv')
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

		""" build RF model """
		# create output folder
		if not os.path.exists('model'):
			os.makedirs('model')

		# get read length for classificator
		#	read_length = getReadlength(args.file)

		# load RF parameters
		print('setting up RF model')
		#param_dist = joblib.load(args.model_param)
		# build RF
		print('build RF model')
		estimator = runRF(X_train, y_train, X_test, y_test)

		""" predict remaining reads """
		print('extract kmer profile')
		if args.is_fastq:
			fastq2fasta(args.file, args.file + '.fasta')
			extractkmers(args.file + '.fasta')
		else:
			extractkmers(args.file)
		if args.is_fastq:	
			X_pred = createKmerMatrix(args.file + '.fasta.kmer.csv')
		else:
			X_pred = createKmerMatrix(args.file + '.kmer.csv')
		print('predict reads')
		predictions = estimator.predict(X_pred)
		class_proba = estimator.predict_proba(X_pred)
		probability = np.amax(class_proba, axis=1) # max probability over two classes
		# split reads

		if args.is_fastq:
			pla_num, chr_num = subsetFastaBasedOnPrediction(args.file, predictions)
		else:
			print("not implemented for fasta output, please use fastq input file")
		print("number of plasmid reads found: %d" %(pla_num))
		print("number of chromsome reads found: %d" %(chr_num))
	else:
		print("plasmid prediction cannot be executed")

	""" print tSNE on subset of data """
	if args.tsne:
		print('generate plot colors')
		gc = extractGC(args.file)
		print('generate matrix')
		X_mat = X.as_matrix()
		print('get plot shape')
		shapes = shapeKmerMatch2(args.file, 'database/plasmid_unique.txt', 'database/chromosome_unique.txt')
		# subset data
		if (args.read_num > X_mat.shape[0]):
			print("subset size is larger than number of reads. Subset size was reduced to number of reads")
			args.read_num = X_mat.shape[0]
		idx = np.random.choice(X_mat.shape[0], args.read_num, replace=False) # roll the dice
		X_subset = X_mat[idx, :]
		gc_subset = gc[idx]
		shape_subset = shapes[idx]
		print('crate tSNE plots')
		print_tsne(X_subset, perplexity = 10, colors = gc_subset, cmap='viridis', marker=shape_subset)
#		print_tsne(X_sc, perplexity = 5, colors = gc, cmap='viridis')
#		print_tsne(X_sc, perplexity = 25, colors = gc, cmap='viridis')
#		print_tsne(X_sc, perplexity = 50, colors = gc, cmap='viridis')
