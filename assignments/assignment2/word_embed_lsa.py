import csv
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from collections import Counter

# choose most common 10.000 words.
# build co-occurrence matrix window size of 3 centered at current word
# calc SVD on co-occurrence matrix using tf.svd
# calc word embeddings

def main(args):
    data = __load_data(args.data) #2D matrix
    print(data.shape)
    
    n_words = 10000
    UNK = 0
    k = 100
    cooc = np.zeros([n_words, n_words], dtype=np.float32)
    w2id = {}

    counts = Counter(data)
    c = counts.most_common(n_words - 1)
    print("%d different words" % len(c))

    idx = 1;
    for  k, _ in c:
        w2id[k] = idx
        idx = idx + 1

    for i in xrange(len(data) - 1):
        n = w2id.get(data[i + 1], UNK)
        c = w2id.get(data[i], UNK)
        cooc[n][c] = cooc[n][c] + 1
        cooc[c][n] = cooc[c][n] + 1

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    embedding = getembedding(sess, cooc, k)
    print(embedding)



def getembedding(sess, cooc, k):
    s = tf.svd(cooc)
    sess.run(s)
    k = min(s[1].get_shape()[1].value, s[2].get_shape()[0].value)
    embedding  = tf.matmul(s[2][:k], tf.transpose(cooc)) # embedding vectors in columns
    return sess.run(embedding)




	
def __load_data(file):
	data = []
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter=" ")
		for line in reader:
			data = np.array(line)
	return data




def __get_parser():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("data", type=str)
	return parser

if __name__ == "__main__":
	args = __get_parser().parse_args()
	main(args)