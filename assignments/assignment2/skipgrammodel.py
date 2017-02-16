from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import sys
import argparse
from process_data import process_data
import math

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

class SkipGramModel:

    def __init__(self, params):
        self.batch_size = params.batch_size
        self.vocab_size = params.vocab_size
        self.embed_size = params.embed_size
        self.num_samples = params.num_samples
        self.learning_rate = params.learning_rate

    def _create_placeholders(self):
        self.center_words = tf.placeholder(tf.int32, shape=[None], name='center_words')
        self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name='target_words')
        return self.center_words, self.target_words

    def _create_embedding(self):
        self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))
        return self.embed_matrix

    def _create_loss(self):
        # define inference
        self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')
        # construct variables for NCE(noise contrastive estimation) loss
        self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0/math.sqrt(self.embed_size)))
        self.nce_bias = tf.Variable(tf.zeros([self.vocab_size], name='nce_bias'))
        #bias = tf.reshape(self.nce_bias, [-1, 1])
        #self.output = tf.add(tf.matmul(self.nce_weight, tf.transpose(self.embed)), bias) #50000 x 128
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight, biases=self.nce_bias, labels=self.target_words, inputs=self.embed, num_sampled=self.num_samples, num_classes=self.vocab_size), name='loss')
        return self.loss

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        return self.optimizer

def execute(model, batch_gen, dict, index_dict):
    print("create placeholders")
    center_words, target_words = model._create_placeholders()
    print("create embedding")
    embed_matrix = model._create_embedding()
    print("create create loss")
    loss = model._create_loss()
    print("create optimizer")
    optimizer = model._create_optimizer()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        total_loss = 0.0
        writer = tf.summary.FileWriter('./my_graph/skipgram/', sess.graph)
        for index in xrange(NUM_TRAIN_STEPS):
            centers, targets = batch_gen.next()
            loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: centers, target_words:targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()
        '''
        while True:
            input = raw_input('Enter word \n')
            if input == 'exit':
                break
            in_id = dict[input]
            y = sess.run(output, feed_dict={center_words: [in_id]})
            y = tf.reshape(y, [-1])
            values, indices = tf.nn.top_k(y, 5)
            ind = sess.run(indices)
            for i in xrange(indices.shape[0]):
                print('%s' %index_dict[i])
        '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--embed_size', type=int, default=EMBED_SIZE)
    parser.add_argument('--num_samples', type=int, default=EMBED_SIZE)
    parser.add_argument('--skip_window', type=int, default=SKIP_WINDOW)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    print("create model")
    model = SkipGramModel(args)
    print("generating batch...")
    batch_gen, dict, index_dict = process_data(args.vocab_size, args.batch_size, args.skip_window, args.data)
    #batch_gen = process_data(args.vocab_size, args.batch_size, args.skip_window, args.data)
    print("...generated batch")
    execute(model, batch_gen, dict, index_dict)
