from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import sys
import argparse



class SkipGramModel:

    def __init__(self, params):
        self.batch_size = params.batch_size
        self.vocab_size = params.vocab_size
        self.embed_size = params.embed_size
        self.num_samples = params.num_samples
        self.learning_rate = params.learning_rate

    def _create_placeholders(self):
        center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center words')
        target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target words')


    def _create_embedding(self):
        embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))


    def _create_loss(self):
        # define inference
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        # construct variables for NCE(noise contrastive estimation) loss
        nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size]))
        nce_bias = tf.Variable(tf.zeros([self.vocab_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=self.num_samples, num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

def execute(model):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        average_loss = 0.0
        for index in xrange()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--embed_size', type=int)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()
    
    model = SkipGramModel(args)
    execute(model)
