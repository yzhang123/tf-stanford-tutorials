""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils
import argparse

# parameters to manage experiments
STYLE = 'guernica'
CONTENT = 'deadpool'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 333
NOISE_RATIO = 0.6 # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

ITERS = 300
LR = 2.0


MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

CKPT_DIR='checkpoints/'
GRAPH_DIR='graphs/'
OUTPUT_DIR='outputs/'
def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    #1/4s sum(F - P)^2
    s = p.shape[1] * p.shape[2] * p.shape[3]
    return tf.reduce_sum(tf.square(f[0] - p[0]), reduction_indices=[0,1,2]) / (4 * s)

def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F 
        Hint: you'll first have to reshape F
    """
    f = tf.reshape(F, [M, N])
    result = tf.matmul(f, f, transpose_a=True)
    return result

def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image, numpy array
        g is the feature representation of the generated image, tensorflow tensor
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    N = a.shape[-1]
    M = a.shape[1] * a.shape[2]
    A = tf.constant(a)
    a_gram = _gram_matrix(A[0], N, M)
    g_gram = _gram_matrix(g[0], N, M)
    loss = tf.reduce_sum(tf.square(a_gram - g_gram), reduction_indices=[0,1]) 
    loss = loss / (4 * N * N * M * M)
    return loss

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    ###############################
    ## TO DO: return total style loss
    loss = tf.zeros_like(E[0])
    for i in range(n_layers):
        loss = loss + E[i]
    loss = loss/5
    return loss
    ###############################

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        ##########################################
        ## TO DO: create total loss. 
        ## Hint: don't forget the content loss and style loss weights
        
        ##########################################
        content_factor = 1.0/20
        total_loss = content_factor * content_loss + (1-content_factor) * style_loss
    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.variable_scope('summary') as scope:
        tf.summary.scalar('content_loss', model['content_loss'])
        tf.summary.scalar('style_loss', model['style_loss'])
        tf.summary.scalar('total_loss', model['total_loss'])
        tf.summary.histogram('content_loss', model['content_loss'])
        tf.summary.histogram('style_loss', model['style_loss'])
        tf.summary.histogram('total_loss', model['total_loss'])
        summary_op = tf.summary.merge_all()
        return summary_op

def train(model, generated_image, initial_image, new):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.
    """

    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ###############################
        ## TO DO: 
        ## 1. initialize your variables
        ## 2. create writer to write your graph
        ###############################
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(GRAPH_DIR + 'lr' + str(LR), sess.graph)
        sess.run(generated_image.assign(initial_image))
        if not new: 
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR +'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        total_loss = 0.0
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TO DO: obtain generated image and loss

                ###############################
                #gen_image, loss, _, summary = sess.run([generated_image, model['total_loss'], model['optimizer'], model['summary_op']])
                gen_image, loss, summary = sess.run([generated_image, model['total_loss'], model['summary_op']])
                total_loss = loss
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = OUTPUT_DIR+CONTENT+'_'+STYLE+'%d.png' % (index)
                utils.save_image(filename, gen_image)

                if not new and (index + 1) % 20 == 0:
                    saver.save(sess, CKPT_DIR+'/style_transfer', index)

def main(args):
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)
    
    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                    input_image, content_image, style_image)
    ###############################
    ## TO DO: create optimizer
    ## model['optimizer'] = ...
    ###############################
    model['optimizer'] = tf.train.AdamOptimizer(LR).minimize(model['total_loss'], global_step=model['global_step'])
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
    train(model, input_image, initial_image, args.new)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default=CONTENT)
    parser.add_argument('--style', type=str,  default=STYLE)
    parser.add_argument('--new', type=bool, default=False)
    args=parser.parse_args()
    CONTENT=args.content
    STYLE=args.style
    STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
    CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
    print('content: %s' % CONTENT_IMAGE)
    print('style: %s' % STYLE_IMAGE)
    print('new: %s' % args.new)


    main(args)
