import tensorflow as tf
slim = tf.contrib.slim
import time
import numpy as np

def DenseNet(x_image, numpix_out, arch='conv', block_size = 12, growth_rate = 12, final_depth = 1):
    '''
    DenseNet takes an image tensor x_image as an input.  It returns a tensor of size
    [m,numpix_out,numpix_out,1], corresponding to the predicted image of the background
    source.
    
    Many of the helper functions here (those that build the network) were taken from 
    the DenseNet implementation by the github user LaurentMazare.
    '''

    layers = block_size
    k = growth_rate
    im_dim = x_image.shape[0]
    is_training = tf.placeholder("bool", shape=[])
    keep_prob = tf.placeholder(tf.float32)
        
    with tf.variable_scope('DenseNet'):
    
        #First conv layer, with 2k feature maps
        X = conv2d(x_image, 1, 2*k,3)
        
        # First dense block, with input features from previous conv2d layer
        X, features = block(X, layers, 2*k, k, is_training, keep_prob)

        # First transition layer, preserving the number of features
        X = batch_activ_conv(X, features, features, 1, is_training, keep_prob)
        X = tf.nn.avg_pool(X, [1,2,2,1], [1,2,2,1], 'VALID')

        # Second dense block, with input features equal to output features of previous block
        X, features = block(X, layers, features, k, is_training, keep_prob)

        # Second transition layer, preserving the number of features
        X = batch_activ_conv(X, features, features, 1, is_training, keep_prob)
        X = tf.nn.avg_pool(X, [1,2,2,1], [1,2,2,1], 'VALID')

        # Third dense block, with input features equal to output features of previous block
        X, features = block(X, layers, features, k, is_training, keep_prob)

        # We perform a batch normalization before our final processing
        X = tf.contrib.layers.batch_norm(X, scale=True, is_training=is_training, updates_collections=None)
        
        # If 'conv' architecture, we take a final 1x1 conv layer with one filter, and use a relu activation 
        if arch == 'conv':            
            X = conv2d(X, features, final_depth, 1)
            X = tf.nn.relu(X)

        # If 'FC16' architecture, we take a 1x1 conv layer with 16 filters, which we flatten and feed to a 
        # fully connected layer with numpix_out*numpix_out units, with relu activation
        elif arch == 'FC16':
            X = conv2d(X, features, 16, 1)
            X = tf.contrib.layers.batch_norm(X, scale=True, is_training = is_training, updates_collections=None)
	    X = tf.nn.relu(X)
            X = tf.contrib.layers.flatten(X)
            X = tf.contrib.layers.fully_connected(X, numpix_out*numpix_out, activation_fn=tf.nn.relu)
            
            X = tf.reshape(X,[-1,numpix_out,numpix_out,1])
        else:  # no choice or bad specified.  Use conv, but warn the user
            print "your choice of network end is bad, but we'll take care of that for you\n"
            X = conv2d(X,features,final_depth,1)
            X = tf.nn.relu(X)
            
    return X , is_training , keep_prob

    
    

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv2d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in xrange(layers):
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    current = tf.concat((current, tmp), axis=3)
    features += growth
  return current, features
