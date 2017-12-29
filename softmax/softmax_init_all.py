import tensorflow as tf
from softmax_init_loss import *

if __name__ == '__main__':
    
    with tf.variable_scope('adam_loss'):
        softmax_init_loss_comparison(100, tf.train.AdamOptimizer, 'Adam')
    with tf.variable_scope('gd_loss'):
        softmax_init_loss_comparison(100, tf.train.GradientDescentOptimizer, 'mini_batch GD')
    with tf.variable_scope('sgd_loss'):
        softmax_init_loss_comparison(1, tf.train.GradientDescentOptimizer, 'SGD')