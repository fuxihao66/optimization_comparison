import tensorflow as tf
from cnn_batch_loss import *

if __name__ == '__main__':

    with tf.variable_scope('adam_loss'):
        cnn_batch_loss_comparison(tf.train.AdamOptimizer, 0.01, 'Adam')
    with tf.variable_scope('gd_loss'):
        cnn_batch_loss_comparison(tf.train.GradientDescentOptimizer, 0.01, 'mini_batch GD')