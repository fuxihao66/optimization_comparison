import tensorflow as tf
from softmax_batch_loss import *
from softmax_batch_loss_lbfgs import *

if __name__ == '__main__':
    with tf.variable_scope('adam_loss'):
        softmax_batch_loss_comparison(tf.train.AdamOptimizer, 0.01, 'Adam')
    with tf.variable_scope('gd_loss'):
        softmax_batch_loss_comparison(tf.train.GradientDescentOptimizer, 0.1, 'mini_batch GD')
    with tf.variable_scope('lbfgs_loss'):
        softmax_batch_loss_lbfgs_comparison()
