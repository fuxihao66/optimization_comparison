import tensorflow as tf
from softmax_lr_loss import *
if __name__ == '__main__':

    softmax_lr_loss_comparison(100, tf.train.AdamOptimizer, 'Adam')
    softmax_lr_loss_comparison(100, tf.train.GradientDescentOptimizer, 'mini_batch GD')
    softmax_lr_loss_comparison(1, tf.train.GradientDescentOptimizer, 'SGD')