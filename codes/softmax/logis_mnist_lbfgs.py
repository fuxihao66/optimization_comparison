from __future__ import print_function
import numpy as np
import tensorflow as tf
from plot_text import *
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
def get_loss(loss_evaled):
    return loss_evaled





mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.losses.softmax_cross_entropy(
            onehot_labels=y, logits=pred)
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


iterations = 20
optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                cost,
                method='L-BFGS-B',
                options={'maxiter': iterations})
                # var_to_bounds={b: (-2, 7), W:(-1,6)})

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
global_step = 0
x_axis = []
y_axis = []
# Start training
with tf.Session() as sess:

    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            optimizer.minimize(sess, feed_dict={x: batch_xs,y: batch_ys},
                                    loss_callback=get_loss, fetches=[cost])
            c = cost.eval({x: batch_xs,y: batch_ys})
            global_step+=1
            x_axis.append(global_step)
            y_axis.append(c)
            avg_cost += c / total_batch
        
        # if (epoch+1) % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("Optimization Finished!")

x_axis = np.array(x_axis)
y_axis = np.array(y_axis)




