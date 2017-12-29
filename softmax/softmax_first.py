from __future__ import print_function
import numpy as np
import tensorflow as tf
from plot_test import *
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100


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

init = tf.global_variables_initializer()
global_step = 0


x_axis = [0]
y_lbfgs_axis = []
y_lbfgs_axis1 = []
with tf.Session() as sess:

    sess.run(init)
    y_lbfgs_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_lbfgs_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            optimizer.minimize(sess, feed_dict={x: batch_xs,y: batch_ys})
            c = cost.eval({x: batch_xs,y: batch_ys})
            avg_cost += c / total_batch
        global_step+=1
        x_axis.append(global_step)
        y_lbfgs_axis.append(avg_cost)
        y_lbfgs_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("LBFGS Optimization Finished!")


y_adam_axis = []
y_adam_axis1 = []
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    y_adam_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_adam_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
            avg_cost += c / total_batch

        y_adam_axis.append(avg_cost)
        y_adam_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        # print("Epoch{}, ".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Adam Optimization Finished!")

y_batch_gd_axis = []
y_batch_gd_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
global_step = 0
with tf.Session() as sess:
    sess.run(init)
    y_batch_gd_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_batch_gd_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
            avg_cost += c / total_batch
                
        y_batch_gd_axis.append(avg_cost)
        y_batch_gd_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("GD Optimization Finished!")





y_sgd_axis = []
y_sgd_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
batch_size = 1
with tf.Session() as sess:
    sess.run(init)
    y_sgd_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_sgd_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
            avg_cost += c / total_batch

        y_sgd_axis.append(avg_cost)
        y_sgd_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("Optimization Finished!")

y_sgd_axis = np.array(y_sgd_axis)
y_lbfgs_axis = np.array(y_lbfgs_axis)
y_adam_axis = np.array(y_adam_axis)
y_batch_gd_axis = np.array(y_batch_gd_axis)

y_sgd_axis1 = np.array(y_sgd_axis1)
y_lbfgs_axis1 = np.array(y_lbfgs_axis1)
y_adam_axis1 = np.array(y_adam_axis1)
y_batch_gd_axis1 = np.array(y_batch_gd_axis1)
x_axis = np.array(x_axis)
label_list = ['l-BFGS', 'Adam', 'mini_batch GD', 'SGD']

show_plot(x_axis, [y_lbfgs_axis, y_adam_axis, y_batch_gd_axis, y_sgd_axis],
                     label_list, 'training loss-epoch', 'epoch', 'loss', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'])
show_plot(x_axis, [y_lbfgs_axis1, y_adam_axis1, y_batch_gd_axis1, y_sgd_axis1],
                     label_list, 'test accuracy-epoch', 'epoch', 'accuracy', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'])