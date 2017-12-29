from __future__ import print_function
import numpy as np
import tensorflow as tf
from plot_test import *
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

training_epochs = 50
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) 
y = tf.placeholder(tf.float32, [None, 10]) 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x, W) + b) 
cost = tf.losses.softmax_cross_entropy(
            onehot_labels=y, logits=pred)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############################  L-BFGS1   #########################

batch_size = 1000
iterations = 20
optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                cost,
                method='L-BFGS-B',
                options={'maxiter': iterations})
                # var_to_bounds={b: (-2, 7), W:(-1,6)})

init = tf.global_variables_initializer()
global_step = 0


x_axis = [0]
y_lbfgs1_axis = []
y_lbfgs1_axis1 = []
with tf.Session() as sess:

    sess.run(init)
    y_lbfgs1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_lbfgs1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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
        y_lbfgs1_axis.append(avg_cost)
        y_lbfgs1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("LBFGS Optimization Finished!")
############################  L-BFGS2   #########################
batch_size = 5500
iterations = 20
optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                cost,
                method='L-BFGS-B',
                options={'maxiter': iterations})
                # var_to_bounds={b: (-2, 7), W:(-1,6)})

init = tf.global_variables_initializer()
y_lbfgs2_axis1 = []
y_lbfgs2_axis = []
with tf.Session() as sess:

    sess.run(init)
    y_lbfgs2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_lbfgs2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            optimizer.minimize(sess, feed_dict={x: batch_xs,y: batch_ys})
            c = cost.eval({x: batch_xs,y: batch_ys})
            avg_cost += c / total_batch
    
        y_lbfgs2_axis.append(avg_cost)
        y_lbfgs2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("LBFGS Optimization Finished!")
############################  Adam1   #########################
batch_size = 100
learning_rate = 0.01
y_adam1_axis1 = []
y_adam1_axis = []
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    y_adam1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_adam1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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

        y_adam1_axis.append(avg_cost)
        y_adam1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Adam Optimization Finished!")
############################  Adam2   #########################
batch_size = 100
learning_rate = 0.001
y_adam2_axis = []
y_adam2_axis1 = []
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    y_adam2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_adam2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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

        y_adam2_axis.append(avg_cost)
        y_adam2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Adam Optimization Finished!")
############################  batch-gd1   #########################
learning_rate = 1
batch_size = 10
y_batch_gd1_axis = []
y_batch_gd1_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_batch_gd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_batch_gd1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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
                
        y_batch_gd1_axis.append(avg_cost)
        y_batch_gd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("GD Optimization Finished!")
############################  batch-gd2   #########################
learning_rate = 0.1
batch_size = 10
y_batch_gd2_axis = []
y_batch_gd2_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_batch_gd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_batch_gd2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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
                
        y_batch_gd2_axis.append(avg_cost)
        y_batch_gd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("GD Optimization Finished!")
############################  sgd1   #########################
batch_size = 1
learning_rate = 0.1
y_sgd1_axis1 = []
y_sgd1_axis = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_sgd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_sgd1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
            avg_cost += c / total_batch
        y_sgd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        y_sgd1_axis.append(avg_cost)
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("Optimization Finished!")
############################  sgd2   #########################
batch_size = 1
learning_rate = 0.01
y_sgd2_axis = []
y_sgd2_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_sgd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_sgd2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
            avg_cost += c / total_batch
        y_sgd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        y_sgd2_axis.append(avg_cost)
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("Optimization Finished!")
##########################################finished
y_lbfgs1_axis = np.array(y_lbfgs1_axis)
y_lbfgs2_axis = np.array(y_lbfgs2_axis)
y_adam1_axis = np.array(y_adam1_axis)
y_adam2_axis = np.array(y_adam2_axis)
y_batch_gd1_axis = np.array(y_batch_gd1_axis)
y_batch_gd2_axis = np.array(y_batch_gd2_axis)
y_sgd1_axis = np.array(y_sgd1_axis)
y_sgd2_axis = np.array(y_sgd2_axis)

y_lbfgs1_axis1 = np.array(y_lbfgs1_axis1)
y_lbfgs2_axis1 = np.array(y_lbfgs2_axis1)
y_adam1_axis1 = np.array(y_adam1_axis1)
y_adam2_axis1 = np.array(y_adam2_axis1)
y_batch_gd1_axis1 = np.array(y_batch_gd1_axis1)
y_batch_gd2_axis1 = np.array(y_batch_gd2_axis1)
y_sgd1_axis1 = np.array(y_sgd1_axis1)
y_sgd2_axis1 = np.array(y_sgd2_axis1)
x_axis = np.array(x_axis)
label_list = ['l-BFGS-1','l-BFGS-2','Adam-1', 'Adam-2','mini_batch GD-1', 'mini_batch GD-2','SGD-1','SGD-2']

show_plot(x_axis, [y_lbfgs1_axis,y_lbfgs2_axis,y_adam1_axis, y_adam2_axis, y_batch_gd1_axis, y_batch_gd2_axis, y_sgd1_axis,  y_sgd2_axis],
                     label_list, 'training loss-epoch', 'epoch', 'loss', 
                     ['#6ed1d1','#356666','#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'],name='loss.html')
show_plot(x_axis, [y_lbfgs1_axis1,y_lbfgs2_axis1,y_adam1_axis1, y_adam2_axis1, y_batch_gd1_axis1, y_batch_gd2_axis1, y_sgd1_axis1,  y_sgd2_axis1],
                     label_list, 'test accuracy-epoch', 'epoch', 'accuracy', 
                     ['#6ed1d1','#356666','#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'])
