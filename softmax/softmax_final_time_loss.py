from __future__ import print_function
import numpy as np
import tensorflow as tf
from plot_test import *
import time

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


x_axis_1 = [0]
y_lbfgs1_axis1 = []
y_lbfgs1_axis = []
with tf.Session() as sess:

    sess.run(init)
    y_lbfgs1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_lbfgs1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    total_time = 0.0
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            start_time = time.time()
            optimizer.minimize(sess, feed_dict={x: batch_xs,y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
    
            x_axis_1.append(total_time)
            c = cost.eval({x: batch_xs,y: batch_ys})            
            y_lbfgs1_axis.append(c)
            y_lbfgs1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
 
        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))   
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
x_axis_2 = [0]
y_lbfgs2_axis = []
y_lbfgs2_axis1 = []
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_lbfgs2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_lbfgs2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            start_time = time.time()
            optimizer.minimize(sess, feed_dict={x: batch_xs,y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            x_axis_2.append(total_time)
            c = cost.eval({x: batch_xs,y: batch_ys})
            y_lbfgs2_axis.append(c)
            y_lbfgs2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("LBFGS Optimization Finished!")
############################  Adam1   #########################
batch_size = 100
learning_rate = 0.01
x_axis_3 = [0]
y_adam1_axis = []
y_adam1_axis1 = []
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_adam1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_adam1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            

            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            x_axis_3.append(total_time)
            y_adam1_axis.append(c)
            y_adam1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
      

        
        # print("Epoch{}, ".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Adam Optimization Finished!")
############################  Adam2   #########################
batch_size = 100
learning_rate = 0.001
x_axis_4 = [0]
y_adam2_axis = []
y_adam2_axis1 = []
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_adam2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_adam2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            

            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            x_axis_4.append(total_time)
            y_adam2_axis.append(c)
            y_adam2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
        # print("Epoch{}, ".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Adam Optimization Finished!")
############################  batch-gd1   #########################
learning_rate = 1
batch_size = 10
x_axis_5 = [0]
y_batch_gd1_axis = []
y_batch_gd1_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_batch_gd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_batch_gd1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            

            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i%50 == 0:
                x_axis_5.append(total_time)
                y_batch_gd1_axis.append(c)
                y_batch_gd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("GD Optimization Finished!")
############################  batch-gd2   #########################
learning_rate = 0.1
batch_size = 10
x_axis_6 = [0]
y_batch_gd2_axis = []
y_batch_gd2_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_batch_gd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_batch_gd2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            if i%50 == 0:
                x_axis_6.append(total_time)
                y_batch_gd2_axis.append(c)
                y_batch_gd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    

        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("GD Optimization Finished!")
############################  sgd1   #########################
batch_size = 1
learning_rate = 0.1
x_axis_7 = [0]
y_sgd1_axis = []
y_sgd1_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_sgd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_sgd1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i % 500 == 0:
                x_axis_7.append(total_time)
                y_sgd1_axis.append(c)
                y_sgd1_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    print("Optimization Finished!")
############################  sgd2   #########################
batch_size = 1
learning_rate = 0.01
x_axis_8 = [0]
y_sgd2_axis = []
y_sgd2_axis1 = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_sgd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    y_sgd2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            start_time = time.time()
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i%500 == 0:
                x_axis_8.append(total_time)
                y_sgd2_axis.append(c)
                y_sgd2_axis1.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
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
x_axis_1 = np.array(x_axis_1)
x_axis_2 = np.array(x_axis_2)
x_axis_3 = np.array(x_axis_3)
x_axis_4 = np.array(x_axis_4)
x_axis_5 = np.array(x_axis_5)
x_axis_6 = np.array(x_axis_6)
x_axis_7 = np.array(x_axis_7)
x_axis_8 = np.array(x_axis_8)
label_list = ['l-BFGS-1','l-BFGS-2','Adam-1', 'Adam-2','mini_batch GD-1', 'mini_batch GD-2','SGD-1','SGD-2']

show_plot_time([x_axis_1,x_axis_2,x_axis_3,x_axis_4,x_axis_5,x_axis_6,x_axis_7,x_axis_8],
                     [y_lbfgs1_axis,y_lbfgs2_axis,y_adam1_axis, y_adam2_axis, y_batch_gd1_axis, y_batch_gd2_axis, y_sgd1_axis,  y_sgd2_axis],
                     label_list, 'training loss-time', 'time', 'loss', 
                     ['#6ed1d1','#356666','#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'], name='loss.html')
show_plot_time([x_axis_1,x_axis_2,x_axis_3,x_axis_4,x_axis_5,x_axis_6,x_axis_7,x_axis_8],
                     [y_lbfgs1_axis1,y_lbfgs2_axis1,y_adam1_axis1, y_adam2_axis1, y_batch_gd1_axis1, y_batch_gd2_axis1, y_sgd1_axis1,  y_sgd2_axis1],
                     label_list, 'test accuracy-time', 'time', 'accuracy', 
                     ['#6ed1d1','#356666','#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'])
