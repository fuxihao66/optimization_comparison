from __future__ import print_function
from cnn_base import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plot_test import *
import time
import numpy as np
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

training_epochs = 50
############################   SGD1   ##############################

learning_rate = 0.01
batch_size = 1
global_step = 0
x_axis_1 = [0]
y_sgd1_axis = []
y_sgd1_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_sgd1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    y_sgd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i% 500 == 0:
                x_axis_1.append(total_time)
                y_sgd1_axis.append(c)
                y_sgd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
                print(i)
        # print("Accuracy:", model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")


############################   SGD2   ##############################

batch_size = 1
learning_rate = 0.001
y_sgd2_axis = []
y_sgd2_axis1 = []
x_axis_2 = [0]
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_sgd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_sgd2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i % 500 == 0:
                x_axis_2.append(total_time)
                y_sgd2_axis.append(c)
                y_sgd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
                print(i)
        # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")


############################   Adam1   ##############################

batch_size = 100
learning_rate = 0.01
x_axis_3 = [0]
y_adam1_axis = []
y_adam1_axis1 = []
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_adam1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_adam1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            x_axis_3.append(total_time)
            y_adam1_axis.append(c)
            y_adam1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")

############################   Adam2   ##############################
y_adam2_axis = []
y_adam2_axis1 = []
x_axis_4 = [0]
batch_size = 100
learning_rate = 0.001
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_adam2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_adam2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            x_axis_4.append(total_time)
            y_adam2_axis.append(c)
            y_adam2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")



############################   batch_GD1   ##############################
batch_size = 10
learning_rate = 0.1
x_axis_5 = [0]
y_batch_gd1_axis = []
y_batch_gd1_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_batch_gd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_batch_gd1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time

            if i%50 == 0:
                x_axis_5.append(total_time)
                y_batch_gd1_axis.append(c)
                y_batch_gd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")
############################   batch_GD2   ##############################

batch_size = 100
learning_rate = 0.1
x_axis_6 = [0]
y_batch_gd2_axis = []
y_batch_gd2_axis1 = []
# TODO: gd
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    total_time = 0.0
    sess.run(init)
    y_batch_gd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_batch_gd2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})

            start_time = time.time()
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            used_time = time.time()-start_time
            total_time+= used_time
            
            x_axis_6.append(total_time)
            y_batch_gd2_axis.append(c)
            y_batch_gd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(total_time)
    print("Optimization Finished!")





y_adam1_axis = np.array(y_adam1_axis)
y_adam2_axis = np.array(y_adam2_axis)
y_batch_gd1_axis = np.array(y_batch_gd1_axis)
y_batch_gd2_axis = np.array(y_batch_gd2_axis)
y_sgd1_axis = np.array(y_sgd1_axis)
y_sgd2_axis = np.array(y_sgd2_axis)

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
label_list = ['Adam-1', 'Adam-2','mini_batch GD-1', 'mini_batch GD-2','SGD-1','SGD-2']

show_plot_time([x_axis_3,x_axis_4,x_axis_5,x_axis_6,x_axis_1,x_axis_2], 
                     [y_adam1_axis, y_adam2_axis, y_batch_gd1_axis, y_batch_gd2_axis, y_sgd1_axis,  y_sgd2_axis],
                     label_list, 'training loss-time', 'time', 'loss', 
                     ['#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'],name='loss.html')
show_plot_time([x_axis_3,x_axis_4,x_axis_5,x_axis_6,x_axis_1,x_axis_2], 
                     [y_adam1_axis1, y_adam2_axis1, y_batch_gd1_axis1, y_batch_gd2_axis1, y_sgd1_axis1,  y_sgd2_axis1],
                     label_list, 'test accuracy-time', 'time', 'accuracy', 
                     ['#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'])