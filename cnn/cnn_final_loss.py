from __future__ import print_function
from cnn_base import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plot_test import *
import numpy as np
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

training_epochs = 50
############################   SGD1   ##############################

learning_rate = 0.01
batch_size = 1
global_step = 0
x_axis = [0]
y_sgd1_axis = []
y_sgd1_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_sgd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_sgd1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
         
            avg_cost += c / total_batch
        global_step+=1
        x_axis.append(global_step)
        y_sgd1_axis.append(avg_cost)
        y_sgd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")


############################   SGD2   ##############################

batch_size = 1
learning_rate = 0.001
y_sgd2_axis = []
y_sgd2_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_sgd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_sgd2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})

            avg_cost += c / total_batch
        y_sgd2_axis.append(avg_cost)
        y_sgd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")


############################   Adam1   ##############################

batch_size = 100
learning_rate = 0.01
y_adam1_axis = []
y_adam1_axis1 = []
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_adam1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_adam1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            avg_cost += c / total_batch

        y_adam1_axis.append(avg_cost)
        y_adam1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    
    print("Optimization Finished!")

############################   Adam2   ##############################
y_adam2_axis = []
y_adam2_axis1 = []
batch_size = 100
learning_rate = 0.001
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_adam2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_adam2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            avg_cost += c / total_batch

        y_adam2_axis.append(avg_cost)
        y_adam2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    
    print("Optimization Finished!")



############################   batch_GD1   ##############################
batch_size = 10
learning_rate = 0.1
y_batch_gd1_axis = []
y_batch_gd1_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_batch_gd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_batch_gd1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})

            avg_cost += c / total_batch
        y_batch_gd1_axis.append(avg_cost)
        y_batch_gd1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")
############################   batch_GD2   ##############################

batch_size = 100
learning_rate = 0.1
y_batch_gd2_axis = []
y_batch_gd2_axis1 = []
# TODO: gd
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

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

            avg_cost += c / total_batch
        y_batch_gd2_axis.append(avg_cost)
        y_batch_gd2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
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
x_axis = np.array(x_axis)
label_list = ['Adam-1', 'Adam-2','mini_batch GD-1', 'mini_batch GD-2','SGD-1','SGD-2']

show_plot(x_axis, [y_adam1_axis, y_adam2_axis, y_batch_gd1_axis, y_batch_gd2_axis, y_sgd1_axis,  y_sgd2_axis],
                     label_list, 'training loss-epoch', 'epoch', 'loss', 
                     ['#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'], name='loss.html')
show_plot(x_axis, [y_adam1_axis1, y_adam2_axis1, y_batch_gd1_axis1, y_batch_gd2_axis1, y_sgd1_axis1,  y_sgd2_axis1],
                     label_list, 'test accuracy-epoch', 'epoch', 'accuracy', 
                     ['#0202ff','#02027c','#ff0000', '#680000','#bf00bf','#5b005b'])