from __future__ import print_function
from cnn_base import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plot_test import *
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 1


#TODO : sgd
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
         
            avg_cost += c / total_batch
        print("Accuracy:", model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")






batch_size = 100


global_step = 0
x_axis = []
y_batch_gd_axis = []
# TODO: gd
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            if i%20 == 0:
                global_step+=1
                x_axis.append(global_step)
                y_batch_gd_axis.append(c)
            avg_cost += c / total_batch

        print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")





y_adam_axis = []
## TODO: Adam
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            if i%20 == 0:
                y_adam_axis.append(c)
            avg_cost += c / total_batch


        print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    
    print("Optimization Finished!")



y_adam_axis = np.array(y_adam_axis)
y_batch_gd_axis = np.array(y_batch_gd_axis)
x_axis = np.array(x_axis)
label_list = ['Adam', 'mini_batch GD']

show_plot(x_axis, [y_adam_axis, y_batch_gd_axis],
                     label_list, 'speed comparison', 'step', 'loss', ['#A6CEE3','#B2DF8A'])