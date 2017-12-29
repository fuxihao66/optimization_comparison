from __future__ import print_function
from cnn_base import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from plot_test import *
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

learning_rate = 0.01
training_epochs = 50
batch_size = 1

global_step = 0
x_axis = [0]
y_sgd_axis = []
y_sgd_axis1 = []
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_sgd_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_sgd_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
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
        y_sgd_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        y_sgd_axis.append(avg_cost)
        print("Accuracy:", model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")






batch_size = 100
y_batch_gd_axis = []
y_batch_gd_axis1 = []
# TODO: gd
model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_batch_gd_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_batch_gd_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})

            avg_cost += c / total_batch
        y_batch_gd_axis.append(avg_cost)
        y_batch_gd_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    print("Optimization Finished!")





y_adam_axis = []
y_adam_axis1 = []
## TODO: Adam
model = basic_cnn()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    y_adam_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    y_adam_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                          model.y: batch_ys})
            avg_cost += c / total_batch

        y_adam_axis.append(avg_cost)
        y_adam_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    
    print("Optimization Finished!")



y_adam_axis = np.array(y_adam_axis)
y_batch_gd_axis = np.array(y_batch_gd_axis)
y_sgd_axis = np.array(y_sgd_axis)
y_adam_axis1 = np.array(y_adam_axis1)
y_batch_gd_axis1 = np.array(y_batch_gd_axis1)
y_sgd_axis1 = np.array(y_sgd_axis1)
x_axis = np.array(x_axis)
label_list = ['Adam', 'mini_batch GD', 'SGD']

show_plot(x_axis, [y_adam_axis, y_batch_gd_axis, y_sgd_axis],
                     label_list, 'training loss-epoch', 'epoch', 'loss', ['#0202ff','#ff0000', '#bf00bf'])
show_plot(x_axis, [y_adam_axis1, y_batch_gd_axis1, y_sgd_axis1],
                     label_list, 'test accuracy-epoch', 'epoch', 'accuracy', ['#0202ff','#ff0000', '#bf00bf'])