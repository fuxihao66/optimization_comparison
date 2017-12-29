
from __future__ import print_function
from cnn_base import *
import tensorflow as tf
import numpy as np
from plot_test import *
from tensorflow.examples.tutorials.mnist import input_data

def cnn_batch_loss_comparison(input_optimizer, lr, label_name):
    mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

    learning_rate = lr
    training_epochs = 50
    batch_list = [10,100,1000,5500]





    model = basic_cnn()
    optimizer = input_optimizer(learning_rate).minimize(model.loss)

    init = tf.global_variables_initializer()
    batch_size = batch_list[0]
    global_step = 0
    x_axis = [0]
    y_1_axis = []
    y_1_axis1 = []
    with tf.Session() as sess:

        sess.run(init)
        y_1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        y_1_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                            model.y: batch_ys})
            
                avg_cost += c / total_batch

            global_step+=1
            x_axis.append(global_step)
            y_1_axis.append(avg_cost)
            y_1_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
            # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        
        print("Optimization Finished!")




    model = basic_cnn()
    optimizer = input_optimizer(learning_rate).minimize(model.loss)
    batch_size = batch_list[1]
    init = tf.global_variables_initializer()
    y_2_axis = []
    y_2_axis1 = []
    with tf.Session() as sess:

        sess.run(init)
        y_2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        y_2_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                            model.y: batch_ys})
            
                # Compute average loss
                avg_cost += c / total_batch
            y_2_axis.append(avg_cost)
            y_2_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
            # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        
        print("Optimization Finished!")


    model = basic_cnn()
    optimizer = input_optimizer(learning_rate).minimize(model.loss)
    batch_size = batch_list[2]
    init = tf.global_variables_initializer()
    y_3_axis = []
    y_3_axis1 = []
    # Start training
    with tf.Session() as sess:

        sess.run(init)
        y_3_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        y_3_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                            model.y: batch_ys})
            
                # Compute average loss
                avg_cost += c / total_batch
            y_3_axis.append(avg_cost)
            y_3_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
            # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        
        print("Optimization Finished!")



   
    model = basic_cnn()
    optimizer = input_optimizer(learning_rate).minimize(model.loss)
    batch_size = batch_list[3]
    init = tf.global_variables_initializer()
    y_4_axis = []
    y_4_axis1 = []
    # Start training
    with tf.Session() as sess:

        sess.run(init)
        y_4_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        y_4_axis.append(model.loss.eval({model.x: mnist.train.images[:1000],model.y: mnist.train.labels[:1000]}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x: batch_xs,
                                                            model.y: batch_ys})
            
                # Compute average loss
                avg_cost += c / total_batch
            y_4_axis.append(avg_cost)
            y_4_axis1.append(model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
            # print("Epoch{}".format(epoch+1), model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
        
        print("Optimization Finished!")

    y_1_axis = np.array(y_1_axis)
    y_2_axis = np.array(y_2_axis)
    y_3_axis = np.array(y_3_axis)
    y_4_axis = np.array(y_4_axis)

    y_1_axis1 = np.array(y_1_axis1)
    y_2_axis1 = np.array(y_2_axis1)
    y_3_axis1 = np.array(y_3_axis1)
    y_4_axis1 = np.array(y_4_axis1)
    x_axis = np.array(x_axis)
    label_list = ['batch_size=10', 'batch_size=100', 'batch_size=1000', 'batch_size=5500']

    show_plot(x_axis, [y_1_axis, y_2_axis, y_3_axis, y_4_axis],
                        label_list, 'CNN '+label_name+' training loss-epoch', 'epoch', 'loss', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'],name='loss.html')

    show_plot(x_axis, [y_1_axis1, y_2_axis1, y_3_axis1, y_4_axis1],
                        label_list, 'CNN '+label_name+' test accuracy-epoch', 'epoch', 'accuracy', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'])