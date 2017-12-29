from __future__ import print_function
import numpy as np
import tensorflow as tf
from plot_test import *
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def softmax_batch_loss_comparison(input_optimizer, lr,label_name):
    mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)
    batch_size_list = [10,100,1000,5500]
    learning_rate = lr
    training_epochs = 50
    

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



    




    batch_size = batch_size_list[0]
       
    global_step = 0
    x_axis = [0]
    y_1_axis = []
    y_1_axis1 = []
    optimizer = input_optimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        y_1_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
        y_1_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            
                avg_cost += c / total_batch
            global_step +=1
            x_axis.append(global_step)
            y_1_axis.append(avg_cost)
            y_1_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
            print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
        print("Optimization Finished!")


    batch_size = batch_size_list[1]
    y_2_axis = []
    y_2_axis1 = []
    optimizer = input_optimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        y_2_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
        y_2_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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
            y_2_axis.append(avg_cost)
            y_2_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
            print("Epoch{}, ".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("Optimization Finished!")



    batch_size = batch_size_list[2]
    y_3_axis = []
    y_3_axis1 = []
    optimizer = input_optimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_3_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
        y_3_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
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
            y_3_axis.append(avg_cost)
            y_3_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
            print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
        print("Optimization Finished!")



    batch_size = batch_size_list[3]
    y_4_axis = []
    y_4_axis1 = []
    optimizer = input_optimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y_4_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
        y_4_axis.append(cost.eval({x: mnist.train.images[:1000],y: mnist.train.labels[:1000]}))
        for epoch in range(training_epochs):
            avg_cost = 0.
            
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                            y: batch_ys})
                
                avg_cost += c / total_batch
            y_4_axis.append(avg_cost)
            y_4_axis1.append(accuracy.eval({  x: mnist.test.images,   y: mnist.test.labels}))
            print("Epoch{}".format(epoch+1), accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
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
    label_list = ['batch size = 10', 'batch size = 100', 'batch size = 1000', 'batch size = 5500']

    show_plot(x_axis, [y_1_axis, y_2_axis, y_3_axis, y_4_axis],
                        label_list, 'softmax '+label_name+' training loss-epoch', 'epoch', 'loss', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'])

    show_plot(x_axis, [y_1_axis1, y_2_axis1, y_3_axis1, y_4_axis1],
                        label_list, 'softmax '+label_name+' test accuracy-epoch', 'epoch', 'accuracy', ['#6ed1d1','#0202ff','#ff0000', '#bf00bf'])