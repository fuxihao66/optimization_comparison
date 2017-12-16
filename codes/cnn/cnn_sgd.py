from __future__ import print_function
from cnn_base import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/media/fuxihao/Data/MyDocuments/kaggle/dog/codes/MNIST-data", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 1
display_step = 1




model = basic_cnn()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    sess.run(init)

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
            
        # if (epoch+1) % display_step == 0:
        #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("Accuracy:", model.accuracy.eval({model.x: mnist.test.images, model.y: mnist.test.labels}))
    
    print("Optimization Finished!")