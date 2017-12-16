import tensorflow as tf
class basic_cnn:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784]) 
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.build_forward() 
        self.build_loss()
        self.build_accuracy()
    def build_forward(self):
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        #   dropout = tf.layers.dropout(
        #       inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        self.logits = tf.layers.dense(inputs=dense, units=10)
    def build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.y, logits=self.logits)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        