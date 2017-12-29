import tensorflow as tf
class softmax_zeros:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784]) 
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.build_forward() 
        self.build_loss()
        self.build_accuracy()
    def build_forward(self):

        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) # Softmax
    def build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.y, logits=self.pred)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class softmax_identity:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784]) 
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.build_forward() 
        self.build_loss()
        self.build_accuracy()
    def build_forward(self):

        # self.W = tf.get_variable(name='W',shape=[784, 10], initializer=tf.initializers.identity)
        self.W = tf.Variable(tf.ones([784, 10]))
        self.b = tf.Variable(tf.ones([10]))

        self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) # Softmax
    def build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.y, logits=self.pred)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class softmax_norm:
    def __init__(self, input_mean=0.0, input_stddev=1.0):
        self.x = tf.placeholder(tf.float32, [None, 784]) 
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.input_mean = input_mean
        self.input_stddev = input_stddev
        self.build_forward() 
        self.build_loss()
        self.build_accuracy()
    def build_forward(self):

        self.W = tf.get_variable(name='W',shape=[784, 10], 
                        initializer=tf.random_normal_initializer(mean=self.input_mean,stddev=self.input_stddev))
        self.b = tf.get_variable(name='b',shape=[10], 
                        initializer=tf.random_normal_initializer(mean=self.input_mean,stddev=self.input_stddev))


        self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) # Softmax
    def build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.y, logits=self.pred)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))